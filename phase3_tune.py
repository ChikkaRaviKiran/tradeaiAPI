#!/usr/bin/env python3
"""
Phase-3: Test trading BOTH NIFTY *and* SENSEX every weekday.

Currently Phase-2 SWAP trades ONE index per day:
   Mon=NIFTY  Tue=SENSEX  Wed=SENSEX  Thu=NIFTY  Fri=NIFTY

Phase-3 idea: also trade the OTHER index on each day at its best window.
   Mon: NIFTY (1-DTE) + SENSEX (3- or 4-DTE far-leg)
   Tue: SENSEX (2-DTE) + NIFTY (0-DTE morning OR 7-DTE far-leg)
   Wed: SENSEX (1-DTE) + NIFTY (6-DTE far-leg)
   Thu: NIFTY (0-DTE Jan-Mar / 5-DTE Apr+) + SENSEX (1- or 0-DTE)
   Fri: NIFTY (4- or 5-DTE) + SENSEX (0-DTE Jan-Mar / 6-DTE Apr+)

For each (date, instrument) in 2026 DB, sweep entry/exit grid and pick
the best window. Then total per weekday-symbol-combo for the BOTH-INDEXES
schedule and compare to Phase-2.
"""
import subprocess
from collections import defaultdict
from datetime import datetime, date, timedelta

LOTS = 3
LOT_SIZE = {"NIFTY": 75, "SENSEX": 20}
STRIKE_STEP = {"NIFTY": 50, "SENSEX": 100}
ENTRIES = ["09:20","09:30","09:45","10:00","10:15","10:30","11:00"]
EXITS = ["10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]


def psql(sql):
    cmd = ["docker","exec","-i","tradeai-postgres","psql","-U","tradeai","-d","tradeai","-t","-A","-F","\t","-c",sql]
    return [line.split("\t") for line in subprocess.check_output(cmd, text=True).strip().split("\n") if line.strip()]


def parse_expiry(s):
    try: return datetime.strptime(s, "%d%b%y").date()
    except Exception: return None


def round_strike(spot, sym):
    step = STRIKE_STEP[sym]
    return int(round(spot/step)*step)


def to_min(t):
    h,m=t.split(":"); return int(h)*60+int(m)


def fmt_rs(x):
    return f"₹{'+' if x>=0 else '-'}{abs(int(x)):,}"


def find_trading_days():
    """For each (date, sym) find the nearest valid expiry >= date in DB."""
    rows = psql("""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '2026-01-01' AND date <= '2026-05-22' ORDER BY date, instrument;""")
    # Group expiries per (date, sym)
    by_day = defaultdict(list)
    for date_s, inst, exp_s in rows:
        try: d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except Exception: continue
        ex = parse_expiry(exp_s)
        if ex is None or ex < d: continue
        by_day[(d, inst)].append((ex, exp_s))
    # Pick NEAREST expiry per day-sym
    result = {}
    for k, lst in by_day.items():
        lst.sort()
        result[k] = lst[0]   # (expiry_date, expiry_str) — the nearest
    return result


def load_bars(d, sym, exp_s):
    spot_rows = psql(f"""SELECT spot FROM option_candles
                         WHERE instrument='{sym}' AND date='{d}' AND expiry='{exp_s}'
                           AND timestamp BETWEEN '{d} 09:18:00' AND '{d} 09:30:00'
                           AND spot IS NOT NULL ORDER BY timestamp LIMIT 1;""")
    if not spot_rows or not spot_rows[0][0]: return None, None
    spot_open = float(spot_rows[0][0])
    atm = round_strike(spot_open, sym); step = STRIKE_STEP[sym]
    strikes = [atm-step, atm, atm+step]
    rows = psql(f"""SELECT strike, option_type, timestamp, close, spot FROM option_candles
                    WHERE instrument='{sym}' AND date='{d}' AND expiry='{exp_s}'
                      AND strike IN ({','.join(str(s) for s in strikes)}) ORDER BY timestamp;""")
    bars = defaultdict(dict); spot_by_min = {}
    for strike, opt, ts, close, spot in rows:
        try:
            dt = datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
            minute = dt.hour*60+dt.minute
            bars[(int(strike),opt)][minute] = float(close)
            if spot: spot_by_min[minute] = float(spot)
        except Exception: continue
    return bars, spot_by_min


def get_at(bars, strike, opt, t_min):
    s = bars.get((strike,opt))
    if not s: return None
    for m in range(t_min, t_min+5):
        if m in s: return s[m]
    return None


def straddle_at(bars, atm, t_min):
    ce = get_at(bars, atm, "CE", t_min); pe = get_at(bars, atm, "PE", t_min)
    return None if ce is None or pe is None else ce+pe


def simulate(bars, spot_by_min, sym, entry, exit_t):
    e_min, x_min = to_min(entry), to_min(exit_t)
    spot_open = None
    for m in range(e_min, e_min+5):
        if m in spot_by_min: spot_open = spot_by_min[m]; break
    if spot_open is None: return None
    atm = round_strike(spot_open, sym)
    c = straddle_at(bars, atm, e_min); d = straddle_at(bars, atm, x_min)
    if c is None or d is None: return None
    return (c-d) * LOT_SIZE[sym] * LOTS


def main():
    print("Loading all (date, sym) combos in 2026...")
    near_exp = find_trading_days()
    print(f"Found {len(near_exp)} (date, sym) entries\n")

    # Load bars for everything
    loaded = {}
    items = list(near_exp.items())
    for i, ((d,sym), (ex, exp_s)) in enumerate(items, 1):
        bars, spot = load_bars(d, sym, exp_s)
        if bars and spot:
            loaded[(d,sym)] = (bars, spot, ex, exp_s, (ex - d).days)
        if i % 20 == 0:
            print(f"  [{i}/{len(items)}] loaded; usable={len(loaded)}")
    print(f"Usable (date, sym) days: {len(loaded)}\n")

    # Group by (weekday, sym, dte)
    by_bucket = defaultdict(list)
    for (d, sym), (bars, spot, ex, exp_s, dte) in loaded.items():
        wd = d.strftime("%a")
        by_bucket[(wd, sym, dte)].append((d, sym, bars, spot))

    print("Bucket counts (>=5 days only):")
    print("  WD  SYM    DTE  count")
    for k in sorted(by_bucket.keys()):
        if len(by_bucket[k]) >= 5:
            wd, sym, dte = k
            print(f"  {wd}  {sym:6s}  {dte:2d}   {len(by_bucket[k])}")
    print()

    # Best entry/exit per (weekday, sym, dte) bucket — only buckets with >=5 days
    print("══════ BEST entry/exit per (WD, SYM, DTE) bucket ══════")
    best_bucket = {}
    for k in sorted(by_bucket.keys()):
        days = by_bucket[k]
        if len(days) < 5: continue
        wd, sym, dte = k
        results = []
        for e in ENTRIES:
            for x in EXITS:
                if to_min(x) <= to_min(e): continue
                pnls = []
                for d, s, bars, spot in days:
                    r = simulate(bars, spot, s, e, x)
                    if r is not None: pnls.append(r)
                if len(pnls) < len(days) * 0.7: continue
                n=len(pnls); w=sum(1 for p in pnls if p>0); cum=sum(pnls); avg=cum/n
                results.append((cum, e, x, n, w, avg, min(pnls), max(pnls)))
        if not results: continue
        results.sort(reverse=True)
        best = results[0]
        cum,e,x,n,w,avg,mn,mx = best
        best_bucket[k] = best
        print(f"  {wd} {sym:6s} DTE{dte:2d}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10s}  avg={fmt_rs(avg):>7s}  min={fmt_rs(mn):>8s}")
    print()

    # Build PHASE-2 baseline (one index per day, picking best DTE per weekday-sym)
    # For each weekday, pick the (sym, dte) that gives highest cum
    best_per_wd = {}
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        cands = [(cum, sym, dte, e, x, n, w) for (w2,sym,dte), (cum,e,x,n,wc,avg,mn,mx) in best_bucket.items()
                 for cum,e,x,n,w in [(cum,e,x,n,wc)] if w2==wd]
        if cands:
            cands.sort(reverse=True)
            best_per_wd[wd] = cands[0]

    print("══════ ONE-INDEX-PER-DAY (best sym/DTE per weekday) ══════")
    total_one = 0
    for wd, (cum, sym, dte, e, x, n, w) in best_per_wd.items():
        print(f"  {wd}: {sym} DTE{dte} {e}→{x}  n={n}  win={int(100*w/n)}%  cum={fmt_rs(cum)}")
        total_one += cum
    print(f"  TOTAL one-index: {fmt_rs(total_one)}\n")

    # PHASE-3: Trade BOTH indexes every day at their best (sym, dte) bucket
    print("══════ BOTH-INDEXES per weekday (best NIFTY + best SENSEX) ══════")
    total_both = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        line = f"  {wd}: "
        wd_total = 0
        for sym in ["NIFTY","SENSEX"]:
            cands = [(cum, dte, e, x, n, w) for (w2,s2,dte), (cum,e,x,n,wc,avg,mn,mx) in best_bucket.items()
                     for cum,e,x,n,w in [(cum,e,x,n,wc)] if w2==wd and s2==sym]
            if cands:
                cands.sort(reverse=True)
                cum, dte, e, x, n, w = cands[0]
                line += f"{sym} DTE{dte} {e}→{x} (n={n} win={int(100*w/n)}% {fmt_rs(cum)})  "
                wd_total += cum
        line += f"=> {fmt_rs(wd_total)}"
        print(line)
        total_both += wd_total
    print(f"  TOTAL both-indexes: {fmt_rs(total_both)}\n")

    print("══════ FINAL COMPARISON ══════")
    print(f"  Phase-2 (one index/day, weekday-labeled DTE)    : ~₹+2,99,038  (prior result)")
    print(f"  Phase-3a (one index/day, REAL-DTE optimised)    : {fmt_rs(total_one)}")
    print(f"  Phase-3b (BOTH indexes/day, REAL-DTE optimised) : {fmt_rs(total_both)}")
    print(f"  Phase-3b uplift vs Phase-3a: {fmt_rs(total_both - total_one)}")


if __name__ == "__main__":
    main()
