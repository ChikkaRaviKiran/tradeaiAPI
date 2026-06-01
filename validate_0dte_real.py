#!/usr/bin/env python3
"""
Test trading 0-DTE on its OWN expiry day (no SWAP) vs SWAP scenario.

Pulls every 0-DTE day in 2026 from DB, sweeps entry/exit grid for ATM
straddle short, reports best per symbol.

Note: NIFTY weekly expiry was Thursday until ~mid-April 2026, then Tuesday.
      SENSEX weekly expiry was Friday in Jan-Mar 2026, became Thursday in April.
"""
import subprocess
from collections import defaultdict
from datetime import datetime

LOTS = 3
LOT_SIZE = {"NIFTY": 75, "SENSEX": 20}
STRIKE_STEP = {"NIFTY": 50, "SENSEX": 100}

ENTRIES = ["09:20","09:30","09:45","10:00","10:15","10:30","10:45","11:00","11:30"]
EXITS = ["10:00","10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]


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


def find_0dte_days():
    rows = psql("""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '2026-01-01' AND date <= '2026-05-22' ORDER BY date, instrument;""")
    days = []
    seen = set()
    for date_s, inst, exp_s in rows:
        try: d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except Exception: continue
        ex = parse_expiry(exp_s)
        if ex is None or ex != d: continue
        key = (d, inst)
        if key in seen: continue
        seen.add(key)
        days.append((d, inst, exp_s))
    return days


def load_bars(d, sym, exp_s):
    spot_rows = psql(f"""SELECT spot FROM option_candles
                         WHERE instrument='{sym}' AND date='{d}' AND expiry='{exp_s}'
                           AND timestamp BETWEEN '{d} 09:18:00' AND '{d} 09:25:00'
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
    return (c-d, (c-d)*LOT_SIZE[sym]*LOTS, c, d)


def main():
    print("Finding all 0-DTE days in 2026 DB...")
    days = find_0dte_days()
    print(f"Found {len(days)} (date, instrument) 0-DTE combos\n")
    by_sym_wd = defaultdict(lambda: defaultdict(int))
    for d, sym, _ in days: by_sym_wd[sym][d.strftime("%a")] += 1
    for sym, wds in by_sym_wd.items():
        print(f"  {sym} 0-DTE by weekday: {dict(wds)}")
    print()

    print("Loading bars...")
    loaded = []
    for i,(d,sym,exp_s) in enumerate(days,1):
        bars, spot = load_bars(d, sym, exp_s)
        if not bars or not spot:
            print(f"  [{i}/{len(days)}] {d} {sym}: no data")
            continue
        loaded.append((d,sym,exp_s,bars,spot))
        if i%10==0: print(f"  [{i}/{len(days)}] loaded")
    print(f"Usable 0-DTE days: {len(loaded)}\n")

    best_by_sym = {}
    for target_sym in ["NIFTY","SENSEX"]:
        ds = [(d,sym,bars,spot) for d,sym,_,bars,spot in loaded if sym==target_sym]
        if not ds:
            print(f"── {target_sym} 0-DTE: no usable days\n"); continue
        print(f"── {target_sym} 0-DTE ({len(ds)} days) ──")
        print("  rk entry  exit   n  win%       cum ₹    avg ₹     min ₹     max ₹")
        results = []
        for e in ENTRIES:
            for x in EXITS:
                if to_min(x) <= to_min(e): continue
                pnls = []
                for d,sym,bars,spot in ds:
                    r = simulate(bars, spot, sym, e, x)
                    if r: pnls.append(r[1])
                if len(pnls) < len(ds)*0.7: continue
                n=len(pnls); w=sum(1 for p in pnls if p>0); cum=sum(pnls); avg=cum/n
                results.append((cum,e,x,n,w,avg,min(pnls),max(pnls)))
        results.sort(reverse=True)
        for rk,(cum,e,x,n,w,avg,mn,mx) in enumerate(results[:5],1):
            print(f"  {rk:2d} {e} {x}  {n:2d}  {int(100*w/n):3d}%  {fmt_rs(cum):>10s}  {fmt_rs(avg):>6s}  {fmt_rs(mn):>8s}  {fmt_rs(mx):>8s}")
        full = next(((cum,e,x,n,w,avg,mn,mx) for cum,e,x,n,w,avg,mn,mx in results if e=="09:20" and x=="15:15"), None)
        if full:
            cum,e,x,n,w,avg,mn,mx = full
            print(f"  FULL-DAY 09:20→15:15: n={n} win={int(100*w/n)}% cum={fmt_rs(cum)} avg={fmt_rs(avg)}")
        if results: best_by_sym[target_sym] = results[0]
        print()

    print("══════ NO-SWAP 0-DTE TOTAL ══════")
    total = 0; total_n = 0; total_w = 0
    for sym, r in best_by_sym.items():
        cum,e,x,n,w,_,_,_ = r
        print(f"  {sym}: best {e}→{x}  n={n}  win={int(100*w/n)}%  cum={fmt_rs(cum)}")
        total+=cum; total_n+=n; total_w+=w
    print(f"  TOTAL 0-DTE (best per sym): n={total_n} wins={total_w}/{total_n} cum={fmt_rs(total)}\n")

    print("══════ WORST 0-DTE FULL-DAY HOLDS (09:20→15:15) ══════")
    full_pnls = []
    for d,sym,_,bars,spot in loaded:
        r = simulate(bars, spot, sym, "09:20", "15:15")
        if r: full_pnls.append((r[1], d, sym))
    full_pnls.sort()
    for pnl,d,sym in full_pnls[:10]: print(f"  {d} {sym}  {fmt_rs(pnl)}")
    print(f"\nFull-day total: {fmt_rs(sum(p for p,_,_ in full_pnls))}  ({len(full_pnls)} days, {sum(1 for p,_,_ in full_pnls if p>0)} wins)")


if __name__=="__main__":
    main()
