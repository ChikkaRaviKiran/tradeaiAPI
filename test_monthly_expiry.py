#!/usr/bin/env python3
"""Test MONTHLY expiry straddles vs weekly.
For each day, use the expiry with DTE between 15-40 (the monthly contract).
Sweep entry/exit windows per (weekday, sym).
"""
from collections import defaultdict
from datetime import datetime
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, simulate, LOT_SIZE, LOTS)

START = "2026-01-01"
END   = "2026-05-22"

ENTRIES = ["09:20","09:30","09:45","10:00","10:30","11:00"]
EXITS   = ["10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]

DTE_MIN = 10   # minimum DTE to qualify as "monthly"
DTE_MAX = 45

def find_monthly_days():
    """Per (date, sym), pick the expiry with DTE in [DTE_MIN, DTE_MAX]."""
    rows = psql(f"""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '{START}' AND date <= '{END}' ORDER BY date, instrument;""")
    by_day = defaultdict(list)
    for date_s, inst, exp_s in rows:
        try: d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except Exception: continue
        ex = parse_expiry(exp_s)
        if ex is None or ex < d: continue
        by_day[(d, inst)].append((ex, exp_s))
    out = {}
    skipped = 0
    for k, lst in by_day.items():
        lst.sort()
        # pick expiry with DTE in window
        d = k[0]
        candidates = [(ex, exp_s, (ex-d).days) for ex, exp_s in lst
                      if DTE_MIN <= (ex-d).days <= DTE_MAX]
        if not candidates:
            skipped += 1
            continue
        # nearest one inside the window (so monthly contract, not next-month)
        candidates.sort(key=lambda x: x[2])
        out[k] = candidates[0]
    print(f"Days with monthly contract available: {len(out)}  (skipped {skipped} with no monthly in {DTE_MIN}-{DTE_MAX} DTE)")
    return out

def main():
    monthly = find_monthly_days()
    print(f"\nLoading bars for {len(monthly)} days (monthly expiry)...")
    loaded = []
    for i, ((d, sym), (ex, exp_s, dte)) in enumerate(sorted(monthly.items()), 1):
        bars = load_chain(d, sym, exp_s)
        if bars:
            loaded.append((d, sym, dte, exp_s, bars))
        if i % 20 == 0:
            print(f"  [{i}/{len(monthly)}] loaded={len(loaded)}")
    print(f"Usable: {len(loaded)}\n")

    # Group by (weekday, sym)
    by_bucket = defaultdict(list)
    for d, sym, dte, exp_s, bars in loaded:
        wd = d.strftime("%a")
        by_bucket[(wd, sym)].append((d, dte, exp_s, bars))

    print("Buckets (>=5 days):")
    for k in sorted(by_bucket.keys()):
        days = by_bucket[k]
        if len(days) >= 5:
            dtes = sorted({x[1] for x in days})
            wd, sym = k
            print(f"  {wd} {sym:6}  n={len(days):2d}  DTEs={dtes}")
    print()

    # Sweep best window per (weekday, sym)
    print("══════ BEST window per (WD, SYM) — MONTHLY expiry ══════")
    best = {}
    for k in sorted(by_bucket.keys()):
        days = by_bucket[k]
        if len(days) < 5: continue
        wd, sym = k
        results = []
        for e in ENTRIES:
            for x in EXITS:
                if to_min(x) <= to_min(e): continue
                pnls = []
                for d, dte, exp_s, bars in days:
                    pnl, _ = simulate(bars, sym, e, x)
                    if pnl is not None: pnls.append(pnl)
                if len(pnls) < len(days) * 0.7: continue
                cum = sum(pnls); w = sum(1 for p in pnls if p>0); n = len(pnls)
                results.append((cum, e, x, n, w, min(pnls), max(pnls)))
        if not results: continue
        results.sort(reverse=True)
        cum, e, x, n, w, mn, mx = results[0]
        best[k] = (cum, e, x, n, w, mn, mx)
        print(f"  {wd} {sym:6}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10s}  avg={fmt_rs(cum/n):>7s}  worst={fmt_rs(mn):>8s}  best={fmt_rs(mx)}")
    print()

    # Pick best sym per weekday
    print("══════ MONTHLY-EXPIRY PHASE-3a (one index/day) ══════")
    by_wd = defaultdict(list)
    for (wd,sym), v in best.items():
        by_wd[wd].append((v[0], sym, v))
    grand = 0; ntot = 0; wtot = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        opts = by_wd.get(wd, [])
        if not opts: continue
        opts.sort(reverse=True)
        cum, sym, (_, e, x, n, w, mn, mx) = opts[0]
        print(f"  {wd}: {sym:6}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10s}  worst={fmt_rs(mn)}")
        grand += cum; ntot += n; wtot += w
    print(f"\n  MONTHLY 3a TOTAL: {fmt_rs(grand)}  over {ntot} trades  win={int(100*wtot/ntot)}%")
    months = 5  # approx
    print(f"  ~per month: {fmt_rs(grand/months)}")

    # Monthly breakdown
    print("\n══════ MONTH-BY-MONTH (best per (WD,SYM) applied to each day) ══════")
    # Replay: for each loaded day, run the best window for its (wd,sym)
    by_month = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for d, sym, dte, exp_s, bars in loaded:
        wd = d.strftime("%a")
        k = (wd, sym)
        if k not in best: continue
        # Only Phase-3a primary
        wd_opts = by_wd.get(wd, [])
        if not wd_opts: continue
        wd_opts.sort(reverse=True)
        if wd_opts[0][1] != sym: continue   # this sym isn't primary today
        cum, _, (_, e, x, _, _, _, _) = wd_opts[0]
        pnl, _ = simulate(bars, sym, e, x)
        if pnl is None: continue
        m = d.strftime("%Y-%m")
        by_month[m]["pnl"] += pnl; by_month[m]["n"] += 1
        if pnl>0: by_month[m]["w"] += 1
    print(f"{'Month':10} {'Days':>5} {'Win%':>5}  {'PnL':>13}")
    g = 0; gn = 0
    for m in sorted(by_month):
        v = by_month[m]
        g += v["pnl"]; gn += v["n"]
        win = int(100*v["w"]/v["n"]) if v["n"] else 0
        print(f"{m:10} {v['n']:>5} {win:>4}%  {fmt_rs(v['pnl']):>13}")
    print(f"{'TOTAL':10} {gn:>5}         {fmt_rs(g):>13}")

if __name__ == "__main__":
    main()
