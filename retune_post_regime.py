#!/usr/bin/env python3
"""Retune for POST-regime data (Apr 15 – May 22).
Sweeps entry/exit windows AND tests nearest vs next-week expiry per (weekday, sym).
Goal: find a schedule that makes May profitable.
"""
from collections import defaultdict
from datetime import datetime
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, simulate,
                                   LOT_SIZE, LOTS, STRIKE_STEP)

START = "2026-04-15"   # post-regime window
END   = "2026-05-22"

ENTRIES = ["09:20","09:30","09:45","10:00","10:30","11:00"]
EXITS   = ["10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]

def find_days_with_expiries():
    """Return {(date, sym): [(expiry_date, expiry_str), ...]} sorted by date."""
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
    for k, lst in by_day.items():
        lst.sort()
        out[k] = lst
    return out

def main():
    days = find_days_with_expiries()
    print(f"Post-regime (date, sym) combos: {len(days)}\n")

    # Try both NEAREST and NEXT expiry per day
    # Build trades: per (wd, sym, expiry_rank) sweep windows
    # expiry_rank: 0=nearest, 1=next
    bucket = defaultdict(list)  # (wd, sym, rank) -> [(d, dte, bars), ...]

    for (d, sym), exps in sorted(days.items()):
        wd = d.strftime("%a")
        for rank, (ex, exp_s) in enumerate(exps[:2]):  # only nearest + next
            dte = (ex - d).days
            bars = load_chain(d, sym, exp_s)
            if not bars: continue
            bucket[(wd, sym, rank, dte)].append((d, bars))

    # Show bucket summary
    print("Buckets (>=3 days):")
    for k in sorted(bucket.keys()):
        if len(bucket[k]) >= 3:
            wd,sym,rank,dte = k
            print(f"  {wd} {sym:6} rank={rank} dte={dte:2d}  n={len(bucket[k])}")
    print()

    # Sweep windows per bucket
    print("══════ BEST window per (WD, SYM, rank, dte) — post-regime only ══════")
    best = {}
    for k, days_list in sorted(bucket.items()):
        if len(days_list) < 3: continue
        wd, sym, rank, dte = k
        results = []
        for e in ENTRIES:
            for x in EXITS:
                if to_min(x) <= to_min(e): continue
                pnls = []
                for d, bars in days_list:
                    pnl, _ = simulate(bars, sym, e, x)
                    if pnl is not None: pnls.append(pnl)
                if len(pnls) < len(days_list) * 0.7: continue
                cum = sum(pnls); w = sum(1 for p in pnls if p>0); n = len(pnls)
                results.append((cum, e, x, n, w, min(pnls)))
        if not results: continue
        results.sort(reverse=True)
        cum, e, x, n, w, mn = results[0]
        best[k] = (cum, e, x, n, w, mn)
        tag = "NEAREST" if rank==0 else "NEXT-WK"
        print(f"  {wd} {sym:6} {tag} dte={dte:2d}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10s}  worst={fmt_rs(mn)}")
    print()

    # For each (wd, sym), pick the BEST option (could be nearest or next-wk)
    print("══════ RECOMMENDED post-regime schedule (one index/day) ══════")
    per_wd_sym = defaultdict(list)
    for (wd,sym,rank,dte), v in best.items():
        per_wd_sym[(wd,sym)].append((v[0], rank, dte, v))   # sort by cum desc
    for (wd,sym), opts in sorted(per_wd_sym.items()):
        opts.sort(reverse=True)
        cum, rank, dte, (_, e, x, n, w, mn) = opts[0]
        tag = "NEAREST" if rank==0 else "NEXT-WK"
        print(f"  {wd} {sym:6}  use {tag} (dte={dte:2d})  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum)}")
    print()

    # Pick the SINGLE best sym per weekday (Phase-3a style)
    print("══════ NEW PHASE-3a — one index per day (post-regime) ══════")
    by_wd = defaultdict(list)
    for (wd,sym,rank,dte), v in best.items():
        by_wd[wd].append((v[0], sym, rank, dte, v))
    grand = 0; total_n = 0; total_w = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        opts = by_wd.get(wd, [])
        if not opts: continue
        opts.sort(reverse=True)
        cum, sym, rank, dte, (_, e, x, n, w, mn) = opts[0]
        tag = "near" if rank==0 else "next"
        print(f"  {wd}: {sym:6} {tag}wk dte={dte:2d}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10s}  worst={fmt_rs(mn)}")
        grand += cum; total_n += n; total_w += w
    print(f"\n  POST-REGIME TOTAL: {fmt_rs(grand)}  over {total_n} trades  win={int(100*total_w/total_n)}%")

if __name__ == "__main__":
    main()
