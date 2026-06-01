#!/usr/bin/env python3
"""Phase-3b breakdown: trade BOTH NIFTY and SENSEX every weekday at their best windows.
Uses PCP-derived ATM + numeric-expiry parsing (honest results)."""
from collections import defaultdict
from datetime import datetime
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   find_trading_days, load_chain, simulate,
                                   LOT_SIZE, LOTS, STRIKE_STEP)

# Phase-3b: each (weekday, sym) → (entry, exit) from best-bucket sweep
SCHEDULE = {
    ("Mon","NIFTY"):  ("09:20","14:00"),
    ("Mon","SENSEX"): ("09:20","12:30"),
    ("Tue","NIFTY"):  ("09:20","15:15"),
    ("Tue","SENSEX"): ("09:20","11:30"),
    ("Wed","NIFTY"):  ("09:20","14:30"),
    ("Wed","SENSEX"): ("09:20","15:00"),
    ("Thu","NIFTY"):  ("09:20","15:00"),
    ("Thu","SENSEX"): ("09:20","12:30"),
    ("Fri","NIFTY"):  ("09:45","14:30"),
    ("Fri","SENSEX"): ("09:20","12:30"),
}

def main():
    near = find_trading_days()
    print(f"Total (date, sym) combos: {len(near)}")
    trades = []
    for (d, sym), (ex, exp_s) in sorted(near.items()):
        wd = d.strftime("%a")
        if (wd, sym) not in SCHEDULE: continue
        entry, exit_t = SCHEDULE[(wd, sym)]
        dte = (ex - d).days
        bars = load_chain(d, sym, exp_s)
        if not bars: continue
        pnl, atm = simulate(bars, sym, entry, exit_t)
        if pnl is None: continue
        trades.append((d, wd, sym, dte, entry, exit_t, atm, pnl))

    print(f"Matched trades: {len(trades)}\n")

    # Per-day combined PnL (both indexes added)
    by_day = defaultdict(list)
    for t in trades: by_day[t[0]].append(t)
    print("══════ PER-DAY (both indexes) ══════")
    print(f"{'Date':12} {'WD':4}  N-PnL        S-PnL        Day-Total      Cum")
    cum = 0
    for d in sorted(by_day):
        wd = d.strftime("%a")
        n_pnl = sum(t[7] for t in by_day[d] if t[2]=="NIFTY")
        s_pnl = sum(t[7] for t in by_day[d] if t[2]=="SENSEX")
        tot = n_pnl + s_pnl
        cum += tot
        print(f"{str(d):12} {wd:4}  {fmt_rs(n_pnl):>12} {fmt_rs(s_pnl):>12} {fmt_rs(tot):>12}  {fmt_rs(cum)}")

    print("\n══════ MONTHLY TOTALS ══════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for t in trades:
        m = t[0].strftime("%Y-%m")
        by_m[m]["pnl"] += t[7]; by_m[m]["n"] += 1
        if t[7]>0: by_m[m]["w"] += 1
    print(f"{'Month':10} {'Trades':>7} {'Wins':>5} {'Win%':>5}  {'PnL':>13}")
    g_pnl=g_n=g_w=0
    for m in sorted(by_m):
        v = by_m[m]; g_pnl += v["pnl"]; g_n += v["n"]; g_w += v["w"]
        print(f"{m:10} {v['n']:>7} {v['w']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>13}")
    print(f"{'TOTAL':10} {g_n:>7} {g_w:>5} {int(100*g_w/g_n):>4}%  {fmt_rs(g_pnl):>13}")

    print("\n══════ BY (WEEKDAY, SYM) ══════")
    by_k = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for t in trades:
        k = (t[1], t[2])
        by_k[k]["pnl"] += t[7]; by_k[k]["n"] += 1
        if t[7]>0: by_k[k]["w"] += 1
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            k=(wd,sym); v=by_k.get(k)
            if not v or v["n"]==0: continue
            e,x = SCHEDULE[k]
            print(f"  {wd} {sym:6} {e}→{x}  n={v['n']:2d}  win={int(100*v['w']/v['n']):3d}%  cum={fmt_rs(v['pnl'])}  avg={fmt_rs(v['pnl']/v['n'])}")

if __name__ == "__main__":
    main()
