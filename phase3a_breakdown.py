#!/usr/bin/env python3
"""Phase-3a daily PnL breakdown: print per-trade row + monthly totals."""
import subprocess
from collections import defaultdict
from datetime import datetime
from phase3_tune import (psql, parse_expiry, round_strike, to_min, fmt_rs,
                         load_bars, simulate, LOT_SIZE, STRIKE_STEP, LOTS)

# Phase-3a schedule: one index/day, REAL DTE, optimised window
SCHEDULE = {
    "Mon": ("NIFTY",  3, "09:20", "14:00"),
    "Tue": ("NIFTY",  2, "09:20", "15:15"),
    "Wed": ("SENSEX", 2, "09:20", "15:00"),
    "Thu": ("SENSEX", 1, "09:20", "12:30"),
    "Fri": ("SENSEX", 0, "09:20", "12:30"),
}

def find_trading_days():
    rows = psql("""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '2026-01-01' AND date <= '2026-05-22' ORDER BY date, instrument;""")
    by_day = defaultdict(list)
    for date_s, inst, exp_s in rows:
        try: d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except Exception: continue
        ex = parse_expiry(exp_s)
        if ex is None or ex < d: continue
        by_day[(d, inst)].append((ex, exp_s))
    result = {}
    for k, lst in by_day.items():
        lst.sort()
        result[k] = lst[0]
    return result

def main():
    near = find_trading_days()
    trades = []  # (date, wd, sym, dte, entry, exit, pnl, win)
    skipped = 0
    for (d, sym), (ex, exp_s) in sorted(near.items()):
        wd = d.strftime("%a")
        if wd not in SCHEDULE: continue
        plan_sym, plan_dte, entry, exit_t = SCHEDULE[wd]
        if sym != plan_sym: continue
        actual_dte = (ex - d).days
        bars, spot = load_bars(d, sym, exp_s)
        if not bars or not spot:
            if d.month >= 4: print(f"  [skip-bars] {d} {sym} exp={exp_s}")
            continue
        pnl = simulate(bars, spot, sym, entry, exit_t)
        if pnl is None:
            if d.month >= 4: print(f"  [skip-sim ] {d} {sym} exp={exp_s} dte={actual_dte}")
            continue
        trades.append((d, wd, sym, actual_dte, entry, exit_t, pnl))

    print(f"\nSchedule-matched days: {len(trades)}  (skipped {skipped} for DTE mismatch)\n")
    print("══════ PER-DAY PnL ══════")
    print(f"{'Date':12} {'WD':4} {'Sym':7} {'DTE':3} {'Window':14}  {'PnL':>12}  Cum")
    cum = 0
    for d, wd, sym, dte, e, x, pnl in trades:
        cum += pnl
        print(f"{str(d):12} {wd:4} {sym:7} {dte:3d} {e}→{x}  {fmt_rs(pnl):>12}  {fmt_rs(cum)}")

    print("\n══════ MONTHLY TOTALS ══════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for d, wd, sym, dte, e, x, pnl in trades:
        key = d.strftime("%Y-%m")
        by_m[key]["pnl"] += pnl
        by_m[key]["n"]   += 1
        by_m[key]["w"]   += 1 if pnl>0 else 0
    print(f"{'Month':10} {'Days':>5} {'Wins':>5} {'Win%':>5}  {'PnL':>13}  {'Avg/day':>10}")
    grand=0; n_tot=0; w_tot=0
    for m in sorted(by_m):
        v = by_m[m]; grand += v["pnl"]; n_tot += v["n"]; w_tot += v["w"]
        print(f"{m:10} {v['n']:>5} {v['w']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>13}  {fmt_rs(v['pnl']/v['n']):>10}")
    print(f"{'TOTAL':10} {n_tot:>5} {w_tot:>5} {int(100*w_tot/n_tot):>4}%  {fmt_rs(grand):>13}  {fmt_rs(grand/n_tot):>10}")

    print("\n══════ BY WEEKDAY ══════")
    by_wd = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for d, wd, sym, dte, e, x, pnl in trades:
        by_wd[wd]["pnl"] += pnl; by_wd[wd]["n"] += 1
        if pnl>0: by_wd[wd]["w"] += 1
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        v = by_wd[wd]
        if v["n"]==0: continue
        sym, dte, e, x = SCHEDULE[wd]
        print(f"  {wd} {sym:6} DTE{dte}  n={v['n']:2d}  win={int(100*v['w']/v['n']):3d}%  cum={fmt_rs(v['pnl'])}  avg={fmt_rs(v['pnl']/v['n'])}")

if __name__ == "__main__":
    main()
