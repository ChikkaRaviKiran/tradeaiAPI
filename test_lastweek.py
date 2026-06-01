#!/usr/bin/env python3
"""Filter Phase-3a trades to LAST WEEK of each month (monthly expiry week)."""
from collections import defaultdict
from datetime import datetime, date
from calendar import monthrange
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   find_trading_days, load_chain, simulate,
                                   LOT_SIZE, LOTS)

# POST-REGIME schedule (current dashboard)
SCHEDULE = {
    "Mon": ("NIFTY",  "09:20", "15:15"),
    "Tue": ("NIFTY",  "09:30", "13:30"),
    "Wed": ("SENSEX", "10:00", "15:15"),
    "Thu": ("SENSEX", "09:20", "11:30"),
    "Fri": ("NIFTY",  "09:45", "14:30"),
}

# OLD schedule for comparison
OLD_SCHEDULE = {
    "Mon": ("NIFTY",  "09:20", "14:00"),
    "Tue": ("NIFTY",  "09:20", "15:15"),
    "Wed": ("SENSEX", "09:20", "15:00"),
    "Thu": ("SENSEX", "09:20", "12:30"),
    "Fri": ("SENSEX", "09:20", "12:30"),
}

def is_last_week(d):
    """True if d is in the LAST 7 days of its month (i.e. monthly-expiry week)."""
    last_day = monthrange(d.year, d.month)[1]
    return (last_day - d.day) <= 6

def is_last_3days(d):
    last_day = monthrange(d.year, d.month)[1]
    return (last_day - d.day) <= 3

def run_schedule(name, schedule, near_exp, filter_fn=None):
    print(f"\n══════ {name} ══════")
    trades = []
    for (d, sym), (ex, exp_s) in sorted(near_exp.items()):
        wd = d.strftime("%a")
        if wd not in schedule: continue
        plan_sym, entry, exit_t = schedule[wd]
        if sym != plan_sym: continue
        if filter_fn and not filter_fn(d): continue
        bars = load_chain(d, sym, exp_s)
        if not bars: continue
        pnl, atm = simulate(bars, sym, entry, exit_t)
        if pnl is None: continue
        trades.append((d, wd, sym, entry, exit_t, atm, pnl))

    if not trades:
        print("  No trades.")
        return
    print(f"{'Date':12} {'WD':4} {'Sym':7} {'ATM':>7} {'Window':14}  {'PnL':>12}  Cum")
    cum = 0
    for d, wd, sym, e, x, atm, pnl in trades:
        cum += pnl
        print(f"{str(d):12} {wd:4} {sym:7} {atm:>7} {e}→{x}  {fmt_rs(pnl):>12}  {fmt_rs(cum)}")

    # monthly
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for d, wd, sym, e, x, atm, pnl in trades:
        m = d.strftime("%Y-%m")
        by_m[m]["pnl"] += pnl; by_m[m]["n"] += 1
        if pnl>0: by_m[m]["w"] += 1
    print(f"\n  Monthly summary:")
    print(f"  {'Month':10} {'Days':>5} {'Win%':>5}  {'PnL':>13}")
    g = gn = gw = 0
    for m in sorted(by_m):
        v = by_m[m]; g += v["pnl"]; gn += v["n"]; gw += v["w"]
        w = int(100*v["w"]/v["n"]) if v["n"] else 0
        print(f"  {m:10} {v['n']:>5} {w:>4}%  {fmt_rs(v['pnl']):>13}")
    w = int(100*gw/gn) if gn else 0
    print(f"  {'TOTAL':10} {gn:>5} {w:>4}%  {fmt_rs(g):>13}")

def main():
    near = find_trading_days()

    # Full data, post-regime schedule (already known)
    run_schedule("ALL DAYS — POST-REGIME schedule", SCHEDULE, near)

    # Last week of month only — both schedules
    run_schedule("LAST WEEK OF MONTH only — POST-REGIME schedule",
                 SCHEDULE, near, filter_fn=is_last_week)
    run_schedule("LAST WEEK OF MONTH only — OLD schedule",
                 OLD_SCHEDULE, near, filter_fn=is_last_week)

    # Last 3 trading days only
    run_schedule("LAST 3 DAYS OF MONTH only — POST-REGIME schedule",
                 SCHEDULE, near, filter_fn=is_last_3days)

if __name__ == "__main__":
    main()
