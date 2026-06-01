"""Check monthly expiry day P&L separately."""
import json
from datetime import datetime
from calendar import monthrange
from optimise_entry_exit_roll import simulate_day, INTRADAY_PATH, IV, time_str_to_minutes

c = json.load(open(INTRADAY_PATH))
months = ['2026-01','2026-02','2026-03','2026-04','2026-05']

monthly_days = []  # (date, sym)
for ym in months:
    y, m = map(int, ym.split('-'))
    last = monthrange(y, m)[1]
    for d in range(last, 0, -1):
        if datetime(y, m, d).weekday() == 1:
            monthly_days.append((f"{ym}-{d:02d}", "NIFTY"))
            break
    for d in range(last, 0, -1):
        if datetime(y, m, d).weekday() == 3:
            monthly_days.append((f"{ym}-{d:02d}", "SENSEX"))
            break

print("Monthly expiry days (always 0-DTE):")
print(f"{'date':12} {'sym':6} {'inCache':>8}  09:20->15:15  09:30->10:00 (early)")
tot = {"hold": 0, "early": 0, "n": 0}
for d, sym in monthly_days:
    bars = c.get(sym, {}).get(d, [])
    if len(bars) < 30:
        print(f"  {d} {sym}  NO DATA")
        continue
    r_hold  = simulate_day(bars, sym, "09:20", "15:15", None, 0, IV)
    r_early = simulate_day(bars, sym, "09:30", "10:00", None, 0, IV)
    if r_hold["skipped"] or r_early["skipped"]:
        print(f"  {d} {sym}  SKIP")
        continue
    print(f"  {d} {sym:6} {'Y':>8}  {r_hold['cum_pnl_pct']:>+7.3f}%      {r_early['cum_pnl_pct']:>+7.3f}%")
    tot["hold"]  += r_hold["cum_pnl_pct"]
    tot["early"] += r_early["cum_pnl_pct"]
    tot["n"] += 1

n = tot["n"]
print(f"\nAvg over {n} monthly expiries: hold-to-close = {tot['hold']/n:+.3f}%  early-exit = {tot['early']/n:+.3f}%")
print(f"Cumulative: hold = {tot['hold']:+.2f}%  early = {tot['early']:+.2f}%")
