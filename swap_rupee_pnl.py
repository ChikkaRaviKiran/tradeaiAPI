"""Rupee P&L for 3 lots NIFTY + 3 lots SENSEX using the SWAP schedule.

Lot sizes (2026): NIFTY=75, SENSEX=20.
Position: short ATM straddle (1 call + 1 put per lot).
P&L rupees = pnl_pct/100 * spot_at_entry * lot_size * num_lots
"""
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from optimise_entry_exit_roll import (
    simulate_day, INTRADAY_PATH, CACHE_PATH, IV,
    INDEX_FOR_WEEKDAY_SWAP, get_dte, time_str_to_minutes,
)

NIFTY_LOT, SENSEX_LOT = 75, 20
NUM_LOTS = 3

# Per-DTE best entry/exit (from earlier optimisation)
BEST_TIMES = {
    1: ("09:20", "15:15"),  # 1-DTE
    2: ("09:30", "15:15"),  # 2-DTE
    3: ("09:30", "15:00"),  # 3-DTE
    4: ("10:15", "14:30"),  # 4-DTE
}

intraday = json.loads(INTRADAY_PATH.read_text())
cache    = json.loads(CACHE_PATH.read_text())
nd, sd   = cache["nifty_daily"], cache["sensex_daily"]

per_day = []  # list of (date, sym, dte, pnl_pct, spot, rupees)

for d_str in sorted(set(nd) | set(sd)):
    wd = datetime.strptime(d_str, "%Y-%m-%d").weekday()
    if wd not in INDEX_FOR_WEEKDAY_SWAP:
        continue
    sym = INDEX_FOR_WEEKDAY_SWAP[wd]
    bars = intraday.get(sym, {}).get(d_str, [])
    daily = (nd if sym == "NIFTY" else sd).get(d_str)
    if not daily or len(bars) < 30:
        continue
    dte = get_dte(d_str, sym)
    if dte not in BEST_TIMES:
        continue
    entry, exit_t = BEST_TIMES[dte]
    out = simulate_day(bars, sym, entry, exit_t, None, dte, IV)
    if out["skipped"]:
        continue
    # spot at entry
    em = time_str_to_minutes(entry)
    ei = next(i for i, b in enumerate(bars) if time_str_to_minutes(b[0]) >= em)
    spot_e = bars[ei][4]
    lot = NIFTY_LOT if sym == "NIFTY" else SENSEX_LOT
    rupees = out["cum_pnl_pct"] / 100 * spot_e * lot * NUM_LOTS
    per_day.append((d_str, sym, dte, out["cum_pnl_pct"], spot_e, rupees))

print(f"3 lots NIFTY (lot={NIFTY_LOT}) + 3 lots SENSEX (lot={SENSEX_LOT}) — SWAP schedule\n")
print(f"Total days traded: {len(per_day)}")

# ── Day-by-day ────────────────────────────────────────────────
print("\n── Day-by-day P&L ──")
print(f"{'date':12} {'day':3} {'sym':6} {'DTE':>3} {'spot':>9} {'pnl%':>7} {'rupees':>10}")
running = 0
for d_str, sym, dte, pct, spot, rup in per_day:
    wd_name = ["Mon","Tue","Wed","Thu","Fri"][datetime.strptime(d_str,"%Y-%m-%d").weekday()]
    running += rup
    print(f"{d_str:12} {wd_name:3} {sym:6} {dte:>3} {spot:>9.1f} {pct:>+6.3f}% {rup:>+10,.0f}")

# ── Monthly aggregates ────────────────────────────────────────
monthly = defaultdict(lambda: {"n": 0, "wins": 0, "rupees": 0.0, "pct_sum": 0.0})
for d_str, sym, dte, pct, spot, rup in per_day:
    m = d_str[:7]
    monthly[m]["n"] += 1
    monthly[m]["rupees"] += rup
    monthly[m]["pct_sum"] += pct
    if rup > 0:
        monthly[m]["wins"] += 1

print("\n── Monthly summary ──")
print(f"{'month':9} {'days':>5} {'wins':>5} {'avg/day':>10} {'cum P&L':>13} {'cum %':>7}")
total_rup = 0
total_pct = 0
total_n = 0
total_w = 0
for m in sorted(monthly):
    s = monthly[m]
    print(f"{m:9} {s['n']:>5} {s['wins']:>5} {s['rupees']/s['n']:>+9,.0f} "
          f"{s['rupees']:>+12,.0f} {s['pct_sum']:>+6.2f}%")
    total_rup += s["rupees"]
    total_pct += s["pct_sum"]
    total_n   += s["n"]
    total_w   += s["wins"]

print(f"{'TOTAL':9} {total_n:>5} {total_w:>5} {total_rup/total_n:>+9,.0f} "
      f"{total_rup:>+12,.0f} {total_pct:>+6.2f}%")
print(f"\nWin rate: {total_w}/{total_n} ({total_w/total_n*100:.0f}%)")
print(f"Avg per trading day: ₹{total_rup/total_n:+,.0f}")
months_span = len(monthly)
print(f"Avg per month ({months_span} months): ₹{total_rup/months_span:+,.0f}")

# ── Best / worst days ─────────────────────────────────────────
per_day_sorted = sorted(per_day, key=lambda x: x[5])
print("\n── 5 worst days ──")
for d_str, sym, dte, pct, spot, rup in per_day_sorted[:5]:
    print(f"  {d_str} {sym} {dte}-DTE  {pct:+.3f}%  ₹{rup:+,.0f}")
print("\n── 5 best days ──")
for d_str, sym, dte, pct, spot, rup in per_day_sorted[-5:][::-1]:
    print(f"  {d_str} {sym} {dte}-DTE  {pct:+.3f}%  ₹{rup:+,.0f}")
