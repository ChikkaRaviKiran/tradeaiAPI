"""Why are expiry days (0-DTE) bad despite max theta? And does early exit help?

Tests on the 31 0-DTE days (NIFTY Tue, SENSEX Thu) in the cache:
  - Multiple early-exit times: 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00, 13:30, 14:00, 14:30, 15:15
  - Two entries: 09:20 (open) and 09:30
  - Decomposes P&L into "theta collected" vs "gamma paid"
"""
import json, math
from datetime import datetime
from pathlib import Path

from optimise_entry_exit_roll import (
    straddle_value, round_strike, time_str_to_minutes,
    INTRADAY_PATH, CACHE_PATH, IV,
)

# 0-DTE mapping: Tue=NIFTY, Thu=SENSEX
ZERO_DTE_MAP = {1: "NIFTY", 3: "SENSEX"}

intraday = json.loads(INTRADAY_PATH.read_text())
cache    = json.loads(CACHE_PATH.read_text())
nd, sd   = cache["nifty_daily"], cache["sensex_daily"]

days = []
for d_str in sorted(set(nd) | set(sd)):
    wd = datetime.strptime(d_str, "%Y-%m-%d").weekday()
    if wd not in ZERO_DTE_MAP:
        continue
    sym = ZERO_DTE_MAP[wd]
    daily = (nd if sym == "NIFTY" else sd).get(d_str)
    bars  = intraday.get(sym, {}).get(d_str, [])
    if not daily or len(bars) < 30:
        continue
    days.append((d_str, sym, bars, daily))

print(f"Loaded {len(days)} 0-DTE days\n")


def sim(bars, sym, entry, exit_t, dte=0):
    em = time_str_to_minutes(entry)
    xm = time_str_to_minutes(exit_t)
    open_m = 9 * 60 + 15
    day_min = (15 * 60 + 30) - open_m
    ei = next((i for i, b in enumerate(bars) if time_str_to_minutes(b[0]) >= em), None)
    xi = next((i for i in range(len(bars) - 1, -1, -1)
               if time_str_to_minutes(bars[i][0]) <= xm), None)
    if ei is None or xi is None or xi <= ei:
        return None
    spot_e = bars[ei][4]
    K = round_strike(spot_e, sym)
    credit = straddle_value(spot_e, K, dte, 0.0, IV)
    spot_x = bars[xi][4]
    t_frac = (time_str_to_minutes(bars[xi][0]) - em) / day_min
    debit  = straddle_value(spot_x, K, dte, t_frac, IV)
    move_pct = abs(spot_x - spot_e) / spot_e * 100
    return {
        "pnl_pct": (credit - debit) / spot_e * 100,
        "intrinsic_at_exit": abs(spot_x - K) / spot_e * 100,
        "move_pct": move_pct,
        "credit_pct": credit / spot_e * 100,
        "debit_pct":  debit  / spot_e * 100,
    }


# ─── Q1: Why bad? Decompose ─────────────────────────────────
print("─── Q1: Decomposition (entry 09:20, hold to 15:15) ───")
print(f"{'date':12} {'sym':6} {'entry%':>7} {'exit%':>7} {'pnl%':>7} {'move%':>7} {'intr%':>7}")
totals = {"pnl": 0, "credit": 0, "debit": 0, "n": 0, "wins": 0, "huge_moves": 0}
for d_str, sym, bars, _ in days:
    r = sim(bars, sym, "09:20", "15:15")
    if not r:
        continue
    totals["pnl"]    += r["pnl_pct"]
    totals["credit"] += r["credit_pct"]
    totals["debit"]  += r["debit_pct"]
    totals["n"]      += 1
    if r["pnl_pct"] > 0:
        totals["wins"] += 1
    if r["move_pct"] > 0.5:
        totals["huge_moves"] += 1
    print(f"{d_str:12} {sym:6} {r['credit_pct']:>+6.3f}% {r['debit_pct']:>+6.3f}%"
          f" {r['pnl_pct']:>+6.3f}% {r['move_pct']:>6.3f}% {r['intrinsic_at_exit']:>6.3f}%")

n = totals["n"]
print(f"\nSummary: n={n} wins={totals['wins']}/{n} ({totals['wins']/n*100:.0f}%)")
print(f"  Theta collected (avg credit at open):   {totals['credit']/n:+.3f}%")
print(f"  Paid to close   (avg debit at exit):    {totals['debit']/n:+.3f}%")
print(f"  Net P&L (avg):                          {totals['pnl']/n:+.3f}%")
print(f"  Days where |spot move| > 0.5%:          {totals['huge_moves']}/{n} ({totals['huge_moves']/n*100:.0f}%)")
print(f"  → Gamma loss (intrinsic at exit) eats theta on big-move days\n")


# ─── Q2: Does early exit help? Sweep all exit times ─────────
print("─── Q2: Early-exit sweep on 0-DTE days ───")
exits = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
         "13:00", "13:30", "14:00", "14:30", "15:00", "15:15"]
for entry in ["09:20", "09:30"]:
    print(f"\n  Entry {entry}:")
    print(f"  {'exit':>5} {'n':>3} {'win%':>5} {'avg%':>7} {'cum%':>7} {'min%':>7} {'max%':>7} {'std%':>6}")
    for x in exits:
        if time_str_to_minutes(x) <= time_str_to_minutes(entry):
            continue
        pnls = []
        for d_str, sym, bars, _ in days:
            r = sim(bars, sym, entry, x)
            if r is not None:
                pnls.append(r["pnl_pct"])
        if not pnls:
            continue
        wins = sum(1 for p in pnls if p > 0)
        avg = sum(pnls) / len(pnls)
        std = math.sqrt(sum((p - avg) ** 2 for p in pnls) / len(pnls))
        print(f"  {x:>5} {len(pnls):>3} {wins/len(pnls)*100:>4.0f}%"
              f" {avg:>+6.3f}% {sum(pnls):>+6.2f}% {min(pnls):>+6.3f}% {max(pnls):>+6.3f}% {std:>5.3f}%")
