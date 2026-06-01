"""Optimise entry time, exit time, and roll threshold for ATM short straddle.

Uses the Bachelier intraday model + 5-min bars from intraday_full_day_cache.json.

Per-day simulation
------------------
1. At entry_time t0: short ATM straddle at strike K0 = round(spot(t0))
   Initial credit = V(spot(t0), K0, dte, 0, IV)

2. Walk forward bar-by-bar to exit_time:
   - At each bar, compute current straddle value V(spot, K_current, dte, t_frac, IV)
   - If |spot - K_current| / spot > roll_threshold AND rolls_done < MAX_ROLLS:
       a) close current straddle at current value (booking loss/gain)
       b) re-open ATM straddle at new K = round(spot)
       c) increment rolls_done; reset internal clock for fresh straddle
   - At exit_time: close current straddle at value V_exit

   Net P&L = sum( credit_received_per_leg - debit_paid_per_leg ) over all rolls

3. Results aggregated per DTE bucket and overall.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, date, time as dtime
from pathlib import Path

IV = 0.17
THETA_FRACTION_PER_DAY = 6.25 / 6.25  # 1 trading day = 6h15m of theta
MAX_ROLLS = 2

INTRADAY_PATH = Path(__file__).parent / "intraday_full_day_cache.json"
CACHE_PATH = Path(__file__).parent / "tune_skipday_cache.json"

INDEX_FOR_WEEKDAY = {0: "NIFTY", 1: "NIFTY", 2: "SENSEX", 3: "SENSEX", 4: "NIFTY"}
# SWAP scenario: on Tue swap NIFTY (0-DTE)→SENSEX (2-DTE),
#                on Thu swap SENSEX (0-DTE)→NIFTY (3-DTE weekly).
INDEX_FOR_WEEKDAY_SWAP = {0: "NIFTY", 1: "SENSEX", 2: "SENSEX", 3: "NIFTY", 4: "NIFTY"}
DTE_NIFTY  = {0: 1, 1: 0, 3: 3, 4: 4}   # Mon=1, Tue=0, Thu=3 (next Tue expiry), Fri=4
DTE_SENSEX = {1: 2, 2: 1, 3: 0}          # Tue=2 (Thu expiry), Wed=1, Thu=0


def get_dte(d_str: str, sym: str) -> int:
    wd = datetime.strptime(d_str, "%Y-%m-%d").weekday()
    return (DTE_NIFTY if sym == "NIFTY" else DTE_SENSEX).get(wd, 1)


def build_days(intraday, cache, mapping):
    """Build day list per the given weekday→index mapping."""
    days = []
    nd = cache["nifty_daily"]
    sd = cache["sensex_daily"]
    for d_str in sorted(set(nd) | set(sd)):
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        wd = dt.weekday()
        if wd not in mapping:
            continue
        sym = mapping[wd]
        daily = (nd if sym == "NIFTY" else sd).get(d_str)
        bars  = intraday.get(sym, {}).get(d_str, [])
        if not daily or len(bars) < 30:
            continue
        dte = get_dte(d_str, sym)
        days.append((d_str, sym, dte, bars))
    return days


def straddle_value(spot: float, strike: float, dte_days: float,
                   t_frac_consumed: float, iv: float) -> float:
    t_remaining = max(dte_days - t_frac_consumed, 0.05)
    atm_part = spot * iv * math.sqrt(t_remaining / 252) * 0.8
    intrinsic = abs(spot - strike)
    return math.sqrt(intrinsic ** 2 + atm_part ** 2)


def round_strike(spot: float, sym: str) -> float:
    step = 50 if sym == "NIFTY" else 100
    return round(spot / step) * step


def time_str_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def simulate_day(bars: list, sym: str, entry: str, exit_t: str,
                 roll_threshold_pct: float | None, dte: int, iv: float) -> dict:
    """Returns dict with cum_pnl_pct, rolls_done."""
    entry_min = time_str_to_minutes(entry)
    exit_min  = time_str_to_minutes(exit_t)
    open_min  = 9 * 60 + 15  # market open
    day_minutes = (15 * 60 + 30) - open_min  # 6h15m = 375

    # Find entry bar (first bar at or after entry time)
    entry_idx = next((i for i, b in enumerate(bars)
                      if time_str_to_minutes(b[0]) >= entry_min), None)
    exit_idx  = next((i for i in range(len(bars) - 1, -1, -1)
                      if time_str_to_minutes(bars[i][0]) <= exit_min), None)
    if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
        return {"cum_pnl_pct": 0.0, "rolls_done": 0, "skipped": True}

    spot_entry_for_pct = bars[entry_idx][4]

    # Initial straddle
    current_strike = round_strike(bars[entry_idx][4], sym)
    leg_entry_time = entry_min
    leg_entry_spot = bars[entry_idx][4]
    v_at_open = straddle_value(leg_entry_spot, current_strike, dte, 0.0, iv)
    cum_credit = v_at_open  # We collect this premium on short
    cum_debit  = 0.0
    rolls_done = 0

    for i in range(entry_idx + 1, exit_idx + 1):
        bar_min = time_str_to_minutes(bars[i][0])
        spot    = bars[i][4]
        t_frac  = (bar_min - leg_entry_time) / day_minutes
        v_now   = straddle_value(spot, current_strike, dte, t_frac, iv)

        # Check roll condition
        drift_pct = abs(spot - current_strike) / spot * 100
        should_roll = (roll_threshold_pct is not None
                       and drift_pct > roll_threshold_pct
                       and rolls_done < MAX_ROLLS
                       and i < exit_idx)  # don't roll on the exit bar

        if should_roll:
            # Close the existing straddle
            cum_debit += v_now
            # Open a new ATM straddle
            current_strike = round_strike(spot, sym)
            leg_entry_time = bar_min
            leg_entry_spot = spot
            v_new = straddle_value(spot, current_strike, dte, 0.0, iv)
            cum_credit += v_new
            rolls_done += 1

    # Final close at exit_idx
    bar_min = time_str_to_minutes(bars[exit_idx][0])
    spot    = bars[exit_idx][4]
    t_frac  = (bar_min - leg_entry_time) / day_minutes
    v_exit  = straddle_value(spot, current_strike, dte, t_frac, iv)
    cum_debit += v_exit

    net_pnl = cum_credit - cum_debit
    return {
        "cum_pnl_pct": net_pnl / spot_entry_for_pct * 100,
        "rolls_done": rolls_done,
        "skipped": False,
    }


def main():
    intraday = json.loads(INTRADAY_PATH.read_text())
    cache = json.loads(CACHE_PATH.read_text())

    # Fixed best timing from prior sweep
    BEST_ENTRY = {0: "10:15", 1: "09:20", 2: "09:30", 3: "09:30", 4: "10:15"}
    BEST_EXIT  = {0: "13:00", 1: "15:15", 2: "15:15", 3: "15:00", 4: "14:30"}

    def sim_schedule(label, mapping):
        days = build_days(intraday, cache, mapping)
        per_dte: dict = {}
        for (d_str, sym, dte, bars) in days:
            e = BEST_ENTRY.get(dte, "09:30")
            x = BEST_EXIT.get(dte, "14:30")
            out = simulate_day(bars, sym, e, x, None, dte, IV)
            if out["skipped"]:
                continue
            per_dte.setdefault(dte, {"sym_counts": {}, "pnls": []})
            per_dte[dte]["pnls"].append(out["cum_pnl_pct"])
            per_dte[dte]["sym_counts"][sym] = per_dte[dte]["sym_counts"].get(sym, 0) + 1

        print(f"\n══ {label} ══")
        all_pnls = []
        for dte in sorted(per_dte):
            pnls = per_dte[dte]["pnls"]
            wins = sum(1 for p in pnls if p > 0)
            syms = per_dte[dte]["sym_counts"]
            sym_s = ", ".join(f"{k}:{v}" for k, v in syms.items())
            e = BEST_ENTRY.get(dte, "?")
            x = BEST_EXIT.get(dte, "?")
            print(f"  {dte}-DTE ({sym_s})  entry={e} exit={x}"
                  f"  n={len(pnls)} win={wins}/{len(pnls)} ({wins/len(pnls)*100:.0f}%)"
                  f"  avg={sum(pnls)/len(pnls):+.3f}%  cum={sum(pnls):+.2f}%"
                  f"  min={min(pnls):+.3f}%")
            all_pnls.extend(pnls)
        if all_pnls:
            wins = sum(1 for p in all_pnls if p > 0)
            print(f"  TOTAL: n={len(all_pnls)} win={wins}/{len(all_pnls)} "
                  f"({wins/len(all_pnls)*100:.0f}%) "
                  f"cum={sum(all_pnls):+.2f}% "
                  f"avg={sum(all_pnls)/len(all_pnls):+.3f}%/day")
        return all_pnls

    # Scenario A: current schedule (NIFTY Mon/Tue/Fri, SENSEX Wed/Thu)
    sim_schedule("SCENARIO A: Current schedule (NIFTY Mon/Tue/Fri, SENSEX Wed/Thu)",
                 INDEX_FOR_WEEKDAY)

    # Scenario B: swap on expiry days
    sim_schedule("SCENARIO B: SWAP \u2014 Tue\u2192SENSEX(2DTE), Thu\u2192NIFTY(3DTE)",
                 INDEX_FOR_WEEKDAY_SWAP)

    # Scenario C: skip expiry days entirely
    skip_expiry = {0: "NIFTY", 2: "SENSEX", 4: "NIFTY"}  # Mon, Wed, Fri only
    sim_schedule("SCENARIO C: SKIP expiry days entirely (Mon/Wed/Fri only)",
                 skip_expiry)


if __name__ == "__main__":
    main()
