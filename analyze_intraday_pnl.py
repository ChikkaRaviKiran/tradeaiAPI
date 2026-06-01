"""Re-evaluate BAD days using INTRADAY straddle P&L (not close-to-open).

Why
---
The simple proxy `THETA - |close-open|` assumes you hold to close and treats
every directional day as a pure loss. In reality:

  * Entry  = 09:20 IST
  * Exit   = 14:30 IST  (~5h15m later)
  * The move that hurts is `|spot_14:30 - spot_09:20|`, NOT |close-open|.
  * By 14:30 the short straddle has already collected ~80% of one full day's
    theta.

We use a Bachelier-style approximation to value the ATM straddle:

    V(t) = sqrt(  |S(t) - K|^2  +  ( S0 * IV * sqrt((T - t)/252) * 0.8 )^2 )

so   short_pnl_pct = ( V(0) - V(exit) ) / S0 * 100

where T is the days-to-expiry at entry. Tuesday (NIFTY) / Thursday (SENSEX)
are 0-DTE; Mon/Wed are 1-DTE; Fri NIFTY is 4-DTE (weekend included).
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import pytz

from app.data.angelone_client import AngelOneClient

logging.basicConfig(level=logging.WARNING)
_IST = pytz.timezone("Asia/Kolkata")

NIFTY_TOKEN, SENSEX_TOKEN = "99926000", "99919000"
INDEX_FOR_WEEKDAY = {0: "NIFTY", 1: "NIFTY", 2: "SENSEX", 3: "SENSEX", 4: "NIFTY"}
ENTRY = dtime(9, 20)
EXIT  = dtime(14, 30)

# Days-to-expiry by weekday (NIFTY expires Tue, SENSEX expires Thu)
DTE_NIFTY  = {0: 1, 1: 0, 4: 4}   # Mon, Tue, Fri
DTE_SENSEX = {2: 1, 3: 0}          # Wed, Thu

# Empirical: weekly index option IV typically 14-22% during this window
IV = 0.17  # 17%

CACHE_PATH    = Path(__file__).parent / "tune_skipday_cache.json"
INTRADAY_PATH = Path(__file__).parent / "intraday_full_day_cache.json"


def load_filter_cache() -> dict:
    return json.loads(CACHE_PATH.read_text())


def build_full_intraday(dates_needed: list[tuple[str, str]]) -> dict:
    """dates_needed: list of (date_str, symbol). Returns {sym: {date: bars}}."""
    if INTRADAY_PATH.exists():
        cache = json.loads(INTRADAY_PATH.read_text())
    else:
        cache = {"NIFTY": {}, "SENSEX": {}}

    todo = [(d, s) for d, s in dates_needed
            if d not in cache.get(s, {})
            or not cache[s][d]
            or len(cache[s][d]) < 50]
    if not todo:
        print(f"Full-day intraday cache hit ({sum(len(v) for v in cache.values())} days)")
        return cache

    print(f"Fetching {len(todo)} day(s) of full-day 1-min bars from AngelOne...")
    client = AngelOneClient()
    for i, (d, sym) in enumerate(todo, 1):
        token = NIFTY_TOKEN if sym == "NIFTY" else SENSEX_TOKEN
        exch  = "NSE" if sym == "NIFTY" else "BSE"
        d0 = datetime.strptime(d, "%Y-%m-%d")
        f = d0.replace(hour=9, minute=15).strftime("%Y-%m-%d %H:%M")
        t = d0.replace(hour=15, minute=30).strftime("%Y-%m-%d %H:%M")
        try:
            bars = client.get_candle_data(
                symbol_token=token, exchange=exch,
                interval="FIVE_MINUTE",  # 5-min bars: ~75 per day, faster
                from_date=f, to_date=t,
            ) or []
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {d} {sym}: ERR {e}")
            bars = []
            time.sleep(1.0)
        rows = []
        for c in bars:
            ts = c.timestamp.strftime("%H:%M")
            rows.append([ts, c.open, c.high, c.low, c.close])
        cache.setdefault(sym, {})[d] = rows
        print(f"  [{i}/{len(todo)}] {d} {sym}: {len(rows)} bars")
        if i % 10 == 0:
            INTRADAY_PATH.write_text(json.dumps(cache))
        time.sleep(0.4)
    INTRADAY_PATH.write_text(json.dumps(cache))
    return cache


def price_at(bars: list, target: dtime) -> float | None:
    """Return the close price at-or-after target time (first match)."""
    tt = target.strftime("%H:%M")
    for b in bars:
        if b[0] >= tt:
            return b[4]
    return None


def straddle_value(spot: float, strike: float, dte_days: float,
                   t_frac_consumed: float, iv: float) -> float:
    """Bachelier ATM straddle value.

    dte_days: days to expiry at ENTRY (0 = same-day, 1 = next-day, etc.)
    t_frac_consumed: fraction of one trading day already elapsed since entry.
      A full trading day = 6.25h. At exit (14:30 from 09:20 entry) = 5.17/6.25 ≈ 0.83
    """
    # Time remaining to expiry in trading days, never below 0.05 (cushion)
    t_remaining = max(dte_days - t_frac_consumed, 0.05)
    atm_part = spot * iv * math.sqrt(t_remaining / 252) * 0.8
    intrinsic = abs(spot - strike)
    return math.sqrt(intrinsic ** 2 + atm_part ** 2)


def get_dte(d: str, sym: str) -> int:
    dt = datetime.strptime(d, "%Y-%m-%d")
    wd = dt.weekday()
    return (DTE_NIFTY if sym == "NIFTY" else DTE_SENSEX).get(wd, 1)


def main():
    cache = load_filter_cache()

    # Build the list of trading days under the user's schedule
    rows = []
    nd = cache["nifty_daily"]
    sd = cache["sensex_daily"]
    seen = set()
    for d_str in sorted(set(nd) | set(sd)):
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        wd = dt.weekday()
        if wd not in INDEX_FOR_WEEKDAY:
            continue
        sym = INDEX_FOR_WEEKDAY[wd]
        daily = (nd if sym == "NIFTY" else sd).get(d_str)
        if not daily:
            continue
        rows.append((d_str, sym, daily))
        seen.add((d_str, sym))

    intraday = build_full_intraday(list(seen))

    print(f"\nAnalysing {len(rows)} days with INTRADAY straddle P&L model")
    print(f"  Entry: {ENTRY}  Exit: {EXIT}  IV assumed: {IV*100:.0f}%\n")

    out = []
    for d_str, sym, daily in rows:
        bars = intraday.get(sym, {}).get(d_str) or []
        if len(bars) < 30:
            continue
        s_entry = price_at(bars, ENTRY)
        s_exit  = price_at(bars, EXIT)
        if not s_entry or not s_exit:
            continue
        strike = round(s_entry / 50) * 50 if sym == "NIFTY" else round(s_entry / 100) * 100

        dte = get_dte(d_str, sym)
        # 09:20 -> 14:30 = 5h10m of a 6h15m day = 0.827 of one trading day
        t_frac = (5 * 60 + 10) / (6 * 60 + 15)

        v0 = straddle_value(s_entry, strike, dte, 0.0,    IV)
        v1 = straddle_value(s_exit,  strike, dte, t_frac, IV)
        pnl_pct = (v0 - v1) / s_entry * 100  # short straddle: collect v0, pay v1

        intraday_move_pct = abs(s_exit - s_entry) / s_entry * 100
        day_dir_pct = abs(daily["c"] - daily["o"]) / daily["o"] * 100
        bad_by_close = day_dir_pct > 0.7
        out.append({
            "date": d_str, "sym": sym, "dte": dte,
            "spot_entry": s_entry, "spot_exit": s_exit, "strike": strike,
            "intraday_move_pct": intraday_move_pct,
            "day_dir_pct": day_dir_pct,
            "v0_pct": v0 / s_entry * 100,
            "v1_pct": v1 / s_entry * 100,
            "straddle_pnl_pct": pnl_pct,
            "bad_by_close": bad_by_close,
        })

    # ── Summary tables ──────────────────────────────────────────
    bad_days   = [r for r in out if r["bad_by_close"]]
    other_days = [r for r in out if not r["bad_by_close"]]

    def fmt(rs):
        if not rs:
            return "n=0"
        pnl = [r["straddle_pnl_pct"] for r in rs]
        win = sum(1 for p in pnl if p > 0)
        return (f"n={len(rs)}  win={win}/{len(rs)} ({win/len(rs)*100:.0f}%)  "
                f"avg={sum(pnl)/len(pnl):+.3f}%  "
                f"min={min(pnl):+.3f}%  max={max(pnl):+.3f}%  "
                f"cum={sum(pnl):+.2f}%")

    print("─── INTRADAY straddle P&L (Bachelier model, exit 14:30) ───")
    print(f"  All days        : {fmt(out)}")
    print(f"  BAD by close    : {fmt(bad_days)}     <- 'pure loss' assumption was WRONG if avg/win > 0")
    print(f"  Non-BAD days    : {fmt(other_days)}")

    # ── BAD-day day-by-day breakdown ───────────────────────────
    print("\n─── Every BAD day (close-to-open > 0.7%) — actual intraday P&L ──────")
    print(f"{'date':12} {'sym':6} {'dte':>3} {'entry':>9} {'exit':>9}"
          f"  {'intraday%':>9} {'closeOO%':>9}  {'v0%':>5} {'v1%':>5}"
          f"  {'pnl%':>7} {'verdict':<10}")
    for r in sorted(bad_days, key=lambda x: x["straddle_pnl_pct"]):
        v = "BIG LOSS" if r["straddle_pnl_pct"] < -0.3 else (
            "small loss" if r["straddle_pnl_pct"] < 0 else (
            "BREAK-EVEN" if r["straddle_pnl_pct"] < 0.1 else "PROFIT"))
        print(f"{r['date']:12} {r['sym']:6} {r['dte']:>3}"
              f" {r['spot_entry']:>9.1f} {r['spot_exit']:>9.1f}"
              f"  {r['intraday_move_pct']:>8.2f}%"
              f" {r['day_dir_pct']:>8.2f}%"
              f"  {r['v0_pct']:>4.2f}% {r['v1_pct']:>4.2f}%"
              f"  {r['straddle_pnl_pct']:>+7.3f}% {v:<10}")

    # ── Comparison: simple proxy vs intraday model ─────────────
    THETA = 0.7
    print("\n─── Proxy vs reality across BAD days ──────────────")
    proxy_total = sum(THETA - r["day_dir_pct"] for r in bad_days)
    real_total  = sum(r["straddle_pnl_pct"] for r in bad_days)
    print(f"  Simple proxy (THETA - |close-open|) total: {proxy_total:+.2f}%")
    print(f"  Bachelier intraday model total           : {real_total:+.2f}%")
    print(f"  -> Reality is {real_total - proxy_total:+.2f}% "
          f"{'BETTER' if real_total > proxy_total else 'WORSE'} than proxy")
    print(f"  -> {sum(1 for r in bad_days if r['straddle_pnl_pct'] > 0)}/"
          f"{len(bad_days)} BAD-by-close days were actually PROFITABLE on the straddle")


if __name__ == "__main__":
    main()
