"""One-off analysis: which May 2026 days would the four filters skip
for the daily ATM-straddle setup?

Schedule (per user):
  Fri / Mon / Tue (NIFTY weekly expiry on Tue) -> NIFTY
  Wed / Thu (SENSEX weekly expiry on Thu)     -> SENSEX

Filters evaluated (all decisions known by 09:30 IST):
  F1 EVENT      -> known event day (expiry-of-same-index already counted;
                   here we flag only RBI/Fed/Budget/major macro from a small
                   hard-coded list).
  F2 VIX_LVL    -> India VIX 09:15 quote > 16
  F3 VIX_CHG    -> India VIX (09:15) up > 5% vs prior-day close
  F4 GAP        -> abs(open vs prior close) > 0.5% (NIFTY) / 0.6% (SENSEX)
  F5 RANGE_15M  -> first 09:15-09:30 high-low > 0.4% (NIFTY) / 0.5% (SENSEX)
                   of the 09:15 open.

Run:
  cd c:\\TradeAI\\backend
  ..\\.venv\\Scripts\\python.exe analyze_skipdays_may2026.py
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta

import pytz

from app.data.angelone_client import AngelOneClient

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
_IST = pytz.timezone("Asia/Kolkata")

# ── Tokens ────────────────────────────────────────────────────────
NIFTY_TOKEN  = "99926000"   # NSE
SENSEX_TOKEN = "99919000"   # BSE
VIX_TOKEN    = "99926017"   # NSE (India VIX)

# ── Known event days (manual; refine as needed) ──────────────────
# Currently empty for May 2026 - update if you know of RBI/Fed dates.
EVENT_DAYS: dict[date, str] = {
    # Example format:
    # date(2026, 5, 7): "RBI MPC",
    # date(2026, 5, 14): "US CPI",
}

# ── User's weekly schedule ───────────────────────────────────────
#   Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
INDEX_FOR_WEEKDAY = {
    0: "NIFTY",   # Mon
    1: "NIFTY",   # Tue (NIFTY expiry)
    2: "SENSEX",  # Wed
    3: "SENSEX",  # Thu (SENSEX expiry)
    4: "NIFTY",   # Fri
}

# ── Filter thresholds ────────────────────────────────────────────
VIX_LEVEL_MAX     = 16.0
VIX_CHANGE_MAX_PC = 5.0
GAP_MAX_PC = {"NIFTY": 0.5, "SENSEX": 0.6}
R15_MAX_PC = {"NIFTY": 0.4, "SENSEX": 0.5}


def _fmt(c) -> str:
    return c.timestamp.strftime("%Y-%m-%d %H:%M") if c else "-"


def _daily_candles(cli: AngelOneClient, token: str, exchange: str,
                   start: date, end: date) -> dict[date, "Candle"]:
    """Fetch ONE_DAY candles for [start, end] (inclusive). Returns date->candle."""
    frm = (datetime.combine(start - timedelta(days=4), datetime.min.time())
           .strftime("%Y-%m-%d 09:15"))
    to_ = (datetime.combine(end, datetime.min.time())
           .strftime("%Y-%m-%d 15:30"))
    rows = cli.get_candle_data(token, exchange, "ONE_DAY", frm, to_)
    out = {}
    for c in rows:
        out[c.timestamp.astimezone(_IST).date()] = c
    return out


def _intraday_915_930(cli: AngelOneClient, token: str, exchange: str,
                      d: date) -> tuple[float | None, float | None, float | None]:
    """Return (open_915, high_915_to_930, low_915_to_930)."""
    frm = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=15).strftime("%Y-%m-%d %H:%M")
    to_ = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=30).strftime("%Y-%m-%d %H:%M")
    rows = cli.get_candle_data(token, exchange, "ONE_MINUTE", frm, to_)
    if not rows:
        return None, None, None
    rows.sort(key=lambda c: c.timestamp)
    return rows[0].open, max(r.high for r in rows), min(r.low for r in rows)


def main() -> None:
    cli = AngelOneClient()
    if not cli.authenticate():
        raise SystemExit("AngelOne authentication failed - check env vars")

    today = datetime.now(_IST).date()
    # If today is May, analyze May 1 .. today. Otherwise full last month.
    start = date(today.year, today.month, 1) if today.month == 5 else date(2026, 5, 1)
    end = today if today.month == start.month else date(2026, 5, 31)

    print(f"\n== Skip-day analysis: {start} .. {end} ==\n")

    # Pull all daily candles up-front (one call per instrument).
    print("Fetching daily candles...")
    nifty_daily  = _daily_candles(cli, NIFTY_TOKEN,  "NSE", start, end)
    sensex_daily = _daily_candles(cli, SENSEX_TOKEN, "BSE", start, end)
    vix_daily    = _daily_candles(cli, VIX_TOKEN,    "NSE", start, end)
    print(f"  NIFTY:{len(nifty_daily)}  SENSEX:{len(sensex_daily)}  VIX:{len(vix_daily)}")

    header = (
        f"{'DATE':12} {'DOW':4} {'IDX':6} {'OPEN':>10} {'GAP%':>7} "
        f"{'R15%':>7} {'VIX':>6} {'dVIX%':>7} {'EVENT':<12} {'SKIP?':<5} REASONS"
    )
    print("\n" + header)
    print("-" * len(header))

    cur = start
    total = skipped = 0
    by_filter: dict[str, int] = {"EVENT": 0, "VIX_LVL": 0, "VIX_CHG": 0,
                                 "GAP": 0, "R15": 0}
    while cur <= end:
        weekday = cur.weekday()
        if weekday > 4:  # weekend
            cur += timedelta(days=1)
            continue
        sym = INDEX_FOR_WEEKDAY.get(weekday)
        if not sym:
            cur += timedelta(days=1)
            continue

        spot_daily = nifty_daily if sym == "NIFTY" else sensex_daily
        spot_token = NIFTY_TOKEN if sym == "NIFTY" else SENSEX_TOKEN
        spot_exch  = "NSE" if sym == "NIFTY" else "BSE"

        today_d = spot_daily.get(cur)
        if not today_d:
            print(f"{cur} {cur.strftime('%a'):4} {sym:6} -- no daily candle (holiday?)")
            cur += timedelta(days=1)
            continue

        # Prior trading day's close (search backward up to 5 calendar days)
        prev_close = None
        for back in range(1, 6):
            prev = spot_daily.get(cur - timedelta(days=back))
            if prev:
                prev_close = prev.close
                break

        # VIX 09:15 quote + prior close
        vix_o915, _, _ = _intraday_915_930(cli, VIX_TOKEN, "NSE", cur)
        vix_prev_close = None
        for back in range(1, 6):
            v = vix_daily.get(cur - timedelta(days=back))
            if v:
                vix_prev_close = v.close
                break

        # Intraday 09:15-09:30 for the spot index
        o915, h, l = _intraday_915_930(cli, spot_token, spot_exch, cur)

        reasons = []
        if cur in EVENT_DAYS:
            reasons.append(f"EVENT:{EVENT_DAYS[cur]}")
            by_filter["EVENT"] += 1
        if vix_o915 is not None and vix_o915 > VIX_LEVEL_MAX:
            reasons.append(f"VIX>{VIX_LEVEL_MAX:.0f}({vix_o915:.1f})")
            by_filter["VIX_LVL"] += 1
        if vix_o915 is not None and vix_prev_close:
            dvix = (vix_o915 - vix_prev_close) / vix_prev_close * 100
            if dvix > VIX_CHANGE_MAX_PC:
                reasons.append(f"dVIX>{VIX_CHANGE_MAX_PC:.0f}%({dvix:+.1f}%)")
                by_filter["VIX_CHG"] += 1
        else:
            dvix = None
        gap_pc = None
        if prev_close and today_d.open:
            gap_pc = (today_d.open - prev_close) / prev_close * 100
            if abs(gap_pc) > GAP_MAX_PC[sym]:
                reasons.append(f"GAP>{GAP_MAX_PC[sym]:.1f}%({gap_pc:+.2f}%)")
                by_filter["GAP"] += 1
        r15_pc = None
        if o915 and h and l:
            r15_pc = (h - l) / o915 * 100
            if r15_pc > R15_MAX_PC[sym]:
                reasons.append(f"R15>{R15_MAX_PC[sym]:.1f}%({r15_pc:.2f}%)")
                by_filter["R15"] += 1

        skip = bool(reasons)
        total += 1
        if skip:
            skipped += 1

        ev_name = EVENT_DAYS.get(cur, "-")
        print(
            f"{cur} {cur.strftime('%a'):4} {sym:6} "
            f"{today_d.open:>10.2f} "
            f"{(f'{gap_pc:+.2f}' if gap_pc is not None else '   --  '):>7} "
            f"{(f'{r15_pc:.2f}' if r15_pc is not None else '   --  '):>7} "
            f"{(f'{vix_o915:.2f}' if vix_o915 is not None else '   -- '):>6} "
            f"{(f'{dvix:+.1f}' if dvix is not None else '  --  '):>7} "
            f"{ev_name:<12} {('YES' if skip else 'no'):<5} "
            f"{'; '.join(reasons) if reasons else ''}"
        )

        cur += timedelta(days=1)

    print()
    print(f"Total candidate trading days : {total}")
    print(f"Days SKIPPED by any filter   : {skipped} "
          f"({(skipped/total*100 if total else 0):.0f}%)")
    print(f"Days TRADED                  : {total - skipped}")
    print("Breakdown (days triggered by each filter; days can be in >1):")
    for k, v in by_filter.items():
        print(f"  {k:9} : {v}")


if __name__ == "__main__":
    main()
