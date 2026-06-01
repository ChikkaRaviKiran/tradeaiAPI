"""Tune the ATM-straddle skip-day filters using real index data.

Approach
--------
1. Pull NIFTY / SENSEX / VIX 1-min + daily candles for a tuning window
   (Mar–May 2026 = ~60 trading days). Cache to disk so re-runs are free.

2. For each candidate day (under the user's Fri/Mon/Tue=NIFTY,
   Wed/Thu=SENSEX schedule) compute these features:
      vix_open, vix_20dma, vix_rel (= vix_open / vix_20dma),
      gap_pc, r15_pc, r30_pc,
      day_dir_pc  (= abs(close - open) / open * 100)   -- straddle pain proxy
      day_range_pc (= (high - low) / open * 100)

3. Tag the day BAD if day_dir_pc > 1.0%  (a directional day that
   meaningfully hurts a 09:20-entry short straddle held to 14:30).
   Tag GOOD if day_range_pc < 0.6%  (calm range day, sellers' paradise).

4. Sweep filter thresholds and rank candidate filter sets by:
       net_edge = sum(day_dir_pc on correctly-skipped BAD days)
                  - 0.5 * sum(day_dir_pc on falsely-skipped GOOD days)
   (BAD-day loss-avoidance counts more than GOOD-day profit-foregone.)

5. Print the top filter combos plus a side-by-side month report.

Run:
    cd c:\\TradeAI\\backend
    ..\\.venv\\Scripts\\python.exe tune_skipday_filters.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz

from app.data.angelone_client import AngelOneClient

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
_IST = pytz.timezone("Asia/Kolkata")

NIFTY_TOKEN, SENSEX_TOKEN, VIX_TOKEN = "99926000", "99919000", "99926017"
INDEX_FOR_WEEKDAY = {0: "NIFTY", 1: "NIFTY", 2: "SENSEX", 3: "SENSEX", 4: "NIFTY"}

CACHE_PATH = Path(__file__).parent / "tune_skipday_cache.json"

# ── Tuning window (3 months) ─────────────────────────────────────
START = date(2026, 3, 1)
END   = date(2026, 5, 22)

# ── Labels for evaluation ────────────────────────────────────────
BAD_THRESHOLD_PC  = 0.7   # day_dir > 0.7% = directional, hurts straddle
GOOD_THRESHOLD_PC = 0.8   # day_range < 0.8% = calm sellers day

# Per-day P&L proxy for a 09:20-entry / 14:30-exit ATM short straddle:
#   pnl_pc = THETA - day_dir_pc
# THETA is the typical credit captured on a calm day (% of spot).
# Tune by gut: NIFTY ATM straddle on a flat day captures roughly 0.6-0.9%
# of spot. We use 0.7% as the breakeven.
THETA_PC = 0.7


# ── Data pull (with caching) ─────────────────────────────────────

def _fetch_daily(cli, token, exch, start, end):
    frm = (datetime.combine(start - timedelta(days=40), datetime.min.time())
           .strftime("%Y-%m-%d 09:15"))
    to_ = (datetime.combine(end, datetime.min.time())
           .strftime("%Y-%m-%d 15:30"))
    rows = cli.get_candle_data(token, exch, "ONE_DAY", frm, to_)
    out = {}
    for c in rows:
        d = c.timestamp.astimezone(_IST).date()
        out[d.isoformat()] = {"o": c.open, "h": c.high, "l": c.low, "c": c.close}
    return out


def _fetch_intraday_open(cli, token, exch, d: date, end_hm="09:30"):
    """1-min bars from 09:15 to end_hm. Returns list of (ts, o, h, l, c)."""
    frm = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=15).strftime("%Y-%m-%d %H:%M")
    to_ = datetime.combine(d, datetime.min.time()).strftime("%Y-%m-%d ") + end_hm
    rows = cli.get_candle_data(token, exch, "ONE_MINUTE", frm, to_)
    return [(c.timestamp.astimezone(_IST).strftime("%H:%M"),
             c.open, c.high, c.low, c.close) for c in rows]


def build_cache():
    if CACHE_PATH.exists():
        print(f"Loading cache from {CACHE_PATH.name}...")
        return json.loads(CACHE_PATH.read_text())

    print(f"Building cache {START} .. {END} (one-time; cached after).")
    cli = AngelOneClient()
    if not cli.authenticate():
        raise SystemExit("AngelOne authentication failed")

    cache = {
        "nifty_daily":  _fetch_daily(cli, NIFTY_TOKEN,  "NSE", START, END),
        "sensex_daily": _fetch_daily(cli, SENSEX_TOKEN, "BSE", START, END),
        "vix_daily":    _fetch_daily(cli, VIX_TOKEN,    "NSE", START, END),
        "intraday": {"NIFTY": {}, "SENSEX": {}, "VIX": {}},
    }

    cur = START
    n = 0
    while cur <= END:
        wd = cur.weekday()
        if wd <= 4:
            sym = INDEX_FOR_WEEKDAY[wd]
            token, exch = ((NIFTY_TOKEN, "NSE") if sym == "NIFTY"
                           else (SENSEX_TOKEN, "BSE"))
            # 09:15-09:45 covers both R15 and R30
            bars = _fetch_intraday_open(cli, token, exch, cur, "09:45")
            cache["intraday"][sym][cur.isoformat()] = bars
            time.sleep(0.6)  # rate-limit guard
            vix_bars = _fetch_intraday_open(cli, VIX_TOKEN, "NSE", cur, "09:30")
            cache["intraday"]["VIX"][cur.isoformat()] = vix_bars
            time.sleep(0.6)
            n += 1
            if n % 10 == 0:
                print(f"  fetched {n} days, last={cur}")
        cur += timedelta(days=1)

    CACHE_PATH.write_text(json.dumps(cache))
    print(f"Cached {n} days to {CACHE_PATH.name}")
    return cache


# ── Feature engineering ──────────────────────────────────────────

def _prev_close(daily: dict, d: date, max_back: int = 6) -> Optional[float]:
    for b in range(1, max_back + 1):
        row = daily.get((d - timedelta(days=b)).isoformat())
        if row:
            return row["c"]
    return None


def _vix_20dma(vix_daily: dict, d: date) -> Optional[float]:
    closes = []
    cur = d - timedelta(days=1)
    look = 0
    while look < 35 and len(closes) < 20:
        row = vix_daily.get(cur.isoformat())
        if row:
            closes.append(row["c"])
        cur -= timedelta(days=1)
        look += 1
    return sum(closes) / len(closes) if closes else None


def features(cache: dict) -> list[dict]:
    rows = []
    nifty_d, sensex_d, vix_d = (cache["nifty_daily"], cache["sensex_daily"],
                                cache["vix_daily"])
    for sym_key, daily in [("NIFTY", nifty_d), ("SENSEX", sensex_d)]:
        intraday = cache["intraday"][sym_key]
        vix_intra = cache["intraday"]["VIX"]
        for date_str, today in daily.items():
            d = date.fromisoformat(date_str)
            if not (START <= d <= END) or d.weekday() > 4:
                continue
            if INDEX_FOR_WEEKDAY[d.weekday()] != sym_key:
                continue
            bars = intraday.get(date_str) or []
            if not bars:
                continue
            # R15 = 09:15-09:30 (first 15 bars), R30 = 09:15-09:45 (first 30)
            b15, b30 = bars[:15], bars[:30]
            r15_high = max(b[2] for b in b15) if b15 else None
            r15_low  = min(b[3] for b in b15) if b15 else None
            r30_high = max(b[2] for b in b30) if b30 else None
            r30_low  = min(b[3] for b in b30) if b30 else None
            open_915 = b15[0][1] if b15 else today["o"]
            vix_bars = vix_intra.get(date_str) or []
            vix_open = vix_bars[0][1] if vix_bars else None
            vix_dma = _vix_20dma(vix_d, d)
            vix_rel = (vix_open / vix_dma) if (vix_open and vix_dma) else None
            prev_c = _prev_close(daily, d)
            gap_pc = ((today["o"] - prev_c) / prev_c * 100) if prev_c else None
            r15_pc = ((r15_high - r15_low) / open_915 * 100) if (r15_high and open_915) else None
            r30_pc = ((r30_high - r30_low) / open_915 * 100) if (r30_high and open_915) else None
            day_dir_pc = abs(today["c"] - today["o"]) / today["o"] * 100
            day_range_pc = (today["h"] - today["l"]) / today["o"] * 100
            rows.append({
                "date": date_str, "dow": d.strftime("%a"), "sym": sym_key,
                "vix_open": vix_open, "vix_dma": vix_dma, "vix_rel": vix_rel,
                "gap_pc": gap_pc, "r15_pc": r15_pc, "r30_pc": r30_pc,
                "day_dir_pc": day_dir_pc, "day_range_pc": day_range_pc,
                "label_bad":  day_dir_pc > BAD_THRESHOLD_PC,
                "label_good": day_range_pc < GOOD_THRESHOLD_PC,
            })
    rows.sort(key=lambda r: r["date"])
    return rows


# ── Filter evaluation ────────────────────────────────────────────

def apply_filter(row: dict, vix_rel_max: float, vix_abs_max: float,
                 gap_max: dict, r15_max: dict) -> list[str]:
    reasons = []
    if row["vix_rel"] is not None and row["vix_rel"] > vix_rel_max:
        reasons.append(f"VIXrel>{vix_rel_max:.2f}")
    if vix_abs_max and row["vix_open"] is not None and row["vix_open"] > vix_abs_max:
        reasons.append(f"VIX>{vix_abs_max:.0f}")
    if row["gap_pc"] is not None and abs(row["gap_pc"]) > gap_max[row["sym"]]:
        reasons.append(f"GAP>{gap_max[row['sym']]:.2f}%")
    if row["r15_pc"] is not None and row["r15_pc"] > r15_max[row["sym"]]:
        reasons.append(f"R15>{r15_max[row['sym']]:.2f}%")
    return reasons


def score(rows: list[dict], cfg: dict) -> dict:
    bad_skipped = good_skipped = bad_traded = good_traded = 0
    traded_pnl = skipped_pnl_if_traded = 0.0
    min_votes = cfg.get("min_votes", 1)
    filter_cfg = {k: v for k, v in cfg.items() if k != "min_votes"}
    for r in rows:
        reasons = apply_filter(r, **filter_cfg)
        skipped = len(reasons) >= min_votes
        # Per-day P&L proxy (positive on calm days, negative on directional)
        pnl = THETA_PC - r["day_dir_pc"]
        if skipped:
            skipped_pnl_if_traded += pnl  # what we GAVE UP by skipping
            if r["label_bad"]:
                bad_skipped += 1
            if r["label_good"]:
                good_skipped += 1
        else:
            traded_pnl += pnl
            if r["label_bad"]:
                bad_traded += 1
            if r["label_good"]:
                good_traded += 1
    total_bad  = bad_skipped + bad_traded
    total_good = good_skipped + good_traded
    return {
        "bad_skipped": bad_skipped, "bad_total": total_bad,
        "good_skipped": good_skipped, "good_total": total_good,
        "recall_bad": (bad_skipped / total_bad * 100) if total_bad else 0,
        "false_skip_good": (good_skipped / total_good * 100) if total_good else 0,
        "traded_pnl": traded_pnl,
        "foregone_pnl": skipped_pnl_if_traded,
        "net_edge": traded_pnl,  # what we'd ACTUALLY make with this filter
        "days_traded": sum(1 for r in rows
                            if len(apply_filter(r, **filter_cfg)) < min_votes),
        "days_total":  len(rows),
    }


def main():
    cache = build_cache()
    rows = features(cache)
    n = len(rows)
    bad = sum(1 for r in rows if r["label_bad"])
    good = sum(1 for r in rows if r["label_good"])
    print(f"\nTotal days: {n}  | BAD (|close-open|>{BAD_THRESHOLD_PC}%): {bad}"
          f" ({bad/n*100:.0f}%)  | GOOD (range<{GOOD_THRESHOLD_PC}%): {good}"
          f" ({good/n*100:.0f}%)\n")

    # ── Sweep candidate filter combos (with K-of-N voting) ───────
    # Each rule is a vote. Skip only if total votes >= min_votes.
    # min_votes=1 = strict (any rule skips), min_votes=2 = loose,
    # min_votes=3 = very loose (need 3 of 4 signals to agree).
    grid = []
    for min_votes in [1, 2, 3]:
        for vix_rel_max in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]:
            for vix_abs_max in [0, 18, 20, 22]:   # 0 = disable absolute
                for gap_n in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                    for r15_n in [0.35, 0.45, 0.55, 0.65, 0.75]:
                        cfg = {
                            "min_votes":  min_votes,
                            "vix_rel_max": vix_rel_max,
                            "vix_abs_max": vix_abs_max,
                            "gap_max": {"NIFTY": gap_n, "SENSEX": gap_n + 0.1},
                            "r15_max": {"NIFTY": r15_n, "SENSEX": r15_n + 0.1},
                        }
                        s = score(rows, cfg)
                        grid.append((cfg, s))

    # Sort by total P&L (we want maximum money made, not just per-day avg)
    grid.sort(key=lambda x: x[1]["traded_pnl"], reverse=True)

    # Add the "no-filter" baseline as rank 0 for comparison
    baseline_cfg = {
        "vix_rel_max": 99.0, "vix_abs_max": 0,
        "gap_max": {"NIFTY": 99.0, "SENSEX": 99.0},
        "r15_max": {"NIFTY": 99.0, "SENSEX": 99.0},
    }
    baseline_s = score(rows, baseline_cfg)

    print("─── Top 20 filter combos by cumulative P&L proxy % ───")
    print(f"{'rank':4} {'K':>2} {'VIXrel':>6} {'VIXabs':>6} {'GAPn':>5} {'R15n':>5}"
          f"  {'traded':>7} {'BADskip':>7}/{'tot':>3}"
          f"  {'GOODskip':>8}/{'tot':>3}"
          f"  {'pnl%':>7}  {'avg/day':>8}")
    print(f"{'NONE':4} {'-':>2} {'-':>6} {'-':>6} {'-':>5} {'-':>5}"
          f"  {baseline_s['days_traded']:>7}"
          f"  {0:>7}/{baseline_s['bad_total']:>3}"
          f"  {0:>8}/{baseline_s['good_total']:>3}"
          f"  {baseline_s['traded_pnl']:>7.2f}"
          f"  {baseline_s['traded_pnl']/max(1,baseline_s['days_traded']):>8.3f}")
    for i, (cfg, s) in enumerate(grid[:20], 1):
        avg = s['traded_pnl'] / max(1, s['days_traded'])
        print(f"{i:4} {cfg.get('min_votes',1):>2}"
              f" {cfg['vix_rel_max']:>6.2f} {cfg['vix_abs_max']:>6}"
              f" {cfg['gap_max']['NIFTY']:>5.2f} {cfg['r15_max']['NIFTY']:>5.2f}"
              f"  {s['days_traded']:>7}"
              f"  {s['bad_skipped']:>7}/{s['bad_total']:>3}"
              f"  {s['good_skipped']:>8}/{s['good_total']:>3}"
              f"  {s['traded_pnl']:>7.2f}"
              f"  {avg:>8.3f}")

    # ── Two recommendations: BALANCED (>=30 days, max pnl) and SELECTIVE (max pnl) ──
    balanced = [(cfg, s) for cfg, s in grid
                if s["days_traded"] >= 30
                and s["traded_pnl"] > baseline_s["traded_pnl"]]
    balanced.sort(key=lambda x: x[1]["traded_pnl"], reverse=True)

    selective = [(cfg, s) for cfg, s in grid
                 if s["days_traded"] >= 20
                 and s["traded_pnl"] > baseline_s["traded_pnl"]]
    selective.sort(key=lambda x: x[1]["traded_pnl"], reverse=True)

    def print_rec(title: str, picks: list, fallback_msg: str):
        print(f"\n─── {title} ────────────────────────────────────────")
        if not picks:
            print(f"  (no combo meets criteria — {fallback_msg})")
            return
        cfg, s = picks[0]
        print(f"  Voting rule         :  skip only if >= {cfg.get('min_votes',1)} of 4 signals fire")
        print(f"  VIX 09:15 / 20-DMA  >  {cfg['vix_rel_max']:.2f}     -> +1 vote")
        print(f"  India VIX 09:15     >  {cfg['vix_abs_max']:>5}      -> +1 vote (0=disabled)")
        print(f"  abs(gap) > {cfg['gap_max']['NIFTY']:.2f}% (NIFTY) / "
              f"{cfg['gap_max']['SENSEX']:.2f}% (SENSEX) -> +1 vote")
        print(f"  R15 09:15-30 range > {cfg['r15_max']['NIFTY']:.2f}% (NIFTY) / "
              f"{cfg['r15_max']['SENSEX']:.2f}% (SENSEX) -> +1 vote")
        print(f"  -> trades {s['days_traded']}/{n} days "
              f"({s['days_traded']/n*100:.0f}%)")
        print(f"  -> catches {s['bad_skipped']}/{s['bad_total']} BAD days "
              f"({s['recall_bad']:.0f}% recall)")
        print(f"  -> falsely skips {s['good_skipped']}/{s['good_total']} GOOD days "
              f"({s['false_skip_good']:.0f}%)")
        print(f"  -> cumulative P&L proxy : {s['traded_pnl']:+.2f}% "
              f"(avg {s['traded_pnl']/max(1,s['days_traded']):+.3f}% per traded day)")

    print(f"\n=== Baseline (no filter): trades {baseline_s['days_traded']} days, "
          f"P&L proxy = {baseline_s['traded_pnl']:+.2f}% "
          f"(avg {baseline_s['traded_pnl']/max(1,baseline_s['days_traded']):+.3f}%/day) ===")
    print_rec("RECOMMENDED — BALANCED (>=30 trading days)",
              balanced, "baseline is best")
    print_rec("RECOMMENDED — SELECTIVE (>=20 trading days)",
              selective, "baseline is best")

    # Use BALANCED for downstream month/day breakdown if available, else SELECTIVE
    rec = balanced[0] if balanced else (selective[0] if selective else grid[0])
    cfg, s = rec
    print(f"\n(using BALANCED filter for month/day breakdown below)")

    # ── Month-by-month verdict with recommended filter ───────────
    print("\n─── Month-by-month using RECOMMENDED filter ──────────────────")
    by_month: dict[str, dict] = {}
    min_votes = cfg.get("min_votes", 1)
    filter_cfg = {k: v for k, v in cfg.items() if k != "min_votes"}
    for r in rows:
        m = r["date"][:7]
        bym = by_month.setdefault(m, {"days": 0, "skipped": 0,
                                      "bad": 0, "bad_skipped": 0,
                                      "good": 0, "good_traded": 0})
        bym["days"] += 1
        reasons = apply_filter(r, **filter_cfg)
        skipped = len(reasons) >= min_votes
        if skipped:
            bym["skipped"] += 1
        if r["label_bad"]:
            bym["bad"] += 1
            if skipped:
                bym["bad_skipped"] += 1
        if r["label_good"]:
            bym["good"] += 1
            if not skipped:
                bym["good_traded"] += 1
    print(f"{'month':8} {'days':>5} {'skip%':>6} "
          f"{'BADcatch':>10} {'GOODtraded':>11}")
    for m, b in sorted(by_month.items()):
        print(f"{m:8} {b['days']:>5} {b['skipped']/b['days']*100:>5.0f}% "
              f"  {b['bad_skipped']}/{b['bad']:<3}"
              f"     {b['good_traded']}/{b['good']:<3}")

    # ── Show daily decisions for May 2026 with the recommended filter ──
    print("\n─── May 2026 daily decisions (recommended filter) ─────────────")
    print(f"{'date':12} {'dow':4} {'sym':6} {'open':>9} {'gap%':>6} "
          f"{'r15%':>6} {'vix':>5} {'vixR':>5} {'dir%':>5} "
          f"{'label':<5} {'skip?':<5} reasons")
    for r in rows:
        if r["date"][:7] != "2026-05":
            continue
        reasons = apply_filter(r, **filter_cfg)
        skipped = len(reasons) >= min_votes
        label = "BAD" if r["label_bad"] else ("GOOD" if r["label_good"] else "mid")
        skip = "YES" if skipped else "no"
        op = cache["nifty_daily" if r["sym"] == "NIFTY" else "sensex_daily"][r["date"]]["o"]
        gap_s  = f"{r['gap_pc']:+.2f}"  if r["gap_pc"]  is not None else "   -"
        r15_s  = f"{r['r15_pc']:.2f}"   if r["r15_pc"]  is not None else "   -"
        vix_s  = f"{r['vix_open']:.1f}" if r["vix_open"] is not None else " -"
        vixr_s = f"{r['vix_rel']:.2f}"  if r["vix_rel"]  is not None else " -"
        print(f"{r['date']:12} {r['dow']:4} {r['sym']:6} "
              f"{op:>9.2f} {gap_s:>6} {r15_s:>6} {vix_s:>5} {vixr_s:>5} "
              f"{r['day_dir_pc']:>5.2f} {label:<5} {skip:<5} "
              f"{'; '.join(reasons)}")


if __name__ == "__main__":
    main()
