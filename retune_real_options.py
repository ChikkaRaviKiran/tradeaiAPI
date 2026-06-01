#!/usr/bin/env python3
"""Real-option-data sweep: optimise entry/exit per DTE + find skip-day filter.

For each schedule day:
  1. Pull ATM straddle 1-min OHLC for entire trading day (single query)
  2. Pull index 1-min OHLC for same day (for gap/range filters)
  3. Pull prev day index close
  4. Sweep entry/exit grid → compute P&L per combo
  5. Aggregate by DTE bucket; report best per bucket
  6. Compute candidate filters: overnight-gap%, first-hour-range%, day-direction
     correlate with day P&L → recommend skip rules
"""
import csv, subprocess, json
from datetime import datetime, timedelta
from collections import defaultdict

NIFTY_LOT, SENSEX_LOT = 75, 20
NUM_LOTS = 3
PSQL = ["docker", "exec", "-i", "tradeai-postgres",
        "psql", "-U", "tradeai", "-d", "tradeai", "-A", "-F", "|", "-t", "-c"]


def q(sql):
    r = subprocess.run(PSQL + [sql], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    out = []
    for line in r.stdout.strip().split("\n"):
        if line.strip():
            out.append(line.split("|"))
    return out


def to_min(hhmm): h, m = hhmm.split(":"); return int(h) * 60 + int(m)


def round_strike(spot, sym):
    step = 50 if sym == "NIFTY" else 100
    return int(round(spot / step) * step)


# ── Step 1: build day cache (1 day = 1 row of intraday data) ────
schedule = list(csv.DictReader(open("swap_schedule.csv")))
print(f"Loading real-DB data for {len(schedule)} days...")

days = []  # list of dicts with intraday CE+PE bars, spot bars, prev close
for i, r in enumerate(schedule, 1):
    d, sym = r["date"], r["sym"]
    # 1. get spot at 09:20 (for ATM determination)
    rs = q(f"SELECT close FROM index_candles WHERE instrument='{sym}' "
           f"AND date='{d}' AND to_char(timestamp,'HH24:MI')='09:20' LIMIT 1;")
    if not rs:
        rs = q(f"SELECT close FROM index_candles WHERE instrument='{sym}' "
               f"AND date='{d}' ORDER BY timestamp LIMIT 1;")
    if not rs:
        print(f"  [{i}] {d} {sym}: no spot, skip")
        continue
    spot920 = float(rs[0][0])
    strike = round_strike(spot920, sym)

    # 2. nearest expiry
    rs = q(f"SELECT DISTINCT expiry FROM option_candles "
           f"WHERE instrument='{sym}' AND date='{d}';")
    if not rs:
        print(f"  [{i}] {d} {sym}: no expiries, skip")
        continue
    target = datetime.strptime(d, "%Y-%m-%d").date()
    parsed = []
    for (e,) in rs:
        try:
            dd = datetime.strptime(e, "%d%b%y").date()
            parsed.append((dd, e))
        except ValueError:
            continue
    parsed.sort()
    expiry = next((e for dd, e in parsed if dd >= target), parsed[0][1] if parsed else None)
    if not expiry:
        print(f"  [{i}] {d} {sym}: no expiry"); continue

    # 3. full-day CE + PE 1-min closes for this strike+expiry
    rs = q(f"SELECT to_char(timestamp,'HH24:MI'), option_type, close "
           f"FROM option_candles WHERE instrument='{sym}' AND date='{d}' "
           f"AND expiry='{expiry}' AND strike={strike} "
           f"AND option_type IN ('CE','PE') ORDER BY timestamp;")
    if not rs:
        print(f"  [{i}] {d} {sym} K={strike} exp={expiry}: no option rows"); continue
    ce = {}; pe = {}
    for hhmm, ot, c in rs:
        v = float(c)
        if ot == "CE": ce[hhmm] = v
        else: pe[hhmm] = v

    # 4. spot 1-min closes for filters
    rs = q(f"SELECT to_char(timestamp,'HH24:MI'), close FROM index_candles "
           f"WHERE instrument='{sym}' AND date='{d}' ORDER BY timestamp;")
    spot_bars = {hhmm: float(c) for hhmm, c in rs}

    # 5. prev close (for gap)
    rs = q(f"SELECT close FROM index_candles WHERE instrument='{sym}' "
           f"AND date < '{d}' ORDER BY timestamp DESC LIMIT 1;")
    prev_close = float(rs[0][0]) if rs else spot920

    days.append({
        "date": d, "sym": sym, "dte": int(r["dte"]),
        "strike": strike, "expiry": expiry,
        "spot920": spot920, "ce": ce, "pe": pe, "spot_bars": spot_bars,
        "prev_close": prev_close,
    })
    if i % 10 == 0:
        print(f"  [{i}/{len(schedule)}] loaded")

print(f"\nUsable days: {len(days)}")


# ── Step 2: helpers ───────────────────────────────────────────
def get_premium(bars, hhmm):
    """Return option close at hhmm, or nearest <= or nearest > if missing."""
    if hhmm in bars:
        return bars[hhmm]
    # nearest <= hhmm
    candidates = sorted(t for t in bars if to_min(t) <= to_min(hhmm))
    if candidates:
        return bars[candidates[-1]]
    candidates = sorted(t for t in bars if to_min(t) >= to_min(hhmm))
    return bars[candidates[0]] if candidates else None


def day_pnl(day, entry, exit_t):
    """P&L in points + percent."""
    ce_in = get_premium(day["ce"], entry)
    pe_in = get_premium(day["pe"], entry)
    ce_ou = get_premium(day["ce"], exit_t)
    pe_ou = get_premium(day["pe"], exit_t)
    if None in (ce_in, pe_in, ce_ou, pe_ou):
        return None, None
    credit = ce_in + pe_in
    debit  = ce_ou + pe_ou
    pnl_pts = credit - debit
    pnl_pct = pnl_pts / day["spot920"] * 100
    return pnl_pts, pnl_pct


def day_rupees(day, pnl_pts):
    lot = NIFTY_LOT if day["sym"] == "NIFTY" else SENSEX_LOT
    return pnl_pts * lot * NUM_LOTS


# ── Step 3: filter features per day ───────────────────────────
def gap_pct(d):
    return (d["spot920"] - d["prev_close"]) / d["prev_close"] * 100


def first_hour_range_pct(d):
    """09:15 - 10:15 range as % of open."""
    bars = [(t, p) for t, p in d["spot_bars"].items() if to_min(t) <= to_min("10:15")]
    if len(bars) < 5: return None
    hi = max(p for _, p in bars); lo = min(p for _, p in bars)
    op = next((p for t, p in bars if t == "09:15"), bars[0][1])
    return (hi - lo) / op * 100


def trend_direction_915_to_945(d):
    """signed move 09:15->09:45 as % of open"""
    b915 = d["spot_bars"].get("09:15") or d["spot920"]
    b945 = d["spot_bars"].get("09:45") or d["spot920"]
    return (b945 - b915) / b915 * 100


for d in days:
    d["gap"]       = gap_pct(d)
    d["fh_range"]  = first_hour_range_pct(d)
    d["dir_30m"]   = trend_direction_915_to_945(d)


# ── Step 4: sweep entry/exit per DTE ──────────────────────────
ENTRIES = ["09:20", "09:30", "09:45", "10:00", "10:15", "10:30"]
EXITS   = ["12:00", "13:00", "13:30", "14:00", "14:30", "15:00", "15:15"]

def stats(pnls_rupees, pnls_pct):
    if not pnls_rupees: return None
    wins = sum(1 for p in pnls_rupees if p > 0)
    avg = sum(pnls_rupees) / len(pnls_rupees)
    return {
        "n": len(pnls_rupees), "wins": wins,
        "win_pct": wins / len(pnls_rupees) * 100,
        "cum_rup": sum(pnls_rupees), "avg_rup": avg,
        "cum_pct": sum(pnls_pct), "avg_pct": sum(pnls_pct) / len(pnls_pct),
        "min_rup": min(pnls_rupees), "max_rup": max(pnls_rupees),
    }


print("\n══════ ENTRY/EXIT SWEEP per DTE bucket (real DB option prices) ══════")
for dte_target in (1, 2, 3, 4):
    sub = [d for d in days if d["dte"] == dte_target]
    if not sub: continue
    print(f"\n── DTE {dte_target}  ({len(sub)} days) ──")
    results = []
    for e in ENTRIES:
        for x in EXITS:
            if to_min(x) <= to_min(e) + 60: continue  # min 1h hold
            rup, pct = [], []
            for d in sub:
                pn_pts, pn_pct = day_pnl(d, e, x)
                if pn_pts is None: continue
                rup.append(day_rupees(d, pn_pts))
                pct.append(pn_pct)
            s = stats(rup, pct)
            if s: results.append((e, x, s))
    # sort by cum_rup
    results.sort(key=lambda r: r[2]["cum_rup"], reverse=True)
    print(f"  {'rk':>2} {'entry':>5} {'exit':>5} {'n':>3} {'win%':>5} "
          f"{'cum ₹':>10} {'avg ₹':>8} {'min ₹':>9} {'max ₹':>9}")
    for i, (e, x, s) in enumerate(results[:5], 1):
        print(f"  {i:>2} {e:>5} {x:>5} {s['n']:>3} {s['win_pct']:>4.0f}% "
              f"{s['cum_rup']:>+9,.0f} {s['avg_rup']:>+7,.0f} "
              f"{s['min_rup']:>+8,.0f} {s['max_rup']:>+8,.0f}")
    # show baseline (current rec)
    cur_map = {1: ("09:20","15:15"), 2: ("09:30","15:15"),
               3: ("09:30","15:00"), 4: ("10:15","14:30")}
    ce, cx = cur_map[dte_target]
    base = next((r for r in results if r[0]==ce and r[1]==cx), None)
    if base:
        s = base[2]
        print(f"  CURRENT REC ({ce}/{cx}): cum=₹{s['cum_rup']:+,.0f}  win={s['win_pct']:.0f}%")


# ── Step 5: filter analysis ───────────────────────────────────
# Use the OVERALL BEST combo per DTE for filter analysis
best_combo_per_dte = {}
for dte_target in (1, 2, 3, 4):
    sub = [d for d in days if d["dte"] == dte_target]
    if not sub: continue
    best = None; best_s = None
    for e in ENTRIES:
        for x in EXITS:
            if to_min(x) <= to_min(e) + 60: continue
            rup, pct = [], []
            for d in sub:
                pn, pp = day_pnl(d, e, x)
                if pn is None: continue
                rup.append(day_rupees(d, pn)); pct.append(pp)
            s = stats(rup, pct)
            if s and (best_s is None or s["cum_rup"] > best_s["cum_rup"]):
                best = (e, x); best_s = s
    best_combo_per_dte[dte_target] = best

# Compute per-day P&L using best combo
per_day_best = []
for d in days:
    e, x = best_combo_per_dte.get(d["dte"], (None, None))
    if not e: continue
    pn, pp = day_pnl(d, e, x)
    if pn is None: continue
    per_day_best.append({
        "date": d["date"], "sym": d["sym"], "dte": d["dte"],
        "gap": d["gap"], "fh_range": d["fh_range"], "dir_30m": d["dir_30m"],
        "rupees": day_rupees(d, pn), "pct": pp,
    })

print(f"\n══════ BEST-COMBO BASELINE (with DTE-optimal entry/exit) ══════")
tot = sum(x["rupees"] for x in per_day_best)
w = sum(1 for x in per_day_best if x["rupees"] > 0)
print(f"Best combo per DTE: {best_combo_per_dte}")
print(f"n={len(per_day_best)}  wins={w}/{len(per_day_best)} ({w/len(per_day_best)*100:.0f}%)  cum=₹{tot:+,.0f}  avg=₹{tot/len(per_day_best):+,.0f}")


# ── Step 6: try filters ───────────────────────────────────────
def apply_filter(per_day, predicate):
    kept = [x for x in per_day if predicate(x)]
    if not kept: return None
    tot = sum(x["rupees"] for x in kept)
    w = sum(1 for x in kept if x["rupees"] > 0)
    return {"n": len(kept), "skipped": len(per_day) - len(kept),
            "wins": w, "win_pct": w/len(kept)*100,
            "cum": tot, "avg": tot/len(kept)}


print(f"\n══════ FILTER SWEEP — single filter ══════")
print(f"  {'filter':45} {'kept':>4} {'skip':>4} {'win%':>5} {'cum ₹':>10} {'avg ₹':>8}")

filters = [
    ("baseline (no filter)",                     lambda x: True),
    ("|gap%| < 0.3",                              lambda x: abs(x["gap"]) < 0.3),
    ("|gap%| < 0.5",                              lambda x: abs(x["gap"]) < 0.5),
    ("|gap%| < 0.8",                              lambda x: abs(x["gap"]) < 0.8),
    ("|gap%| < 1.0",                              lambda x: abs(x["gap"]) < 1.0),
    ("gap up < 0.5 (allow any down)",             lambda x: x["gap"] < 0.5),
    ("fh_range < 0.4%",                           lambda x: x["fh_range"] is not None and x["fh_range"] < 0.4),
    ("fh_range < 0.5%",                           lambda x: x["fh_range"] is not None and x["fh_range"] < 0.5),
    ("fh_range < 0.6%",                           lambda x: x["fh_range"] is not None and x["fh_range"] < 0.6),
    ("fh_range < 0.8%",                           lambda x: x["fh_range"] is not None and x["fh_range"] < 0.8),
    ("|dir_30m| < 0.2%",                          lambda x: abs(x["dir_30m"]) < 0.2),
    ("|dir_30m| < 0.3%",                          lambda x: abs(x["dir_30m"]) < 0.3),
    ("|dir_30m| < 0.5%",                          lambda x: abs(x["dir_30m"]) < 0.5),
]
for name, pred in filters:
    s = apply_filter(per_day_best, pred)
    if s is None: continue
    print(f"  {name:45} {s['n']:>4} {s['skipped']:>4} {s['win_pct']:>4.0f}% "
          f"{s['cum']:>+9,.0f} {s['avg']:>+7,.0f}")


print(f"\n══════ FILTER SWEEP — combined (gap + fh_range) ══════")
print(f"  {'filter':45} {'kept':>4} {'skip':>4} {'win%':>5} {'cum ₹':>10} {'avg ₹':>8}")
combined = [
    (f"gap<{g}, fh<{r}",
     lambda x, g=g, r=r: abs(x["gap"])<g and x["fh_range"] is not None and x["fh_range"]<r)
    for g in (0.5, 0.8, 1.0) for r in (0.5, 0.6, 0.8)
]
for name, pred in combined:
    s = apply_filter(per_day_best, pred)
    if s is None: continue
    print(f"  {name:45} {s['n']:>4} {s['skipped']:>4} {s['win_pct']:>4.0f}% "
          f"{s['cum']:>+9,.0f} {s['avg']:>+7,.0f}")


# ── Step 7: 5 worst days with features ────────────────────────
per_day_best.sort(key=lambda x: x["rupees"])
print(f"\n══════ 10 WORST DAYS (with features) ══════")
print(f"  {'date':12} {'sym':6} dte {'gap%':>6} {'fh_rg%':>7} {'dir30m%':>8} ₹P&L")
for x in per_day_best[:10]:
    print(f"  {x['date']:12} {x['sym']:6}  {x['dte']}  "
          f"{x['gap']:>+5.2f} {x['fh_range'] or 0:>6.2f} {x['dir_30m']:>+7.2f}  "
          f"₹{x['rupees']:+,.0f}")

# save full
json.dump({"per_day": per_day_best, "best_combo": best_combo_per_dte},
          open("retune_results.json", "w"), indent=2, default=str)
print("\nSaved retune_results.json")
