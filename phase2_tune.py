#!/usr/bin/env python3
"""Phase-2 tuning: per-weekday + PT/SL + dynamic intraday exit.

Builds on retune_real_options.py. Loads same intraday cache shape,
then tests:
  A. Per-(weekday, index) entry/exit (finer than per-DTE).
  B. Profit-target % of credit + stop-loss % of credit.
  C. Time-based exit + PT/SL hybrid (whichever first).
  D. Trailing exit (exit when premium decays X% from morning peak).
"""
import csv, subprocess, json
from datetime import datetime
from collections import defaultdict

NIFTY_LOT, SENSEX_LOT = 75, 20
NUM_LOTS = 3
PSQL = ["docker", "exec", "-i", "tradeai-postgres",
        "psql", "-U", "tradeai", "-d", "tradeai", "-A", "-F", "|", "-t", "-c"]

def q(sql):
    r = subprocess.run(PSQL + [sql], capture_output=True, text=True, timeout=120)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    return [line.split("|") for line in r.stdout.strip().split("\n") if line.strip()]

def to_min(h): return int(h.split(":")[0])*60 + int(h.split(":")[1])
def round_strike(spot, sym): step = 50 if sym=="NIFTY" else 100; return int(round(spot/step)*step)

# ── Load days (re-use logic) ─────────────────────────────────
schedule = list(csv.DictReader(open("swap_schedule.csv")))
print(f"Loading {len(schedule)} days from DB...")
days = []
for i, r in enumerate(schedule, 1):
    d, sym = r["date"], r["sym"]
    rs = q(f"SELECT close FROM index_candles WHERE instrument='{sym}' AND date='{d}' "
           f"AND to_char(timestamp,'HH24:MI')='09:20' LIMIT 1;")
    if not rs:
        rs = q(f"SELECT close FROM index_candles WHERE instrument='{sym}' AND date='{d}' ORDER BY timestamp LIMIT 1;")
    if not rs: continue
    spot920 = float(rs[0][0])
    strike = round_strike(spot920, sym)
    rs = q(f"SELECT DISTINCT expiry FROM option_candles WHERE instrument='{sym}' AND date='{d}';")
    if not rs: continue
    target = datetime.strptime(d, "%Y-%m-%d").date()
    parsed = sorted([(datetime.strptime(e[0],"%d%b%y").date(), e[0]) for e in rs if len(e[0])==7])
    expiry = next((e for dd, e in parsed if dd >= target), parsed[0][1] if parsed else None)
    if not expiry: continue
    rs = q(f"SELECT to_char(timestamp,'HH24:MI'), option_type, close FROM option_candles "
           f"WHERE instrument='{sym}' AND date='{d}' AND expiry='{expiry}' AND strike={strike} "
           f"AND option_type IN ('CE','PE') ORDER BY timestamp;")
    if not rs: continue
    ce = {}; pe = {}
    for hhmm, ot, c in rs:
        (ce if ot=="CE" else pe)[hhmm] = float(c)
    days.append({
        "date": d, "sym": sym, "dte": int(r["dte"]),
        "weekday": datetime.strptime(d, "%Y-%m-%d").strftime("%a"),
        "spot920": spot920, "ce": ce, "pe": pe,
    })
    if i % 10 == 0: print(f"  [{i}/{len(schedule)}]")
print(f"Usable: {len(days)}\n")

# ── helpers ──────────────────────────────────────────────────
def get_at(bars, hhmm):
    if hhmm in bars: return bars[hhmm]
    le = sorted(t for t in bars if to_min(t) <= to_min(hhmm))
    if le: return bars[le[-1]]
    ge = sorted(t for t in bars if to_min(t) >= to_min(hhmm))
    return bars[ge[0]] if ge else None

def straddle_at(day, hhmm):
    c = get_at(day["ce"], hhmm); p = get_at(day["pe"], hhmm)
    return None if c is None or p is None else c + p

def lot_size(sym): return NIFTY_LOT if sym=="NIFTY" else SENSEX_LOT

def to_rupees(day, pts): return pts * lot_size(day["sym"]) * NUM_LOTS

def stats(rs):
    if not rs: return None
    w = sum(1 for r in rs if r > 0)
    return {"n": len(rs), "w": w, "win": w/len(rs)*100,
            "cum": sum(rs), "avg": sum(rs)/len(rs),
            "min": min(rs), "max": max(rs)}

# ── A. Per-WEEKDAY entry/exit ─────────────────────────────────
print("══════ A. Per-WEEKDAY entry/exit sweep ══════")
ENTRIES = ["09:20","09:30","09:45","10:00","10:15","10:30","10:45"]
EXITS   = ["11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]
best_by_wd = {}
for wd in ("Mon","Tue","Wed","Thu","Fri"):
    sub = [d for d in days if d["weekday"]==wd]
    if not sub: continue
    sym0 = sub[0]["sym"]; dte0 = sub[0]["dte"]
    results = []
    for e in ENTRIES:
        for x in EXITS:
            if to_min(x) <= to_min(e) + 60: continue
            rs = []
            for d in sub:
                c = straddle_at(d, e); o = straddle_at(d, x)
                if c is None or o is None: continue
                rs.append(to_rupees(d, c - o))
            s = stats(rs)
            if s: results.append((e, x, s))
    results.sort(key=lambda r: r[2]["cum"], reverse=True)
    print(f"\n  {wd} ({sym0} DTE-{dte0}, {len(sub)} days)")
    for i, (e, x, s) in enumerate(results[:3], 1):
        print(f"    #{i} {e}->{x}  cum=₹{s['cum']:+,.0f}  win={s['win']:.0f}%  avg=₹{s['avg']:+,.0f}  min=₹{s['min']:+,.0f}")
    best_by_wd[wd] = (results[0][0], results[0][1])

print(f"\n  Best per-weekday: {best_by_wd}")

# Compute combined P&L with per-weekday best
total_a = 0
for d in days:
    e, x = best_by_wd[d["weekday"]]
    c = straddle_at(d, e); o = straddle_at(d, x)
    if c is None or o is None: continue
    total_a += to_rupees(d, c - o)
print(f"  TOTAL (per-weekday best): ₹{total_a:+,.0f}\n")


# ── B. Profit-target + stop-loss using 1-min path ────────────
print("══════ B. PT/SL exit (per-weekday entries, walk 1-min bars) ══════")
def simulate_with_ptsl(day, entry, exit_t, pt_pct, sl_pct):
    """Walk bars from entry to exit, exit early if PT or SL hit. Return rupees."""
    c0 = straddle_at(day, entry)
    if c0 is None: return None
    bars = sorted(set(day["ce"].keys()) | set(day["pe"].keys()))
    bars = [b for b in bars if to_min(entry) <= to_min(b) <= to_min(exit_t)]
    pt_value = c0 * (1 - pt_pct/100)   # debit to close = credit × (1 - target%)
    sl_value = c0 * (1 + sl_pct/100)   # debit to close = credit × (1 + stop%)
    for b in bars:
        v = straddle_at(day, b)
        if v is None: continue
        if v <= pt_value or v >= sl_value:
            return to_rupees(day, c0 - v)
    # exit at exit_t
    vx = straddle_at(day, exit_t)
    return to_rupees(day, c0 - vx) if vx is not None else None

# Sweep PT/SL grid using per-weekday best entry/exit as baseline
print(f"  {'PT%':>5} {'SL%':>5} {'n':>3} {'win%':>5} {'cum ₹':>11} {'avg ₹':>8}")
for pt in (20, 30, 40, 50, 60, 70, 80):
    for sl in (60, 80, 100, 120, 150, 200):
        rs = []
        for d in days:
            e, x = best_by_wd[d["weekday"]]
            r = simulate_with_ptsl(d, e, x, pt, sl)
            if r is not None: rs.append(r)
        s = stats(rs)
        if s and s["cum"] > total_a * 0.9:  # only show if competitive
            print(f"  {pt:>4}% {sl:>4}% {s['n']:>3} {s['win']:>4.0f}% "
                  f"{s['cum']:>+10,.0f} {s['avg']:>+7,.0f}")

# Best PT/SL search
print(f"\n  Searching best PT/SL grid (max cum)...")
best_ptsl = None
for pt in range(20, 91, 5):
    for sl in range(50, 301, 10):
        rs = []
        for d in days:
            e, x = best_by_wd[d["weekday"]]
            r = simulate_with_ptsl(d, e, x, pt, sl)
            if r is not None: rs.append(r)
        s = stats(rs)
        if s and (best_ptsl is None or s["cum"] > best_ptsl[2]["cum"]):
            best_ptsl = (pt, sl, s)
pt, sl, s = best_ptsl
print(f"  BEST: PT={pt}% SL={sl}% → cum=₹{s['cum']:+,.0f} win={s['win']:.0f}% avg=₹{s['avg']:+,.0f}")
print(f"  vs no PT/SL baseline:  ₹{total_a:+,.0f}")
print(f"  Improvement: ₹{s['cum']-total_a:+,.0f} ({(s['cum']-total_a)/abs(total_a)*100:+.1f}%)\n")


# ── C. Per-weekday PT/SL ─────────────────────────────────────
print("══════ C. Per-WEEKDAY best PT/SL (tighter) ══════")
total_c = 0
ptsl_by_wd = {}
for wd in ("Mon","Tue","Wed","Thu","Fri"):
    sub = [d for d in days if d["weekday"]==wd]
    if not sub: continue
    e, x = best_by_wd[wd]
    best = None
    for pt in range(20, 91, 5):
        for sl in range(50, 301, 10):
            rs = []
            for d in sub:
                r = simulate_with_ptsl(d, e, x, pt, sl)
                if r is not None: rs.append(r)
            s = stats(rs)
            if s and (best is None or s["cum"] > best[2]["cum"]):
                best = (pt, sl, s)
    pt, sl, s = best
    ptsl_by_wd[wd] = (pt, sl)
    total_c += s["cum"]
    print(f"  {wd} ({e}->{x}): PT={pt}% SL={sl}% n={s['n']} win={s['win']:.0f}% "
          f"cum=₹{s['cum']:+,.0f}  avg=₹{s['avg']:+,.0f}  min=₹{s['min']:+,.0f}")
print(f"\n  TOTAL (per-weekday best entry+exit+PT/SL): ₹{total_c:+,.0f}")
print(f"  vs Phase-A baseline ₹{total_a:+,.0f}: ₹{total_c-total_a:+,.0f} "
      f"({(total_c-total_a)/abs(total_a)*100:+.1f}%)\n")


# ── D. Compare all configs ───────────────────────────────────
print("══════ FINAL COMPARISON ══════")
configs = [
    ("Old recommendation (DTE-based, no PT/SL)", 183483),
    ("Phase-1: DTE-optimised entry/exit",         250185),
    ("A. Per-weekday entry/exit",                 total_a),
    ("B. Best global PT/SL on A",                 best_ptsl[2]["cum"]),
    ("C. Per-weekday PT/SL",                      total_c),
]
for name, v in configs:
    print(f"  {name:50} ₹{v:+,.0f}")

# save
json.dump({
    "best_by_wd": {k: list(v) for k, v in best_by_wd.items()},
    "ptsl_by_wd": {k: list(v) for k, v in ptsl_by_wd.items()},
    "totals": {n: v for n, v in configs},
}, open("phase2_results.json","w"), indent=2)
