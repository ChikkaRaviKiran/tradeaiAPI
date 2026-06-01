"""strategy_bakeoff_v6.py — capital-constrained bake-off (cap = ₹4.4L peak concurrent margin).

For each variant we compute peak concurrent margin across the day. If peak > cap,
that day is SKIPPED (infeasible). Otherwise PnL is scaled by the per-lot count.
"""
from collections import defaultdict
from datetime import datetime
from strategy_bakeoff import (day_pnl, fmt_rs, B, psql)

START, END = "2026-01-01", "2026-05-22"
CAP = 4_40_000  # ₹4.4 lakh capital cap

rows = psql(f"SELECT DISTINCT date FROM option_candles WHERE date BETWEEN '{START}' AND '{END}' ORDER BY date;")
days = sorted({datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows})
print(f"Days: {len(days)}  |  Capital cap: ₹{CAP:,}", flush=True)

# ───── Approx SPAN+exposure margin per lot (₹) for short option leg ─────
# ATM-most-expensive, drops slightly as we go OTM
MARGIN_PER_LOT = {  # (sym, abs_offset_steps) -> ₹/lot for SHORT STRANGLE (both legs)
    ("NIFTY", 0): 1_10_000,
    ("NIFTY", 1): 1_05_000,
    ("NIFTY", 2):   95_000,
    ("NIFTY", 3):   90_000,
    ("SENSEX", 0): 1_10_000,
    ("SENSEX", 1): 1_05_000,
    ("SENSEX", 2):   95_000,
    ("SENSEX", 3):   90_000,
}
def margin_for(sym, off, lots):
    return MARGIN_PER_LOT.get((sym, abs(off)), 1_10_000) * lots

def to_min(t):
    h, m = t.split(":"); return int(h)*60 + int(m)

def peak_concurrent_margin(specs):
    """specs: list[(sym, off, e_s, x_s, rk, lots)]. Returns peak margin needed at any minute."""
    events = []
    for (sym, off, e_s, x_s, rk, lots) in specs:
        m = margin_for(sym, off, lots)
        events.append((to_min(e_s), +m))
        events.append((to_min(x_s), -m))
    events.sort()
    cur = 0; peak = 0
    for (_, dm) in events:
        cur += dm
        if cur > peak: peak = cur
    return peak

def per_lot_pnl(d, sym, off, e_s, x_s, rk):
    """day_pnl uses LOTS=3 hardcoded; per-lot = total/3."""
    p = day_pnl(d, (sym, off, e_s, x_s, rk))
    return None if p is None else p / 3.0

# ───── variant decision functions ─────
# Each returns list[(sym, off, e_s, x_s, rk, lots)]

# === Group 1: single-index baseline & variants (max 3 lots one index) ===
def v_3a_orig(d, wd, h, ad):
    """Current production 3a: single index per day at 3 lots."""
    s = B.get(wd); 
    if not s: return []
    sym, off, e, x, rk = s
    return [(sym, off, e, x, rk, 3)]

def v_all_nifty_3(d, wd, h, ad):
    """All days NIFTY 3 lots same window as 3a."""
    s = B.get(wd)
    if not s: return []
    _, off, e, x, rk = s
    return [("NIFTY", off, e, x, rk, 3)]

def v_all_sensex_3(d, wd, h, ad):
    """All days SENSEX 3 lots same window as 3a."""
    s = B.get(wd)
    if not s: return []
    _, off, e, x, rk = s
    return [("SENSEX", off, e, x, rk, 3)]

# === Group 2: dual 1+1 lots ===
def v_dual_1_1_3a_sched(d, wd, h, ad):
    """3a schedule but ADD the other index at 1 lot each (so both at 1 lot)."""
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0,1), ("SENSEX",3,"09:20","11:00",0,1)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0,1), ("SENSEX",0,"09:30","14:30",0,1)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0,1), ("NIFTY",0,"09:45","15:00",0,1)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0,1), ("NIFTY",0,"10:00","15:15",0,1)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0,1), ("SENSEX",1,"10:00","13:00",0,1)]
    return []

def v_dual_1_1_v5_full(d, wd, h, ad):
    """v5 winner (TQ + Wed-extra + Mon ATM) at 1+1 lots."""
    if wd == 0: return [("NIFTY",0,"09:20","11:00",0,1), ("SENSEX",0,"09:20","11:00",0,1)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0,1), ("SENSEX",0,"09:30","14:30",0,1),
                        ("NIFTY",1,"10:30","13:30",0,1), ("SENSEX",1,"10:30","13:30",0,1)]
    if wd == 2: return [("NIFTY",0,"09:45","15:00",0,1), ("SENSEX",0,"09:45","15:00",0,1),
                        ("NIFTY",1,"10:30","13:30",0,1)]
    if wd == 3: return [("NIFTY",0,"10:00","15:15",0,1), ("SENSEX",0,"10:00","15:15",0,1)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0,1), ("SENSEX",1,"10:00","13:00",0,1)]
    return []

# === Group 3: dual 2+1 / 1+2 (NIFTY or SENSEX heavy) ===
def v_dual_2_1_nifty_heavy(d, wd, h, ad):
    """3a schedule's primary index gets 2 lots, secondary 1 lot. Primary chosen as 'NIFTY-heavy'."""
    # Mon/Tue/Fri: NIFTY 2 + SENSEX 1  ;  Wed/Thu: SENSEX 1 + NIFTY 2 (NIFTY still heavy)
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0,2), ("SENSEX",3,"09:20","11:00",0,1)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0,2), ("SENSEX",0,"09:30","14:30",0,1)]
    if wd == 2: return [("NIFTY",0,"09:45","15:00",0,2), ("SENSEX",0,"09:45","15:00",0,1)]
    if wd == 3: return [("NIFTY",0,"10:00","15:15",0,2), ("SENSEX",0,"10:00","15:15",0,1)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0,2), ("SENSEX",1,"10:00","13:00",0,1)]
    return []

def v_dual_1_2_sensex_heavy(d, wd, h, ad):
    """SENSEX 2 + NIFTY 1 every day (3a schedule)."""
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0,1), ("SENSEX",3,"09:20","11:00",0,2)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0,1), ("SENSEX",0,"09:30","14:30",0,2)]
    if wd == 2: return [("NIFTY",0,"09:45","15:00",0,1), ("SENSEX",0,"09:45","15:00",0,2)]
    if wd == 3: return [("NIFTY",0,"10:00","15:15",0,1), ("SENSEX",0,"10:00","15:15",0,2)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0,1), ("SENSEX",1,"10:00","13:00",0,2)]
    return []

# === Group 4: dual 2+2 ATM single window (peak ≈ ₹4.4L) ===
def v_dual_2_2_atm(d, wd, h, ad):
    """N2 + S2 ATM single window per day (no extras to keep margin in budget)."""
    if wd == 0: return [("NIFTY",0,"09:20","11:00",0,2), ("SENSEX",0,"09:20","11:00",0,2)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0,2), ("SENSEX",0,"09:30","14:30",0,2)]
    if wd == 2: return [("NIFTY",0,"09:45","15:00",0,2), ("SENSEX",0,"09:45","15:00",0,2)]
    if wd == 3: return [("NIFTY",0,"10:00","15:15",0,2), ("SENSEX",0,"10:00","15:15",0,2)]
    if wd == 4: return [("NIFTY",0,"10:00","13:00",0,2), ("SENSEX",0,"10:00","13:00",0,2)]
    return []

def v_dual_2_2_otm1(d, wd, h, ad):
    """N2 + S2 at +1 OTM (margin slightly lower; could leave headroom)."""
    if wd > 4: return []
    s = B.get(wd)
    if not s: return []
    _, _, e, x, rk = s
    return [("NIFTY",1,e,x,0,2), ("SENSEX",1,e,x,0,2)]

# === Group 5: time-rotation (non-overlapping → 3 lots each leg) ===
def v_rotate_n3_morning_s3_afternoon(d, wd, h, ad):
    """NIFTY 3 lots morning, exits by lunch; then SENSEX 3 lots afternoon. Peak ≈ ₹3.3L."""
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0,3), ("SENSEX",0,"12:00","15:00",0,3)]
    if wd == 1: return [("NIFTY",0,"09:30","12:00",0,3), ("SENSEX",0,"12:15","14:45",0,3)]
    if wd == 2: return [("NIFTY",0,"09:45","12:00",0,3), ("SENSEX",0,"12:15","15:00",0,3)]
    if wd == 3: return [("NIFTY",0,"10:00","12:00",0,3), ("SENSEX",0,"12:15","15:15",0,3)]
    if wd == 4: return [("NIFTY",1,"10:00","12:00",0,3), ("SENSEX",1,"12:15","14:45",0,3)]
    return []

def v_rotate_s3_morning_n3_afternoon(d, wd, h, ad):
    """SENSEX morning, NIFTY afternoon. Peak ≈ ₹3.3L."""
    if wd == 0: return [("SENSEX",3,"09:20","11:00",0,3), ("NIFTY",0,"12:00","15:00",0,3)]
    if wd == 1: return [("SENSEX",0,"09:30","12:00",0,3), ("NIFTY",0,"12:15","14:45",0,3)]
    if wd == 2: return [("SENSEX",0,"09:45","12:00",0,3), ("NIFTY",0,"12:15","15:00",0,3)]
    if wd == 3: return [("SENSEX",0,"10:00","12:00",0,3), ("NIFTY",0,"12:15","15:15",0,3)]
    if wd == 4: return [("SENSEX",1,"10:00","12:00",0,3), ("NIFTY",1,"12:15","14:45",0,3)]
    return []

# === Group 6: per-day best (no concurrent) — single index at 3 lots, chosen index per day from ultimate sweep ===
# Already encoded by v_3a_orig (mon NIFTY, tue NIFTY, wed SENSEX, thu SENSEX, fri NIFTY). Equivalent.

# === Group 7: hybrid — use rotation only on Tue (the biggest contributor) ===
def v_3a_plus_tue_rotation(d, wd, h, ad):
    """Use 3a everywhere; on Tue add SENSEX afternoon at 3 lots after NIFTY morning ends."""
    if wd == 1:
        return [("NIFTY",0,"09:30","12:00",0,3), ("SENSEX",0,"12:15","14:45",0,3)]
    s = B.get(wd)
    if not s: return []
    sym, off, e, x, rk = s
    return [(sym, off, e, x, rk, 3)]

def v_3a_plus_wed_rotation(d, wd, h, ad):
    """3a + on Wed: SENSEX morning + NIFTY afternoon rotation."""
    if wd == 2:
        return [("SENSEX",0,"09:45","12:00",0,3), ("NIFTY",0,"12:15","15:00",0,3)]
    s = B.get(wd); 
    if not s: return []
    sym, off, e, x, rk = s
    return [(sym, off, e, x, rk, 3)]

def v_3a_full_rotation_all_days(d, wd, h, ad):
    """3a's primary index morning at 3 lots, then OTHER index afternoon at 3 lots, every day."""
    return v_rotate_n3_morning_s3_afternoon(d, wd, h, ad) if wd in (0,1,4) else v_rotate_s3_morning_n3_afternoon(d, wd, h, ad)

# === Group 8: 1+1 + Tue rotation (best of dual + rotation) ===
def v_dual_1_1_plus_tue_rotation(d, wd, h, ad):
    """Both indexes 1 lot per 3a schedule, but on Tue use rotation at 3 lots each (non-overlap)."""
    if wd == 1:
        return [("NIFTY",0,"09:30","12:00",0,3), ("SENSEX",0,"12:15","14:45",0,3)]
    return v_dual_1_1_3a_sched(d, wd, h, ad)

# === Group 9: dual 2+1 with Tue rotation ===
def v_2_1_plus_tue_rotation(d, wd, h, ad):
    """NIFTY 2 + SENSEX 1 normal days; Tue rotation N3+S3."""
    if wd == 1:
        return [("NIFTY",0,"09:30","12:00",0,3), ("SENSEX",0,"12:15","14:45",0,3)]
    return v_dual_2_1_nifty_heavy(d, wd, h, ad)

VARIANTS = [
    ("3a ORIG (single-idx 3 lots)",          v_3a_orig),
    ("ALL-NIFTY 3 lots",                     v_all_nifty_3),
    ("ALL-SENSEX 3 lots",                    v_all_sensex_3),
    ("DUAL 1+1 — 3a schedule",               v_dual_1_1_3a_sched),
    ("DUAL 1+1 — v5 full schedule",          v_dual_1_1_v5_full),
    ("DUAL 2+1 — NIFTY heavy",               v_dual_2_1_nifty_heavy),
    ("DUAL 1+2 — SENSEX heavy",              v_dual_1_2_sensex_heavy),
    ("DUAL 2+2 — ATM single window",         v_dual_2_2_atm),
    ("DUAL 2+2 — +1 OTM single window",      v_dual_2_2_otm1),
    ("ROTATE N3-morn + S3-aft",              v_rotate_n3_morning_s3_afternoon),
    ("ROTATE S3-morn + N3-aft",              v_rotate_s3_morning_n3_afternoon),
    ("3a + Tue ROTATION (N3+S3)",            v_3a_plus_tue_rotation),
    ("3a + Wed ROTATION (S3+N3)",            v_3a_plus_wed_rotation),
    ("FULL ROTATION (every day)",            v_3a_full_rotation_all_days),
    ("DUAL 1+1 + Tue ROTATION",              v_dual_1_1_plus_tue_rotation),
    ("DUAL 2+1 + Tue ROTATION",              v_2_1_plus_tue_rotation),
]

results = {}
for name, fn in VARIANTS:
    print(f"  {name}...", flush=True)
    day_total = {}; history = {}
    n_trades = n_wins = 0
    n_infeasible = 0
    monthly = defaultdict(float)
    max_margin_seen = 0
    for d in days:
        wd = d.weekday()
        if wd > 4: continue
        specs = fn(d, wd, history, days)
        if not specs:
            history[d] = 0.0; continue
        # check capital cap
        peak = peak_concurrent_margin(specs)
        if peak > max_margin_seen: max_margin_seen = peak
        if peak > CAP:
            n_infeasible += 1
            history[d] = 0.0
            continue
        tot = 0.0; any_ok = False
        for (sym, off, e, x, rk, lots) in specs:
            pl = per_lot_pnl(d, sym, off, e, x, rk)
            if pl is None: continue
            tot += pl * lots
            any_ok = True
        if any_ok:
            day_total[d] = tot; history[d] = tot
            monthly[d.strftime("%Y-%m")] += tot
            n_trades += 1
            if tot > 0: n_wins += 1
        else:
            history[d] = 0.0
    total = sum(day_total.values())
    worst = min(day_total.values()) if day_total else 0
    best  = max(day_total.values()) if day_total else 0
    win   = round(100*n_wins/n_trades) if n_trades else 0
    avg   = total/n_trades if n_trades else 0
    results[name] = dict(total=total, trades=n_trades, win=win, worst=worst,
                         best=best, avg=avg, monthly=dict(monthly),
                         peak_margin=max_margin_seen, infeasible=n_infeasible)

baseline = results["3a ORIG (single-idx 3 lots)"]["total"]
ranked = sorted(results.items(), key=lambda kv: -kv[1]["total"])

print()
print(f"══════ Baseline 3a (3 lots single-idx): {fmt_rs(baseline)}  |  Cap: ₹{CAP:,} ══════")
print()
print(f"{'Variant':<38}{'PeakMar':>10}{'Tr':>5}{'Win%':>6}{'Total':>14}{'Avg/d':>10}{'Worst':>11}{'Best':>11}{'vs 3a':>13}")
print("-"*118)
for name, r in ranked:
    diff = r["total"] - baseline
    flag = " ★" if diff > 0 else ""
    pm = f"₹{r['peak_margin']/100000:.2f}L"
    print(f"{name:<38}{pm:>10}{r['trades']:>5}{r['win']:>5}%{fmt_rs(r['total']):>14}"
          f"{fmt_rs(r['avg']):>10}{fmt_rs(r['worst']):>11}{fmt_rs(r['best']):>11}{fmt_rs(diff):>13}{flag}")

print()
print("══════ MONTHLY — top 6 ══════")
top = ranked[:6]
months = sorted({m for _, r in top for m in r["monthly"]})
print(f"{'Variant':<38}" + "".join(f"{m:>13}" for m in months) + f"{'TOTAL':>13}")
print("-"*(38 + 13*(len(months)+1)))
for name, r in top:
    line = f"{name:<38}" + "".join(f"{fmt_rs(r['monthly'].get(m,0)):>13}" for m in months)
    line += f"{fmt_rs(r['total']):>13}"
    print(line)

print()
print("══════ EVERY MONTH POSITIVE? (top 8) ══════")
for name, r in ranked[:8]:
    mn = min(r['monthly'].values()) if r['monthly'] else 0
    flag = "✓ ALL POSITIVE" if mn > 0 else "✗ has loss month"
    print(f"  {name:<36} min-mo={fmt_rs(mn):>13}   {flag}")

print()
print("══════ ROI on ₹4.4L capital (top 8) ══════")
for name, r in ranked[:8]:
    monthly_avg = r['total'] / 5  # 5 months sample
    roi_pct = (monthly_avg / CAP) * 100
    print(f"  {name:<36} ₹{monthly_avg:>11,.0f}/mo  =  {roi_pct:>5.2f}% / month on ₹4.4L")
