"""strategy_bakeoff_v3.py — explore refinements on v2's ALL-double / Tue+Wed-double winners."""
from collections import defaultdict
from datetime import datetime, date
from strategy_bakeoff import (day_pnl, fmt_rs, B, psql)

START, END = "2026-01-01", "2026-05-22"
rows = psql(f"SELECT DISTINCT date FROM option_candles WHERE date BETWEEN '{START}' AND '{END}' ORDER BY date;")
days = sorted({datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows})
print(f"Days: {len(days)}", flush=True)

def is_last_week(d, ad): return not any(x.month == d.month and x > d for x in ad)
def is_first_week(d):    return d.day <= 7

def mon_pnl(d, h):
    iso = d.isocalendar()
    for past_d, p in h.items():
        if past_d.isocalendar()[:2] == iso[:2] and past_d.weekday() == 0:
            return p
    return None

def wtd(d, h):
    iso = d.isocalendar()
    return sum(p for pd, p in h.items() if pd.isocalendar()[:2] == iso[:2] and pd < d)

def v_3a(d, wd, h, ad):
    s = B.get(wd); return [s] if s else []

# ─── re-validate top v2 winners ───
def v_all_double(d, wd, h, ad):
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0), ("SENSEX",3,"09:20","11:00",0)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0), ("NIFTY",0,"10:00","15:15",0)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0), ("SENSEX",1,"10:00","13:00",0)]
    return []

def v_tue_wed_mon_widen(d, wd, h, ad):
    if wd == 1: base = [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    elif wd == 2: base = [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    else:
        s = B.get(wd); base = [s] if s else []
    mp = mon_pnl(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

# ─── ALL-double + adaptive defenses ───
def v_alldbl_mon_widen(d, wd, h, ad):
    base = v_all_double(d, wd, h, ad)
    mp = mon_pnl(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

def v_alldbl_fwk_atm(d, wd, h, ad):
    base = v_all_double(d, wd, h, ad)
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    return base

def v_alldbl_lwk_p1(d, wd, h, ad):
    base = v_all_double(d, wd, h, ad)
    if is_last_week(d, ad):
        base = [(s,o+1,e,x,r) for (s,o,e,x,r) in base]
    return base

def v_alldbl_lwk_p2(d, wd, h, ad):
    base = v_all_double(d, wd, h, ad)
    if is_last_week(d, ad):
        base = [(s,o+2,e,x,r) for (s,o,e,x,r) in base]
    return base

def v_alldbl_wtd_stop(d, wd, h, ad):
    """If week-to-date < -10k, skip rest of week."""
    if wd >= 1 and wtd(d, h) < -10000:
        return []
    return v_all_double(d, wd, h, ad)

def v_alldbl_wtd_widen(d, wd, h, ad):
    """If week-to-date < -10k, widen rest of week to +3."""
    base = v_all_double(d, wd, h, ad)
    if wd >= 1 and wtd(d, h) < -10000:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

def v_alldbl_full_stack(d, wd, h, ad):
    """ALL-double + Mon-loss-widen + first-wk ATM + last-wk +1."""
    base = v_all_double(d, wd, h, ad)
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    if is_last_week(d, ad):
        base = [(s,o+1,e,x,r) for (s,o,e,x,r) in base]
    mp = mon_pnl(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

# ─── Fri experiments (Fri was NIFTY+1 only) ───
def v_alldbl_fri_atm(d, wd, h, ad):
    base = v_all_double(d, wd, h, ad)
    if wd == 4: base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    return base

def v_alldbl_fri_skip(d, wd, h, ad):
    if wd == 4: return []
    return v_all_double(d, wd, h, ad)

def v_alldbl_fri_p2(d, wd, h, ad):
    base = v_all_double(d, wd, h, ad)
    if wd == 4: base = [(s,2,e,x,r) for (s,o,e,x,r) in base]
    return base

# ─── Mon experiments ───
def v_alldbl_mon_atm(d, wd, h, ad):
    if wd == 0: return [("NIFTY",0,"09:20","11:00",0), ("SENSEX",0,"09:20","11:00",0)]
    return v_all_double(d, wd, h, ad)

def v_alldbl_mon_skip(d, wd, h, ad):
    if wd == 0: return []
    return v_all_double(d, wd, h, ad)

# ─── Tue mega: triple+quad ───
def v_tue_quad_all_dbl(d, wd, h, ad):
    """ALL-days DOUBLE + extra NIFTY late Tue window."""
    base = v_all_double(d, wd, h, ad)
    if wd == 1:
        base.append(("NIFTY",1,"10:30","13:30",0))
    return base

def v_tue_triple_wed_thu_dbl(d, wd, h, ad):
    """Tue triple + Wed-double + Thu-double + Mon-default + Fri-default."""
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0),
                        ("NIFTY",1,"10:30","13:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0), ("NIFTY",0,"10:00","15:15",0)]
    s = B.get(wd); return [s] if s else []

# ─── all-double with different Wed/Thu offsets ───
def v_alldbl_wed_thu_p1(d, wd, h, ad):
    if wd == 2: return [("SENSEX",1,"09:45","15:00",0), ("NIFTY",1,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",1,"10:00","15:15",0), ("NIFTY",1,"10:00","15:15",0)]
    return v_all_double(d, wd, h, ad)

def v_alldbl_wed_thu_p2(d, wd, h, ad):
    if wd == 2: return [("SENSEX",2,"09:45","15:00",0), ("NIFTY",2,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",2,"10:00","15:15",0), ("NIFTY",2,"10:00","15:15",0)]
    return v_all_double(d, wd, h, ad)

VARIANTS = [
    ("3a baseline",                       v_3a),
    ("ALL-days DOUBLE (v2 winner)",       v_all_double),
    ("Tue+Wed dbl + Mon-loss-widen",      v_tue_wed_mon_widen),
    ("ALL-dbl + Mon-loss-widen",          v_alldbl_mon_widen),
    ("ALL-dbl + first-wk ATM",            v_alldbl_fwk_atm),
    ("ALL-dbl + last-wk +1 wider",        v_alldbl_lwk_p1),
    ("ALL-dbl + last-wk +2 wider",        v_alldbl_lwk_p2),
    ("ALL-dbl + WTD-stop -10K",           v_alldbl_wtd_stop),
    ("ALL-dbl + WTD-widen -10K",          v_alldbl_wtd_widen),
    ("ALL-dbl FULL STACK",                v_alldbl_full_stack),
    ("ALL-dbl + Fri ATM",                 v_alldbl_fri_atm),
    ("ALL-dbl + Fri SKIP",                v_alldbl_fri_skip),
    ("ALL-dbl + Fri +2",                  v_alldbl_fri_p2),
    ("ALL-dbl + Mon ATM",                 v_alldbl_mon_atm),
    ("ALL-dbl + Mon SKIP",                v_alldbl_mon_skip),
    ("ALL-dbl + Tue extra NIFTY",         v_tue_quad_all_dbl),
    ("Tue-triple + Wed-dbl + Thu-dbl",    v_tue_triple_wed_thu_dbl),
    ("ALL-dbl + Wed/Thu +1",              v_alldbl_wed_thu_p1),
    ("ALL-dbl + Wed/Thu +2",              v_alldbl_wed_thu_p2),
]

results = {}
for name, fn in VARIANTS:
    print(f"  {name}...", flush=True)
    day_total = {}; history = {}
    n_trades = n_wins = 0
    monthly = defaultdict(float)
    for d in days:
        wd = d.weekday()
        if wd > 4: continue
        specs = fn(d, wd, history, days)
        if not specs:
            history[d] = 0.0; continue
        tot = 0.0; any_ok = False
        for sp in specs:
            p = day_pnl(d, sp)
            if p is None: continue
            tot += p; any_ok = True
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
                         best=best, avg=avg, monthly=dict(monthly))

baseline = results["3a baseline"]["total"]
ranked = sorted(results.items(), key=lambda kv: -kv[1]["total"])

print()
print(f"══════ Baseline 3a: {fmt_rs(baseline)} ══════")
print()
print(f"{'Variant':<36}{'Tr':>5}{'Win%':>6}{'Total':>14}{'Avg/d':>10}{'Worst':>11}{'Best':>11}{'vs 3a':>13}")
print("-"*106)
for name, r in ranked:
    diff = r["total"] - baseline
    flag = " ★" if diff > 0 else ""
    print(f"{name:<36}{r['trades']:>5}{r['win']:>5}%{fmt_rs(r['total']):>14}{fmt_rs(r['avg']):>10}"
          f"{fmt_rs(r['worst']):>11}{fmt_rs(r['best']):>11}{fmt_rs(diff):>13}{flag}")

print()
print("══════ MONTHLY — top 6 ══════")
top6 = ranked[:6]
months = sorted({m for _, r in top6 for m in r["monthly"]})
print(f"{'Variant':<36}" + "".join(f"{m:>13}" for m in months) + f"{'TOTAL':>13}")
print("-"*(36 + 13*(len(months)+1)))
for name, r in top6:
    line = f"{name:<36}" + "".join(f"{fmt_rs(r['monthly'].get(m,0)):>13}" for m in months)
    line += f"{fmt_rs(r['total']):>13}"
    print(line)

print()
print("══════ EVERY MONTH POSITIVE? (top 8) ══════")
for name, r in ranked[:8]:
    mn = min(r['monthly'].values()) if r['monthly'] else 0
    flag = "✓ ALL POSITIVE" if mn > 0 else "✗ has loss month"
    print(f"  {name:<34} min-mo={fmt_rs(mn):>13}   {flag}")

print()
print("══════ RISK (top 8) ══════")
for name, r in ranked[:8]:
    rr = (r['avg'] / abs(r['worst'])) if r['worst'] else 0
    print(f"  {name:<34} avg/worst={rr:6.2f}  worst={fmt_rs(r['worst']):>11}  best={fmt_rs(r['best']):>11}")
