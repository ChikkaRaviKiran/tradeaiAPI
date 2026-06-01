"""strategy_bakeoff_v5.py — final tuning around Tue QUAD + Wed-extra."""
from collections import defaultdict
from datetime import datetime
from strategy_bakeoff import (day_pnl, fmt_rs, B, psql)

START, END = "2026-01-01", "2026-05-22"
rows = psql(f"SELECT DISTINCT date FROM option_candles WHERE date BETWEEN '{START}' AND '{END}' ORDER BY date;")
days = sorted({datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows})
print(f"Days: {len(days)}", flush=True)

def is_first_week(d): return d.day <= 7
def is_last_week(d, ad): return not any(x.month == d.month and x > d for x in ad)
def mon_pnl(d, h):
    iso = d.isocalendar()
    for past_d, p in h.items():
        if past_d.isocalendar()[:2] == iso[:2] and past_d.weekday() == 0:
            return p
    return None

def v_3a(d, wd, h, ad):
    s = B.get(wd); return [s] if s else []

# Tue QUAD base (the v4 risk-adj winner)
def _tue_quad_base(d, wd):
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0), ("SENSEX",3,"09:20","11:00",0)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0),
                        ("NIFTY",1,"10:30","13:30",0), ("SENSEX",1,"10:30","13:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0), ("NIFTY",0,"10:00","15:15",0)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0), ("SENSEX",1,"10:00","13:00",0)]
    return []

def v_tue_quad(d, wd, h, ad):
    return _tue_quad_base(d, wd)

# Tue QUAD + Mon ATM
def v_tq_mon_atm(d, wd, h, ad):
    if wd == 0: return [("NIFTY",0,"09:20","11:00",0), ("SENSEX",0,"09:20","11:00",0)]
    return _tue_quad_base(d, wd)

# Tue QUAD + Wed extra NIFTY late (no Thu extra)
def v_tq_wed_extra(d, wd, h, ad):
    base = _tue_quad_base(d, wd)
    if wd == 2: base.append(("NIFTY",1,"10:30","13:30",0))
    return base

# Tue QUAD + Wed extra + Mon ATM
def v_tq_wed_mon(d, wd, h, ad):
    base = v_tq_mon_atm(d, wd, h, ad)
    if wd == 2: base.append(("NIFTY",1,"10:30","13:30",0))
    return base

# Tue QUAD + Wed extra + Mon ATM + first-wk ATM
def v_tq_wed_mon_fwk(d, wd, h, ad):
    base = v_tq_wed_mon(d, wd, h, ad)
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    return base

# Try Tue extra at different offset/window
def v_tq_extra_atm(d, wd, h, ad):
    """Tue QUAD but extras at ATM instead of +1."""
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0), ("SENSEX",3,"09:20","11:00",0)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0),
                        ("NIFTY",0,"10:30","13:30",0), ("SENSEX",0,"10:30","13:30",0)]
    return _tue_quad_base(d, wd)

def v_tq_extra_p2(d, wd, h, ad):
    """Tue QUAD extras at +2 (wider)."""
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0),
                        ("NIFTY",2,"10:30","13:30",0), ("SENSEX",2,"10:30","13:30",0)]
    return _tue_quad_base(d, wd)

def v_tq_extra_late(d, wd, h, ad):
    """Tue QUAD extras 11:30-14:00 (later mid-day window)."""
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0),
                        ("NIFTY",1,"11:30","14:00",0), ("SENSEX",1,"11:30","14:00",0)]
    return _tue_quad_base(d, wd)

def v_tq_extra_short(d, wd, h, ad):
    """Tue QUAD extras 10:00-12:00 (shorter early window)."""
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0),
                        ("NIFTY",1,"10:00","12:00",0), ("SENSEX",1,"10:00","12:00",0)]
    return _tue_quad_base(d, wd)

# Tue 6-trade (triple windows × 2 indexes)
def v_tue_hex(d, wd, h, ad):
    if wd == 1: return [
        ("NIFTY",0,"09:30","12:00",0), ("SENSEX",0,"09:30","12:00",0),
        ("NIFTY",1,"10:30","13:30",0), ("SENSEX",1,"10:30","13:30",0),
        ("NIFTY",0,"12:30","14:30",0), ("SENSEX",0,"12:30","14:30",0),
    ]
    return _tue_quad_base(d, wd)

# Mon-loss-widen on top of Tue QUAD + Wed extra + Mon ATM
def v_final_adaptive(d, wd, h, ad):
    base = v_tq_wed_mon(d, wd, h, ad)
    mp = mon_pnl(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

VARIANTS = [
    ("3a baseline",                          v_3a),
    ("Tue QUAD (v4 risk-adj winner)",        v_tue_quad),
    ("TQ + Mon ATM",                         v_tq_mon_atm),
    ("TQ + Wed-extra NIFTY",                 v_tq_wed_extra),
    ("TQ + Wed-extra + Mon ATM",             v_tq_wed_mon),
    ("TQ + Wed-extra + Mon ATM + fwk-ATM",   v_tq_wed_mon_fwk),
    ("TQ extras at ATM (not +1)",            v_tq_extra_atm),
    ("TQ extras at +2",                      v_tq_extra_p2),
    ("TQ extras late 11:30-14:00",           v_tq_extra_late),
    ("TQ extras short 10:00-12:00",          v_tq_extra_short),
    ("Tue HEX (6 trades)",                   v_tue_hex),
    ("FINAL adaptive (TQ+Wed+Mon+widen)",    v_final_adaptive),
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
print(f"{'Variant':<42}{'Tr':>5}{'Win%':>6}{'Total':>14}{'Avg/d':>10}{'Worst':>11}{'Best':>11}{'vs 3a':>13}")
print("-"*112)
for name, r in ranked:
    diff = r["total"] - baseline
    flag = " ★" if diff > 0 else ""
    print(f"{name:<42}{r['trades']:>5}{r['win']:>5}%{fmt_rs(r['total']):>14}{fmt_rs(r['avg']):>10}"
          f"{fmt_rs(r['worst']):>11}{fmt_rs(r['best']):>11}{fmt_rs(diff):>13}{flag}")

print()
print("══════ MONTHLY — top 5 ══════")
top = ranked[:5]
months = sorted({m for _, r in top for m in r["monthly"]})
print(f"{'Variant':<42}" + "".join(f"{m:>13}" for m in months) + f"{'TOTAL':>13}")
print("-"*(42 + 13*(len(months)+1)))
for name, r in top:
    line = f"{name:<42}" + "".join(f"{fmt_rs(r['monthly'].get(m,0)):>13}" for m in months)
    line += f"{fmt_rs(r['total']):>13}"
    print(line)

print()
print("══════ EVERY MONTH POSITIVE? (top 6) ══════")
for name, r in ranked[:6]:
    mn = min(r['monthly'].values()) if r['monthly'] else 0
    flag = "✓ ALL POSITIVE" if mn > 0 else "✗ has loss month"
    print(f"  {name:<40} min-mo={fmt_rs(mn):>13}   {flag}")
