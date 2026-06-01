"""strategy_bakeoff_v2.py — layered refinements on the Tue-DOUBLE v1 winner."""
from collections import defaultdict
from datetime import datetime, date
from strategy_bakeoff import (day_pnl, find_expiries, fmt_rs, B, WD, psql)

START = "2026-01-01"
END   = "2026-05-22"

rows = psql(f"SELECT DISTINCT date FROM option_candles WHERE date BETWEEN '{START}' AND '{END}' ORDER BY date;")
days = sorted({datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows})
print(f"Days: {len(days)}", flush=True)

def is_last_week(d, all_days):
    return not any(x.month == d.month and x > d for x in all_days)
def is_first_week(d):
    return d.day <= 7

def mon_pnl_this_week(d, history):
    iso = d.isocalendar()
    for past_d, p in history.items():
        if past_d.isocalendar()[:2] == iso[:2] and past_d.weekday() == 0:
            return p
    return None

def v_3a(d, wd, h, ad):
    s = B.get(wd); return [s] if s else []

def v_tue_double(d, wd, h, ad):
    if wd == 1:
        return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    s = B.get(wd); return [s] if s else []

def v_tue_double_mon_widen(d, wd, h, ad):
    base = v_tue_double(d, wd, h, ad)
    mp = mon_pnl_this_week(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

def v_tue_double_fwk_atm(d, wd, h, ad):
    base = v_tue_double(d, wd, h, ad)
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    return base

def v_tue_double_lwk_p2(d, wd, h, ad):
    base = v_tue_double(d, wd, h, ad)
    if is_last_week(d, ad):
        base = [(s,o+2,e,x,r) for (s,o,e,x,r) in base]
    return base

def v_tue_double_no_mon(d, wd, h, ad):
    if wd == 0: return []
    return v_tue_double(d, wd, h, ad)

def v_tue_triple(d, wd, h, ad):
    if wd == 1:
        return [("NIFTY",0,"09:30","14:30",0),
                ("SENSEX",0,"09:30","14:30",0),
                ("NIFTY",1,"10:30","13:30",0)]
    s = B.get(wd); return [s] if s else []

def v_tue_sensex_early(d, wd, h, ad):
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:20","12:30",0)]
    s = B.get(wd); return [s] if s else []
def v_tue_sensex_late(d, wd, h, ad):
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"11:00","15:00",0)]
    s = B.get(wd); return [s] if s else []
def v_tue_sensex_p1(d, wd, h, ad):
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",1,"09:30","14:30",0)]
    s = B.get(wd); return [s] if s else []
def v_tue_sensex_p2(d, wd, h, ad):
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",2,"09:30","14:30",0)]
    s = B.get(wd); return [s] if s else []
def v_tue_nifty_p1_sensex(d, wd, h, ad):
    if wd == 1: return [("NIFTY",1,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    s = B.get(wd); return [s] if s else []

def v_wed_double(d, wd, h, ad):
    base = v_3a(d, wd, h, ad)
    if wd == 2: base.append(("NIFTY",0,"09:45","15:00",0))
    return base
def v_thu_double(d, wd, h, ad):
    base = v_3a(d, wd, h, ad)
    if wd == 3: base.append(("NIFTY",0,"10:00","15:15",0))
    return base
def v_tue_wed_double(d, wd, h, ad):
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    s = B.get(wd); return [s] if s else []
def v_tue_wed_thu_double(d, wd, h, ad):
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0), ("NIFTY",0,"10:00","15:15",0)]
    s = B.get(wd); return [s] if s else []
def v_all_double(d, wd, h, ad):
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0), ("SENSEX",3,"09:20","11:00",0)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0), ("NIFTY",0,"10:00","15:15",0)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0), ("SENSEX",1,"10:00","13:00",0)]
    return []

def v_stacked_combo(d, wd, h, ad):
    if wd == 1:
        base = [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    else:
        s = B.get(wd); base = [s] if s else []
    if not base: return []
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    if is_last_week(d, ad):
        base = [(s,o+2,e,x,r) for (s,o,e,x,r) in base]
    mp = mon_pnl_this_week(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

def v_tue_wed_double_mon_widen(d, wd, h, ad):
    base = v_tue_wed_double(d, wd, h, ad)
    mp = mon_pnl_this_week(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

VARIANTS = [
    ("3a baseline",                       v_3a),
    ("Tue-DOUBLE (v1 winner)",            v_tue_double),
    ("Tue-double + Mon-loss-widen",       v_tue_double_mon_widen),
    ("Tue-double + first-wk ATM",         v_tue_double_fwk_atm),
    ("Tue-double + last-wk +2 wider",     v_tue_double_lwk_p2),
    ("Tue-double + no Mon",               v_tue_double_no_mon),
    ("Tue-double + Wed-double",           v_tue_wed_double),
    ("Tue+Wed+Thu DOUBLE",                v_tue_wed_thu_double),
    ("ALL-days DOUBLE",                   v_all_double),
    ("Tue TRIPLE (N+S+N late)",           v_tue_triple),
    ("Tue SENSEX 09:20-12:30",            v_tue_sensex_early),
    ("Tue SENSEX 11:00-15:00",            v_tue_sensex_late),
    ("Tue SENSEX +1 OTM",                 v_tue_sensex_p1),
    ("Tue SENSEX +2 OTM",                 v_tue_sensex_p2),
    ("Tue NIFTY+1 + SENSEX ATM",          v_tue_nifty_p1_sensex),
    ("Wed-double only",                   v_wed_double),
    ("Thu-double only",                   v_thu_double),
    ("Stacked combo (all)",               v_stacked_combo),
    ("Tue+Wed double + Mon-loss-widen",   v_tue_wed_double_mon_widen),
]

results = {}
for name, fn in VARIANTS:
    print(f"  {name}...", flush=True)
    day_total = {}
    history = {}
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
            day_total[d] = tot
            history[d] = tot
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
