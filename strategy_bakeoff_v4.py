"""strategy_bakeoff_v4.py — push max profit by combining v3 winners + multi-window."""
from collections import defaultdict
from datetime import datetime
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

def v_3a(d, wd, h, ad):
    s = B.get(wd); return [s] if s else []

def v_all_dbl(d, wd, h, ad):
    if wd == 0: return [("NIFTY",3,"09:20","11:00",0), ("SENSEX",3,"09:20","11:00",0)]
    if wd == 1: return [("NIFTY",0,"09:30","14:30",0), ("SENSEX",0,"09:30","14:30",0)]
    if wd == 2: return [("SENSEX",0,"09:45","15:00",0), ("NIFTY",0,"09:45","15:00",0)]
    if wd == 3: return [("SENSEX",0,"10:00","15:15",0), ("NIFTY",0,"10:00","15:15",0)]
    if wd == 4: return [("NIFTY",1,"10:00","13:00",0), ("SENSEX",1,"10:00","13:00",0)]
    return []

def v_v3_winner(d, wd, h, ad):
    """v3 leader: ALL-dbl + Tue extra NIFTY 10:30-13:30."""
    base = v_all_dbl(d, wd, h, ad)
    if wd == 1: base.append(("NIFTY",1,"10:30","13:30",0))
    return base

# ─── push 1: ALL-dbl + Tue extra NIFTY + Mon ATM ───
def v_v3win_mon_atm(d, wd, h, ad):
    if wd == 0: return [("NIFTY",0,"09:20","11:00",0), ("SENSEX",0,"09:20","11:00",0)]
    return v_v3_winner(d, wd, h, ad)

# ─── push 2: also Mon ATM longer window 09:20-13:00 ───
def v_v3win_mon_atm_long(d, wd, h, ad):
    if wd == 0: return [("NIFTY",0,"09:20","13:00",0), ("SENSEX",0,"09:20","13:00",0)]
    return v_v3_winner(d, wd, h, ad)

# ─── push 3: + first-wk ATM (override Mon+3, Fri+1 → ATM in week-1) ───
def v_v3win_mon_atm_fwk(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    return base

# ─── push 4: + first-wk ATM + last-wk +1 ───
def v_v3win_full(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if is_first_week(d):
        base = [(s,0,e,x,r) for (s,o,e,x,r) in base]
    if is_last_week(d, ad):
        base = [(s,o+1,e,x,r) for (s,o,e,x,r) in base]
    return base

# ─── push 5: Tue QUAD (N+S + N late + S late) ───
def v_tue_quad(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if wd == 1:
        base.append(("SENSEX",1,"10:30","13:30",0))
    return base

# ─── push 6: Wed extra trade (Wed is SENSEX-expiry-1, NIFTY-expiry day-1?) ───
def v_wed_extra(d, wd, h, ad):
    """ALL-dbl + Tue extra NIFTY + Wed extra NIFTY late."""
    base = v_v3win_mon_atm(d, wd, h, ad)
    if wd == 2:
        base.append(("NIFTY",1,"10:30","13:30",0))
    return base

# ─── push 7: Thu extra NIFTY ───
def v_thu_extra(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if wd == 3:
        base.append(("NIFTY",1,"10:30","13:30",0))
    return base

# ─── push 8: All-extra (Tue+Wed+Thu extra NIFTY 10:30-13:30) ───
def v_all_extra(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if wd in (1, 2, 3):
        base.append(("NIFTY",1,"10:30","13:30",0))
    return base

# ─── push 9: full-day NIFTY+SENSEX every day (single window) ───
def v_full_day(d, wd, h, ad):
    if wd == 0: return [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    if wd == 1: return [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    if wd == 2: return [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    if wd == 3: return [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    if wd == 4: return [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    return []

# ─── push 10: full-day +1 OTM both indexes every day ───
def v_full_day_p1(d, wd, h, ad):
    if wd > 4: return []
    return [("NIFTY",1,"09:20","15:00",0), ("SENSEX",1,"09:20","15:00",0)]

# ─── push 11: bigger Mon — try Mon ATM 09:20-15:00 (full day on Mon) ───
def v_mon_full(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if wd == 0:
        base = [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    return base

# ─── push 12: replace Fri NIFTY+1 with ATM and add late NIFTY ───
def v_fri_strong(d, wd, h, ad):
    base = v_v3win_mon_atm(d, wd, h, ad)
    if wd == 4:
        base = [("NIFTY",0,"09:20","14:00",0), ("SENSEX",0,"09:20","14:00",0)]
    return base

# ─── push 13: MEGA — full day + extras on Tue ───
def v_mega(d, wd, h, ad):
    base = []
    if wd == 0: base = [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    elif wd == 1:
        base = [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0),
                ("NIFTY",1,"10:30","13:30",0)]
    elif wd == 2:
        base = [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    elif wd == 3:
        base = [("NIFTY",0,"09:20","15:00",0), ("SENSEX",0,"09:20","15:00",0)]
    elif wd == 4:
        base = [("NIFTY",0,"09:20","14:00",0), ("SENSEX",0,"09:20","14:00",0)]
    return base

# ─── push 14: MEGA + first-wk ATM (already ATM) + Mon-loss-widen ───
def v_mega_adaptive(d, wd, h, ad):
    base = v_mega(d, wd, h, ad)
    mp = mon_pnl(d, h)
    if wd >= 1 and mp is not None and mp < 0:
        base = [(s, max(o,3), e, x, r) for (s,o,e,x,r) in base]
    return base

VARIANTS = [
    ("3a baseline",                          v_3a),
    ("v3 winner: ALL-dbl+Tue+N",             v_v3_winner),
    ("+ Mon ATM",                            v_v3win_mon_atm),
    ("+ Mon ATM long (09:20-13:00)",         v_v3win_mon_atm_long),
    ("+ Mon ATM + first-wk ATM",             v_v3win_mon_atm_fwk),
    ("+ Mon ATM + fwk-ATM + lwk +1",         v_v3win_full),
    ("Tue QUAD (N+S+N+S late)",              v_tue_quad),
    ("+ Wed extra NIFTY late",               v_wed_extra),
    ("+ Thu extra NIFTY late",               v_thu_extra),
    ("+ Tue+Wed+Thu extra NIFTY",            v_all_extra),
    ("FULL-DAY both indexes ATM (09:20-15:00)", v_full_day),
    ("FULL-DAY both indexes +1 OTM",         v_full_day_p1),
    ("v3win + Mon FULL-DAY",                 v_mon_full),
    ("v3win + Fri FULL-DAY ATM",             v_fri_strong),
    ("MEGA (full-day + Tue extra N)",        v_mega),
    ("MEGA + Mon-loss-widen",                v_mega_adaptive),
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
print("══════ MONTHLY — top 6 ══════")
top6 = ranked[:6]
months = sorted({m for _, r in top6 for m in r["monthly"]})
print(f"{'Variant':<42}" + "".join(f"{m:>13}" for m in months) + f"{'TOTAL':>13}")
print("-"*(42 + 13*(len(months)+1)))
for name, r in top6:
    line = f"{name:<42}" + "".join(f"{fmt_rs(r['monthly'].get(m,0)):>13}" for m in months)
    line += f"{fmt_rs(r['total']):>13}"
    print(line)

print()
print("══════ EVERY MONTH POSITIVE? (top 8) ══════")
for name, r in ranked[:8]:
    mn = min(r['monthly'].values()) if r['monthly'] else 0
    flag = "✓ ALL POSITIVE" if mn > 0 else "✗ has loss month"
    print(f"  {name:<40} min-mo={fmt_rs(mn):>13}   {flag}")
