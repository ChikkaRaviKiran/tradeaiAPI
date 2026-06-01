"""Detailed month-by-month breakdown of 3a strategy."""
from collections import defaultdict
from datetime import datetime
from strategy_bakeoff import day_pnl, fmt_rs, B, psql

START, END = "2026-01-01", "2026-05-22"
rows = psql(f"SELECT DISTINCT date FROM option_candles WHERE date BETWEEN '{START}' AND '{END}' ORDER BY date;")
days = sorted({datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows})

WD = ["Mon","Tue","Wed","Thu","Fri"]
by_month = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "days": [], "wd_pnl": defaultdict(float), "wd_n": defaultdict(int)})

for d in days:
    wd = d.weekday()
    if wd > 4: continue
    s = B.get(wd)
    if not s: continue
    p = day_pnl(d, s)
    if p is None: continue
    m = d.strftime("%Y-%m")
    by_month[m]["trades"] += 1
    if p > 0: by_month[m]["wins"] += 1
    by_month[m]["pnl"] += p
    by_month[m]["days"].append((d, wd, p))
    by_month[m]["wd_pnl"][wd] += p
    by_month[m]["wd_n"][wd] += 1

print(f"{'Month':<10}{'Days':>6}{'Wins':>6}{'Win%':>7}{'PnL':>14}{'Avg/tr':>10}{'Best day':>32}{'Worst day':>32}")
print("-"*117)
grand = {"trades":0, "wins":0, "pnl":0}
for m in sorted(by_month):
    r = by_month[m]
    win = round(100*r["wins"]/r["trades"]) if r["trades"] else 0
    avg = r["pnl"]/r["trades"] if r["trades"] else 0
    best = max(r["days"], key=lambda x: x[2])
    worst = min(r["days"], key=lambda x: x[2])
    bd = f"{best[0]} {WD[best[1]]} {fmt_rs(best[2])}"
    wd = f"{worst[0]} {WD[worst[1]]} {fmt_rs(worst[2])}"
    print(f"{m:<10}{r['trades']:>6}{r['wins']:>6}{win:>6}%{fmt_rs(r['pnl']):>14}{fmt_rs(avg):>10}{bd:>32}{wd:>32}")
    grand["trades"] += r["trades"]; grand["wins"] += r["wins"]; grand["pnl"] += r["pnl"]

print("-"*117)
gw = round(100*grand["wins"]/grand["trades"])
ga = grand["pnl"]/grand["trades"]
print(f"{'TOTAL':<10}{grand['trades']:>6}{grand['wins']:>6}{gw:>6}%{fmt_rs(grand['pnl']):>14}{fmt_rs(ga):>10}")

print()
print("══════ By weekday across all 5 months ══════")
print(f"{'WD':<6}{'Trades':>8}{'PnL':>14}{'Avg/tr':>10}{'% of total':>12}")
wd_tot = defaultdict(lambda: {"n":0,"pnl":0})
for m, r in by_month.items():
    for w, p in r["wd_pnl"].items():
        wd_tot[w]["pnl"] += p
        wd_tot[w]["n"] += r["wd_n"][w]
for w in range(5):
    r = wd_tot[w]
    avg = r["pnl"]/r["n"] if r["n"] else 0
    pct = 100*r["pnl"]/grand["pnl"]
    print(f"{WD[w]:<6}{r['n']:>8}{fmt_rs(r['pnl']):>14}{fmt_rs(avg):>10}{pct:>11.1f}%")

print()
print("══════ Calendar days in each month ══════")
cal = defaultdict(lambda: {"all":0, "trade":0})
for d in days:
    cal[d.strftime("%Y-%m")]["all"] += 1
    if d.weekday() < 5: cal[d.strftime("%Y-%m")]["trade"] += 1
for m in sorted(cal):
    c = cal[m]
    print(f"  {m}: {c['all']:>2} calendar days  /  {c['trade']:>2} weekday(s)  (sample range)")

print()
print(f"NOTE: May data ends 2026-05-22 (only ~16 calendar days), causing the 5-month avg")
print(f"of ₹96,568 to under-represent typical monthly PnL.")
print(f"Average of complete months Jan-Apr = ₹{(grand['pnl']-by_month['2026-05']['pnl'])/4:,.0f} / month")
