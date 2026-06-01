#!/usr/bin/env python3
"""POST-REGIME ONLY strategy search (Apr 15 – May 26, 2026).

Same engine as strategy_search_6mo.py but only the post-regime window,
relaxed min-sample (small n), and no regime split.
"""
from collections import defaultdict
from datetime import datetime, date
import json
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, LOT_SIZE, LOTS, STRIKE_STEP)
from strategy_search_6mo import (find_atm, pnl_straddle, pnl_ironfly,
                                  pnl_ironcondor, pnl_naked, pnl_ratio,
                                  build_catalog)

START = "2026-04-15"
END   = "2026-05-26"

ENTRIES = ["09:20", "09:30", "09:45", "10:00", "10:30", "11:00"]
EXITS   = ["10:30", "11:00", "11:30", "12:00", "12:30",
           "13:00", "13:30", "14:00", "14:30", "15:00", "15:15"]
MIN_SAMPLE = 4
COVERAGE   = 0.65

def find_days():
    rows = psql(f"""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '{START}' AND date <= '{END}' ORDER BY date, instrument;""")
    by_day = defaultdict(list)
    for date_s, inst, exp_s in rows:
        try: d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except Exception: continue
        ex = parse_expiry(exp_s)
        if ex is None or ex < d: continue
        by_day[(d, inst)].append((ex, exp_s))
    return {k: sorted(v) for k,v in by_day.items()}

def main():
    near = find_days()
    print(f"(date,sym) combos: {len(near)}   window {START} → {END}", flush=True)

    print("Loading chains (nearest weekly)...", flush=True)
    cache = {}
    items = list(near.items())
    for i, ((d, sym), exps) in enumerate(items, 1):
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = (bars, (ex-d).days, exp_s)
        if i % 20 == 0:
            print(f"  [{i}/{len(items)}] cached={len(cache)}", flush=True)
    print(f"Cached chains: {len(cache)}\n", flush=True)

    by_bucket = defaultdict(list)
    for (d, sym), (bars, dte, _) in cache.items():
        by_bucket[(d.strftime("%a"), sym)].append((d, bars, dte))

    print("Days per (weekday, index):")
    for k in sorted(by_bucket): print(f"  {k}: {len(by_bucket[k])}")
    print()

    results = []
    for (wd, sym), days in by_bucket.items():
        if len(days) < MIN_SAMPLE: continue
        catalog = build_catalog(sym)
        atm_cache = {}
        for d, bars, _ in days:
            for e in ENTRIES:
                atm_cache[(d, e)] = find_atm(bars, to_min(e), sym)
        for strat_name, fn in catalog:
            for e in ENTRIES:
                em = to_min(e)
                for x in EXITS:
                    xm = to_min(x)
                    if xm <= em: continue
                    pnls = []
                    for d, bars, _ in days:
                        atm = atm_cache.get((d, e))
                        if atm is None: continue
                        p = fn(bars, atm, em, xm)
                        if p is None: continue
                        pnls.append(p)
                    n = len(pnls)
                    if n < len(days) * COVERAGE: continue
                    cum = sum(pnls); w = sum(1 for p in pnls if p>0)
                    results.append({
                        "wd": wd, "sym": sym, "strat": strat_name,
                        "entry": e, "exit": x,
                        "n": n, "win": w, "cum": cum,
                        "worst": min(pnls), "best_day": max(pnls),
                        "avg": cum/n,
                    })

    print(f"Total result rows: {len(results)}\n", flush=True)

    # Top 5 per (wd, sym)
    print("══════════ TOP-5 per (WEEKDAY, INDEX) — POST-REGIME ONLY ══════════")
    by_ws = defaultdict(list)
    for r in results: by_ws[(r["wd"], r["sym"])].append(r)
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            rs = by_ws.get((wd, sym), [])
            if not rs: continue
            rs.sort(key=lambda r: -r["cum"])
            print(f"\n  {wd} {sym} (n_days={len(by_bucket[(wd,sym)])}):")
            print(f"    {'Strategy':18} {'n':>3} {'Win':>4} {'En':5}→{'Ex':5}  {'Cum':>11}  {'Avg/d':>10}  {'Worst':>10}")
            for r in rs[:5]:
                print(f"    {r['strat']:18} {r['n']:>3} {int(100*r['win']/r['n']):>3}% {r['entry']}→{r['exit']}  {fmt_rs(r['cum']):>11}  {fmt_rs(r['avg']):>10}  {fmt_rs(r['worst']):>10}")

    # Phase-3a winner per day (any index, any strategy)
    print("\n══════════ PHASE-3a POST-REGIME (one index/day, ANY strategy) ══════════")
    by_wd = defaultdict(list)
    for r in results: by_wd[r["wd"]].append(r)
    chosen = {}
    grand = ntot = wtot = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        opts = by_wd.get(wd, [])
        if not opts: continue
        # prefer profitable trades; tie-break by avg-per-day to avoid sample-size bias
        opts.sort(key=lambda r: (-r["cum"], -r["avg"]))
        r = opts[0]
        chosen[wd] = r
        print(f"  {wd}: {r['sym']:6} {r['strat']:18} {r['entry']}→{r['exit']}  "
              f"n={r['n']:2d} win={int(100*r['win']/r['n']):3d}%  "
              f"cum={fmt_rs(r['cum']):>10}  avg={fmt_rs(r['avg']):>8}  worst={fmt_rs(r['worst'])}")
        grand += r["cum"]; ntot += r["n"]; wtot += r["win"]
    if ntot:
        # extrapolate to monthly (avg 21 trading days/mo, so ~4-5 weeks)
        weeks = ntot / 5
        per_week = grand / weeks if weeks else 0
        per_month = per_week * 4.33
        print(f"\n  PHASE-3a POST TOTAL: {fmt_rs(grand)} over {ntot} trades, {int(100*wtot/ntot)}% win")
        print(f"  ≈ {fmt_rs(per_week)}/week  ≈ {fmt_rs(per_month)}/month  (extrapolated)")

    # Monthly replay
    print("\n══════════ MONTHLY P&L (post-regime months only) ══════════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for (d, sym), (bars, _, _) in cache.items():
        wd = d.strftime("%a")
        if wd not in chosen: continue
        r = chosen[wd]
        if sym != r["sym"]: continue
        atm = find_atm(bars, to_min(r["entry"]), sym)
        if atm is None: continue
        catalog = dict(build_catalog(sym))
        p = catalog[r["strat"]](bars, atm, to_min(r["entry"]), to_min(r["exit"]))
        if p is None: continue
        m = d.strftime("%Y-%m")
        by_m[m]["pnl"] += p; by_m[m]["n"] += 1
        if p > 0: by_m[m]["w"] += 1
    print(f"  {'Month':10} {'Days':>5} {'Win%':>5}  {'PnL':>12}")
    gp=gn=gw=0
    for m in sorted(by_m):
        v=by_m[m]; gp+=v["pnl"]; gn+=v["n"]; gw+=v["w"]
        print(f"  {m:10} {v['n']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>12}")
    if gn:
        print(f"  {'TOTAL':10} {gn:>5} {int(100*gw/gn):>4}%  {fmt_rs(gp):>12}")

    # Per-day trade log of the chosen schedule
    print("\n══════════ DAILY TRADE LOG (chosen schedule) ══════════")
    print(f"  {'Date':12} {'WD':4} {'Sym':7} {'Strategy':18} {'Window':14}  {'PnL':>11}  Cum")
    rows = []
    for (d, sym), (bars, _, _) in cache.items():
        wd = d.strftime("%a")
        if wd not in chosen: continue
        r = chosen[wd]
        if sym != r["sym"]: continue
        atm = find_atm(bars, to_min(r["entry"]), sym)
        if atm is None: continue
        catalog = dict(build_catalog(sym))
        p = catalog[r["strat"]](bars, atm, to_min(r["entry"]), to_min(r["exit"]))
        if p is None: continue
        rows.append((d, wd, sym, r["strat"], f"{r['entry']}→{r['exit']}", p))
    rows.sort()
    cum = 0
    for d, wd, sym, strat, win, p in rows:
        cum += p
        print(f"  {str(d):12} {wd:4} {sym:7} {strat:18} {win:14}  {fmt_rs(p):>11}  {fmt_rs(cum)}")

    with open("/tmp/strategy_search_postregime.json","w") as f:
        json.dump({"window":[START,END],
                   "phase3a": {wd:{k:(v if not isinstance(v,float) else round(v,2)) for k,v in r.items()}
                               for wd,r in chosen.items()}}, f, indent=2, default=str)
    print("\nWrote /tmp/strategy_search_postregime.json")

if __name__ == "__main__":
    main()
