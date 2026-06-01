#!/usr/bin/env python3
"""Post-regime only (Apr 15 → May 26) with NIFTY lot=65, SENSEX lot=20,
₹+6000 TP / ₹-6000 SL hard gates per trade."""
import json
from collections import defaultdict
from phase3a_breakdown_v2 import to_min, load_chain, LOTS
from strategy_definedrisk_sweep import (find_atm, get_price, build_strategies,
                                         find_days, REGIME_SPLIT)

LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}   # corrected
MAX_PROFIT = 6000.0
MAX_LOSS   = 6000.0

def simulate_gated(bars, legs, e_min, x_min, sl_pct, tp_pct, lotsize):
    entries = []
    for k, side, qty in legs:
        e = get_price(bars, k, side, e_min)
        if e is None: return None
        entries.append((k, side, qty, e))
    credit_pts = sum((e if q == -1 else -e) for _,_,q,e in entries)
    if credit_pts <= 0: return None
    sl_amt_pts = credit_pts * sl_pct / 100.0
    tp_amt_pts = credit_pts * tp_pct / 100.0 if tp_pct else None
    notional = lotsize * LOTS
    exit_m, exit_reason = x_min, "TIME"
    exit_mtm_pts = None
    for m in range(e_min + 1, x_min + 1):
        ok = True; mtm = 0.0
        for k, side, qty, e in entries:
            p = get_price(bars, k, side, m)
            if p is None: ok = False; break
            mtm += (e - p) if qty == -1 else (p - e)
        if not ok: continue
        mtm_rs = mtm * notional
        if mtm_rs >= MAX_PROFIT:
            exit_m, exit_reason, exit_mtm_pts = m, "TP_RS", mtm; break
        if mtm_rs <= -MAX_LOSS:
            exit_m, exit_reason, exit_mtm_pts = m, "SL_RS", mtm; break
        if tp_amt_pts is not None and mtm >= tp_amt_pts:
            exit_m, exit_reason, exit_mtm_pts = m, "TP", mtm; break
        if mtm <= -sl_amt_pts:
            exit_m, exit_reason, exit_mtm_pts = m, "SL", mtm; break
    if exit_mtm_pts is None:
        mtm = 0.0
        for k, side, qty, e in entries:
            p = get_price(bars, k, side, x_min)
            if p is None: return None
            mtm += (e - p) if qty == -1 else (p - e)
        exit_mtm_pts = mtm
    return exit_mtm_pts * notional, exit_reason, mtm

def run(cache, chosen_wd, chosen_wdsym, label):
    by_month = defaultdict(lambda: {"pnl":0,"n":0,"w":0,"tp":0,"sl":0,"tp_rs":0,"sl_rs":0,"time":0,"worst":0,"best":0})
    daily = defaultdict(float)
    trades = []
    for (d, sym) in sorted(cache.keys()):
        if d < REGIME_SPLIT: continue
        wd = d.strftime("%a")
        bars, dte, _ = cache[(d, sym)]
        spec = None
        if chosen_wdsym and (wd, sym) in chosen_wdsym:
            spec = chosen_wdsym[(wd, sym)]
        elif chosen_wd and wd in chosen_wd and chosen_wd[wd]["sym"] == sym:
            spec = chosen_wd[wd]
        if spec is None: continue
        atm = find_atm(bars, to_min(spec["entry"]), sym)
        if atm is None: continue
        legs = build_strategies(sym)[spec["strat"]](atm)
        r = simulate_gated(bars, legs, to_min(spec["entry"]), to_min(spec["exit"]),
                            spec["sl"], spec["tp"], LOT_SIZE[sym])
        if r is None: continue
        pnl, reason, _ = r
        trades.append((d, wd, sym, spec["strat"], atm, spec["entry"], spec["exit"],
                       reason, pnl))
        m = d.strftime("%Y-%m")
        v = by_month[m]
        v["pnl"] += pnl; v["n"] += 1
        if pnl > 0: v["w"] += 1
        v[reason.lower()] = v.get(reason.lower(),0) + 1
        if pnl < v["worst"]: v["worst"] = pnl
        if pnl > v["best"]:  v["best"]  = pnl
        daily[d] += pnl

    print(f"\n══════════ {label} ══════════")
    print(f"  Window: 2026-04-15 → 2026-05-26   NIFTY lot=65  SENSEX lot=20  lots={LOTS}")
    print(f"  Gates : TP ₹+{int(MAX_PROFIT)}  SL ₹-{int(MAX_LOSS)}  per trade\n")
    print(f"  {'Month':<10}{'Tr':>4}{'Win%':>6}{'PnL':>14}   TIME/TP%/SL%/TP₹/SL₹   Worst       Best")
    tot_pnl=tot_n=tot_w=0
    for m in sorted(by_month):
        v = by_month[m]
        wp = 100*v['w']/v['n'] if v['n'] else 0
        print(f"  {m:<10}{v['n']:>4}{wp:>5.0f}%  ₹{v['pnl']:>+11,.0f}   "
              f"{v.get('time',0)}/{v.get('tp',0)}/{v.get('sl',0)}/{v.get('tp_rs',0)}/{v.get('sl_rs',0)}    "
              f"₹{v['worst']:>+7,.0f}  ₹{v['best']:>+7,.0f}")
        tot_pnl += v['pnl']; tot_n += v['n']; tot_w += v['w']
    wp = 100*tot_w/tot_n if tot_n else 0
    print(f"  {'TOTAL':<10}{tot_n:>4}{wp:>5.0f}%  ₹{tot_pnl:>+11,.0f}")
    print(f"\n  Worst single day (sum across both indices if 3b): ₹{min(daily.values()):+,.0f}")
    print(f"  Best  single day: ₹{max(daily.values()):+,.0f}")
    days = sorted(daily.keys())
    print(f"  Trading days: {len(days)}   Daily avg: ₹{tot_pnl/len(days):+,.0f}")

    # Per-trade log
    print(f"\n  ── Trade-by-trade ──")
    print(f"  {'Date':<11}{'WD':<4}{'Sym':<7}{'Strategy':<14}{'ATM':>7}  {'En':<6}{'→Ex':<7}{'Why':<7}{'PnL ₹':>12}")
    for d, wd, sym, strat, atm, en, ex, reason, pnl in trades:
        print(f"  {d.isoformat()}  {wd:<3}{sym:<7}{strat:<14}{atm:>7}  {en:<6}{ex:<6} {reason:<7}₹{pnl:>+10,.0f}")

def main():
    with open("/tmp/strategy_definedrisk.json") as f:
        sched = json.load(f)
    print("Loading post-regime chains...")
    near = find_days()
    cache = {}
    for (d, sym), exps in near.items():
        if d < REGIME_SPLIT: continue
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = (bars, (ex-d).days, exp_s)
    print(f"  cache: {len(cache)} (date,sym) combos")

    chosen_3a = {wd: {**s, "sl": int(s["sl"]),
                       "tp": (int(s["tp"]) if s.get("tp") not in (None,"","None") else None)}
                  for wd, s in sched["phase3a"].items()}
    chosen_3b = {}
    for key, s in sched["phase3b"].items():
        wd, sym = key.split("_")
        chosen_3b[(wd, sym)] = {**s, "sl": int(s["sl"]),
                                 "tp": (int(s["tp"]) if s.get("tp") not in (None,"","None") else None)}

    run(cache, chosen_3a, None, "PHASE-3a  POST-REGIME  one index/day")
    run(cache, None, chosen_3b, "PHASE-3b  POST-REGIME  both indices/day")

if __name__ == "__main__":
    main()
