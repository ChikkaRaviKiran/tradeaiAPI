#!/usr/bin/env python3
"""Apply hard ₹ gates (+6000 TP / -6000 SL per trade) on top of the chosen
Phase-3a + Phase-3b defined-risk schedule. Print summary only."""
import json
from collections import defaultdict
from phase3a_breakdown_v2 import (to_min, load_chain, LOT_SIZE, LOTS)
from strategy_definedrisk_sweep import (find_atm, get_price, build_strategies,
                                         find_days, REGIME_SPLIT)

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
        # Rupee gates first
        if mtm_rs >= MAX_PROFIT:
            exit_m, exit_reason, exit_mtm_pts = m, "TP_RS", mtm; break
        if mtm_rs <= -MAX_LOSS:
            exit_m, exit_reason, exit_mtm_pts = m, "SL_RS", mtm; break
        # % gates
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
    return mtm * 0 + exit_mtm_pts * notional, exit_reason

def run(cache, chosen_map_per_wd, chosen_map_per_wdsym, label):
    by_month = defaultdict(lambda: {"pnl":0,"n":0,"w":0,"tp":0,"sl":0,"tp_rs":0,"sl_rs":0,"time":0,"worst":0})
    pre_pnl=post_pnl=pre_n=post_n=0
    total_pnl=total_n=total_w=0
    daily_pnl = defaultdict(float)
    for (d, sym) in sorted(cache.keys()):
        wd = d.strftime("%a")
        bars, dte, _ = cache[(d, sym)]
        spec = None
        if chosen_map_per_wdsym and (wd, sym) in chosen_map_per_wdsym:
            spec = chosen_map_per_wdsym[(wd, sym)]
        elif chosen_map_per_wd and wd in chosen_map_per_wd and chosen_map_per_wd[wd]["sym"] == sym:
            spec = chosen_map_per_wd[wd]
        if spec is None: continue
        atm = find_atm(bars, to_min(spec["entry"]), sym)
        if atm is None: continue
        legs = build_strategies(sym)[spec["strat"]](atm)
        r = simulate_gated(bars, legs, to_min(spec["entry"]), to_min(spec["exit"]),
                            spec["sl"], spec["tp"], LOT_SIZE[sym])
        if r is None: continue
        pnl, reason = r
        m = d.strftime("%Y-%m")
        v = by_month[m]
        v["pnl"] += pnl; v["n"] += 1
        if pnl > 0: v["w"] += 1
        v[reason.lower()] = v.get(reason.lower(),0) + 1
        if pnl < v["worst"]: v["worst"] = pnl
        total_pnl += pnl; total_n += 1
        if pnl > 0: total_w += 1
        if d >= REGIME_SPLIT: post_pnl += pnl; post_n += 1
        else:                 pre_pnl  += pnl; pre_n  += 1
        daily_pnl[d] += pnl

    print(f"\n══════════ {label}  (TP gate ₹+{int(MAX_PROFIT)} / SL gate ₹-{int(MAX_LOSS)} per trade) ══════════")
    print(f"  {'Month':<10}{'Days':>5}{'Win%':>6}{'PnL':>14}   TIME/TP%/SL%/TP₹/SL₹  Worst")
    for m in sorted(by_month):
        v = by_month[m]
        wp = 100*v['w']/v['n'] if v['n'] else 0
        print(f"  {m:<10}{v['n']:>5}{wp:>5.0f}%  ₹{v['pnl']:>+11,.0f}   "
              f"{v.get('time',0)}/{v.get('tp',0)}/{v.get('sl',0)}/{v.get('tp_rs',0)}/{v.get('sl_rs',0)}  ₹{v['worst']:>+8,.0f}")
    wp = 100*total_w/total_n if total_n else 0
    print(f"  {'TOTAL':<10}{total_n:>5}{wp:>5.0f}%  ₹{total_pnl:>+11,.0f}")
    print(f"  Pre-regime : ₹{pre_pnl:>+11,.0f}  ({pre_n} trades)")
    print(f"  Post-regime: ₹{post_pnl:>+11,.0f}  ({post_n} trades)")
    worst_day = min(daily_pnl.values()) if daily_pnl else 0
    best_day  = max(daily_pnl.values()) if daily_pnl else 0
    print(f"  Worst single day (combined): ₹{worst_day:+,.0f}")
    print(f"  Best  single day (combined): ₹{best_day:+,.0f}")
    return total_pnl, total_n

def main():
    with open("/tmp/strategy_definedrisk.json") as f:
        sched = json.load(f)
    print("Loading all 6-month chains for chosen schedule...")
    near = find_days()
    cache = {}
    for (d, sym), exps in near.items():
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

    run(cache, chosen_3a, None,       "PHASE-3a (one index/day)")
    run(cache, None,       chosen_3b, "PHASE-3b (both indices/day)")
    print(f"\nNote: TP₹/SL₹ columns count days where the ₹6000 gate triggered before the %-based SL/TP or scheduled time exit.")

if __name__ == "__main__":
    main()
