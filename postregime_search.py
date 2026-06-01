#!/usr/bin/env python3
"""Search BEST post-regime strategy. Constraint: both Apr 2026 AND May 2026
must be net positive. NIFTY lot=65, SENSEX lot=20, 3 lots. Gates ±₹6000."""
from collections import defaultdict
from datetime import date
from phase3a_breakdown_v2 import to_min, load_chain, LOTS
from strategy_definedrisk_sweep import (find_atm, get_price, build_strategies,
                                         find_days, ENTRIES, EXITS, SL_PCTS,
                                         TP_PCTS)

LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}
MAX_P = 6000.0
MAX_L = 6000.0
REGIME_START = date(2026, 4, 14)
APR_END = date(2026, 4, 30)

def sim(bars, legs, em, xm, sl_pct, tp_pct, lot):
    ent = []
    for k, side, qty in legs:
        e = get_price(bars, k, side, em)
        if e is None: return None
        ent.append((k, side, qty, e))
    credit = sum((e if q == -1 else -e) for _,_,q,e in ent)
    if credit <= 0: return None
    sl_pts = credit * sl_pct / 100.0
    tp_pts = credit * tp_pct / 100.0 if tp_pct else None
    N = lot * LOTS
    last = None
    for m in range(em + 1, xm + 1):
        ok = True; mtm = 0.0
        for k, side, qty, e in ent:
            p = get_price(bars, k, side, m)
            if p is None: ok = False; break
            mtm += (e - p) if qty == -1 else (p - e)
        if not ok: continue
        rs = mtm * N
        if rs >=  MAX_P:    return rs, "TP_RS"
        if rs <= -MAX_L:    return rs, "SL_RS"
        if tp_pts is not None and mtm >= tp_pts: return mtm*N, "TP"
        if mtm <= -sl_pts:                        return mtm*N, "SL"
        last = mtm
    if last is None: return None
    return last * N, "TIME"

def main():
    print("Loading post-regime chains (Apr 14 → May 26)...")
    near = find_days()
    cache = {}
    for (d, sym), exps in near.items():
        if d < REGIME_START: continue
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = bars
    print(f"  cache: {len(cache)} (date,sym)")

    by_bucket = defaultdict(list)   # (wd, sym) -> [(date, bars)]
    for (d, sym), bars in cache.items():
        by_bucket[(d.strftime("%a"), sym)].append((d, bars))

    # ─── per-(wd,sym) candidate search ───
    print("\nSearching per-(weekday,index) candidates...")
    candidates = []   # list of dicts
    for (wd, sym), days in by_bucket.items():
        strat_map = build_strategies(sym)
        lot = LOT_SIZE[sym]
        for sname, leg_fn in strat_map.items():
            for e in ENTRIES:
                em = to_min(e)
                for x in EXITS:
                    xm = to_min(x)
                    if xm <= em: continue
                    for sl in SL_PCTS:
                        for tp in TP_PCTS:
                            trades = []
                            for d, bars in days:
                                atm = find_atm(bars, em, sym)
                                if atm is None: continue
                                legs = leg_fn(atm)
                                r = sim(bars, legs, em, xm, sl, tp, lot)
                                if r is None: continue
                                trades.append((d, r[0], r[1]))
                            if len(trades) < max(3, int(0.7*len(days))): continue
                            apr_pnl = sum(p for d,p,_ in trades if d <= APR_END)
                            may_pnl = sum(p for d,p,_ in trades if d  > APR_END)
                            apr_n   = sum(1 for d,_,_ in trades if d <= APR_END)
                            may_n   = sum(1 for d,_,_ in trades if d  > APR_END)
                            if apr_n < 1 or may_n < 2: continue
                            if apr_pnl <= 0 or may_pnl <= 0: continue
                            tot = apr_pnl + may_pnl
                            wins = sum(1 for _,p,_ in trades if p > 0)
                            worst = min(p for _,p,_ in trades)
                            candidates.append({
                                "wd": wd, "sym": sym, "strat": sname,
                                "en": e, "ex": x, "sl": sl, "tp": tp or 0,
                                "n": len(trades), "wins": wins,
                                "apr_n": apr_n, "may_n": may_n,
                                "apr": apr_pnl, "may": may_pnl, "tot": tot,
                                "worst": worst, "trades": trades,
                            })

    # ─── best per (wd, sym) ───
    print(f"  candidates passing Apr>0 & May>0: {len(candidates)}")
    best_by_bucket = {}
    for c in candidates:
        key = (c["wd"], c["sym"])
        if key not in best_by_bucket or c["tot"] > best_by_bucket[key]["tot"]:
            best_by_bucket[key] = c

    # Sort buckets by ₹ contribution
    order = sorted(best_by_bucket.items(), key=lambda kv: -kv[1]["tot"])

    print("\n══════════ BEST per (weekday × index) — Apr>0 AND May>0 ══════════")
    print(f"  {'WD':<4}{'Sym':<7}{'Strategy':<14}{'En':<7}{'→Ex':<7}{'SL%':>4}{'TP%':>4}"
          f"{'Apr':>9}{'May':>9}{'Total':>10}{'Worst':>9}  W/N")
    for (wd, sym), c in order:
        wn = f"{c['wins']}/{c['n']}"
        print(f"  {wd:<4}{sym:<7}{c['strat']:<14}{c['en']:<6}{c['ex']:<6}"
              f"{c['sl']:>4}{c['tp']:>4}"
              f"  ₹{c['apr']:>+6,.0f} ₹{c['may']:>+6,.0f} ₹{c['tot']:>+7,.0f} ₹{c['worst']:>+6,.0f}  {wn}")

    # ─── PHASE-3a: one index/day = pick higher-total per weekday ───
    print("\n══════════ PHASE-3a winner per weekday (one index/day) ══════════")
    best_by_wd = {}
    for (wd, sym), c in best_by_bucket.items():
        if wd not in best_by_wd or c["tot"] > best_by_wd[wd]["tot"]:
            best_by_wd[wd] = c
    tot3a = ap3a = my3a = 0; tr3a = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        c = best_by_wd.get(wd)
        if not c:
            print(f"  {wd}: NO STRATEGY satisfies Apr>0 & May>0"); continue
        print(f"  {wd}: {c['sym']:<7}{c['strat']:<14}{c['en']}→{c['ex']}  SL={c['sl']}% TP={c['tp']}%   "
              f"Apr=₹{c['apr']:+,.0f} May=₹{c['may']:+,.0f} Tot=₹{c['tot']:+,.0f}  worst=₹{c['worst']:+,.0f}")
        tot3a += c["tot"]; ap3a += c["apr"]; my3a += c["may"]; tr3a += c["n"]
    print(f"\n  PHASE-3a TOTAL: ₹{tot3a:+,.0f}   Apr=₹{ap3a:+,.0f}  May=₹{my3a:+,.0f}   trades={tr3a}")

    # ─── PHASE-3b: both indices/day where both pass ───
    print("\n══════════ PHASE-3b (both indices/day if both pass) ══════════")
    tot3b = ap3b = my3b = 0; tr3b = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            c = best_by_bucket.get((wd, sym))
            if not c: continue
            print(f"  {wd}/{sym}: {c['strat']:<14}{c['en']}→{c['ex']}  SL={c['sl']}% TP={c['tp']}%   "
                  f"Apr=₹{c['apr']:+,.0f} May=₹{c['may']:+,.0f} Tot=₹{c['tot']:+,.0f}")
            tot3b += c["tot"]; ap3b += c["apr"]; my3b += c["may"]; tr3b += c["n"]
    print(f"\n  PHASE-3b TOTAL: ₹{tot3b:+,.0f}   Apr=₹{ap3b:+,.0f}  May=₹{my3b:+,.0f}   trades={tr3b}")

    # ─── Phase-3a trade-by-trade ───
    print("\n══════════ PHASE-3a trade-by-trade ══════════")
    print(f"  {'Date':<11}{'WD':<4}{'Sym':<7}{'Strategy':<14}{'En':<6}{'→Ex':<6}{'Why':<8}{'PnL':>11}")
    flat = []
    for wd, c in best_by_wd.items():
        for d, pnl, why in c["trades"]:
            flat.append((d, wd, c["sym"], c["strat"], c["en"], c["ex"], why, pnl))
    for d, wd, sym, strat, en, ex, why, pnl in sorted(flat):
        print(f"  {d.isoformat()}  {wd:<3}{sym:<7}{strat:<14}{en:<6}{ex:<5} {why:<7}₹{pnl:>+9,.0f}")

if __name__ == "__main__":
    main()
