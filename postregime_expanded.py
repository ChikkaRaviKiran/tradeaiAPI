#!/usr/bin/env python3
"""EXPANDED post-regime SL-only search with extra selling strategies:
  Originals: straddle, strangle+1/+2, ironfly W2/W3/W4, condor S1W3/S2W4
  NEW:       strangle+3/+4/+5, ironfly W5/W6,
             condor S1W4/S2W5/S2W6/S3W5/S3W6,
             jade_lizard (short PE + bear call spread),
             jade_lizard_inv (short CE + bull put spread)
SL-only, no TP. Apr>0 AND May>0 constraint. 3 lots."""
from collections import defaultdict
from datetime import date
from phase3a_breakdown_v2 import to_min, load_chain, LOTS, STRIKE_STEP
from strategy_definedrisk_sweep import (find_atm, get_price, find_days,
                                         ENTRIES, EXITS, SL_PCTS,
                                         legs_straddle, legs_strangle,
                                         legs_ironfly, legs_condor)

LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}
MAX_L = 6000.0
REGIME_START = date(2026, 4, 14)
APR_END = date(2026, 4, 30)

# ─── NEW strategy leg builders ───
def legs_jade_lizard(atm, step):
    """Short ATM put + bear call spread (short ATM+1 CE, long ATM+3 CE).
    No upside risk if total credit > width of call spread."""
    return [(atm, "PE", -1),
            (atm + 1*step, "CE", -1),
            (atm + 3*step, "CE", +1)]

def legs_jade_lizard_inv(atm, step):
    """Short ATM call + bull put spread (short ATM-1 PE, long ATM-3 PE).
    No downside risk if total credit > width of put spread."""
    return [(atm, "CE", -1),
            (atm - 1*step, "PE", -1),
            (atm - 3*step, "PE", +1)]

def legs_broken_wing_call(atm, step):
    """Broken-wing call butterfly: short 1 ATM CE, long 2 (ATM+2) CE,
    short 1 (ATM+3) CE. Net credit, undefined risk above ATM+3 but small."""
    return [(atm, "CE", -1),
            (atm + 2*step, "CE", +2),
            (atm + 3*step, "CE", -1)]

def build_strategies_expanded(sym):
    step = STRIKE_STEP[sym]
    base = {
        "straddle":     lambda atm: legs_straddle(atm, step),
        "strangle+1":   lambda atm: legs_strangle(atm, step, 1),
        "strangle+2":   lambda atm: legs_strangle(atm, step, 2),
        "strangle+3":   lambda atm: legs_strangle(atm, step, 3),
        "strangle+4":   lambda atm: legs_strangle(atm, step, 4),
        "strangle+5":   lambda atm: legs_strangle(atm, step, 5),
        "ironfly_W2":   lambda atm: legs_ironfly(atm, step, 2),
        "ironfly_W3":   lambda atm: legs_ironfly(atm, step, 3),
        "ironfly_W4":   lambda atm: legs_ironfly(atm, step, 4),
        "ironfly_W5":   lambda atm: legs_ironfly(atm, step, 5),
        "ironfly_W6":   lambda atm: legs_ironfly(atm, step, 6),
        "condor_S1W3":  lambda atm: legs_condor(atm, step, 1, 3),
        "condor_S1W4":  lambda atm: legs_condor(atm, step, 1, 4),
        "condor_S2W4":  lambda atm: legs_condor(atm, step, 2, 4),
        "condor_S2W5":  lambda atm: legs_condor(atm, step, 2, 5),
        "condor_S2W6":  lambda atm: legs_condor(atm, step, 2, 6),
        "condor_S3W5":  lambda atm: legs_condor(atm, step, 3, 5),
        "condor_S3W6":  lambda atm: legs_condor(atm, step, 3, 6),
        "jade_lizard":      lambda atm: legs_jade_lizard(atm, step),
        "jade_lizard_inv":  lambda atm: legs_jade_lizard_inv(atm, step),
        "bwfly_call":       lambda atm: legs_broken_wing_call(atm, step),
    }
    return base

def sim(bars, legs, em, xm, sl_pct, lot):
    ent = []
    for k, side, qty in legs:
        e = get_price(bars, k, side, em)
        if e is None: return None
        ent.append((k, side, qty, e))
    credit = sum((e if q == -1 else -e) * abs(q) for _,_,q,e in
                 [(k,s,q,e) for k,s,q,e in ent])
    # rewrite: credit = sum over legs of -qty*e (short → +e, long → −e), scaled by |qty|
    credit = sum((-q) * e for _,_,q,e in ent)
    if credit <= 0: return None
    sl_pts = credit * sl_pct / 100.0
    N = lot * LOTS
    last = None
    for m in range(em + 1, xm + 1):
        ok = True; mtm = 0.0
        for k, side, qty, e in ent:
            p = get_price(bars, k, side, m)
            if p is None: ok = False; break
            mtm += qty * (e - p) if qty == -1 else qty * (p - e)
            # FIX: for short (qty=-1) pnl = e - p; for long (qty=+1) pnl = p - e
            # but if qty=+2 (broken wing) we need 2*(p-e). Use abs(qty)*sign:
        if not ok: continue
        # recompute mtm correctly:
        mtm = 0.0
        for k, side, qty, e in ent:
            p = get_price(bars, k, side, m)
            if qty < 0:   mtm += abs(qty) * (e - p)   # short
            else:         mtm += abs(qty) * (p - e)   # long
        rs = mtm * N
        if rs <= -MAX_L:           return rs, "SL_RS"
        if mtm <= -sl_pts:         return mtm*N, "SL"
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

    by_bucket = defaultdict(list)
    for (d, sym), bars in cache.items():
        by_bucket[(d.strftime("%a"), sym)].append((d, bars))

    print("\nSearching expanded strategy set (21 strategies × 4 entries × 5 exits × 4 SLs)...")
    candidates = []
    n_evaluated = 0
    for (wd, sym), days in by_bucket.items():
        strat_map = build_strategies_expanded(sym)
        lot = LOT_SIZE[sym]
        for sname, leg_fn in strat_map.items():
            for e in ENTRIES:
                em = to_min(e)
                for x in EXITS:
                    xm = to_min(x)
                    if xm <= em: continue
                    for sl in SL_PCTS:
                        n_evaluated += 1
                        trades = []
                        for d, bars in days:
                            atm = find_atm(bars, em, sym)
                            if atm is None: continue
                            legs = leg_fn(atm)
                            r = sim(bars, legs, em, xm, sl, lot)
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
                            "en": e, "ex": x, "sl": sl,
                            "n": len(trades), "wins": wins,
                            "apr": apr_pnl, "may": may_pnl, "tot": tot,
                            "worst": worst, "trades": trades,
                        })

    print(f"  evaluated: {n_evaluated} combos   passing constraint: {len(candidates)}")

    # Top 5 candidates per (weekday × index) bucket
    by_bucket_cands = defaultdict(list)
    for c in candidates:
        by_bucket_cands[(c["wd"], c["sym"])].append(c)
    for k in by_bucket_cands:
        by_bucket_cands[k].sort(key=lambda c: -c["tot"])

    # Best per bucket
    best_by_bucket = {k: v[0] for k, v in by_bucket_cands.items()}

    # ── Show TOP 3 per (weekday × index) including NEW strategies ──
    print("\n══════════ TOP-3 per (weekday × index) ══════════")
    print(f"  {'WD':<4}{'Sym':<7}{'Rk':<3}{'Strategy':<17}{'En':<7}{'→Ex':<7}{'SL%':>4}"
          f"{'Total':>11}{'Worst':>10}  W/N")
    order_bk = sorted(by_bucket_cands.keys(), key=lambda k: -by_bucket_cands[k][0]["tot"])
    for k in order_bk:
        for rank, c in enumerate(by_bucket_cands[k][:3], 1):
            tag = " *NEW" if c["strat"] in ("strangle+3","strangle+4","strangle+5",
                                              "ironfly_W5","ironfly_W6",
                                              "condor_S1W4","condor_S2W5","condor_S2W6",
                                              "condor_S3W5","condor_S3W6",
                                              "jade_lizard","jade_lizard_inv","bwfly_call") else ""
            wn = f"{c['wins']}/{c['n']}"
            print(f"  {k[0]:<4}{k[1]:<7}{rank:<3}{c['strat']:<17}{c['en']:<6}{c['ex']:<6}{c['sl']:>4}"
                  f"  ₹{c['tot']:>+8,.0f} ₹{c['worst']:>+7,.0f}  {wn}{tag}")

    # Phase-3a winner per weekday
    print("\n══════════ PHASE-3a EXPANDED — one index/day ══════════")
    best_by_wd = {}
    for (wd, sym), c in best_by_bucket.items():
        if wd not in best_by_wd or c["tot"] > best_by_wd[wd]["tot"]:
            best_by_wd[wd] = c
    tot = ap = my = 0; tr = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        c = best_by_wd.get(wd)
        if not c:
            print(f"  {wd}: NONE"); continue
        print(f"  {wd}: {c['sym']:<7}{c['strat']:<17}{c['en']}→{c['ex']}  SL={c['sl']}%   "
              f"Apr=₹{c['apr']:+,.0f} May=₹{c['may']:+,.0f} Tot=₹{c['tot']:+,.0f}  "
              f"worst=₹{c['worst']:+,.0f}  W/N={c['wins']}/{c['n']}")
        tot += c["tot"]; ap += c["apr"]; my += c["may"]; tr += c["n"]
    print(f"\n  PHASE-3a EXPANDED TOTAL: ₹{tot:+,.0f}   Apr=₹{ap:+,.0f}  May=₹{my:+,.0f}   trades={tr}")

    # Trade-by-trade
    print("\n══════════ Trade-by-trade ══════════")
    print(f"  {'#':<3}{'Date':<11}{'WD':<4}{'Sym':<7}{'Strategy':<17}{'En':<6}{'→Ex':<6}{'Why':<8}{'PnL':>10}{'Cum':>11}")
    flat = []
    for wd, c in best_by_wd.items():
        for d, pnl, why in c["trades"]:
            flat.append((d, wd, c["sym"], c["strat"], c["en"], c["ex"], why, pnl))
    cum = 0
    for i, (d, wd, sym, strat, en, ex, why, pnl) in enumerate(sorted(flat), 1):
        cum += pnl
        print(f"  {i:<3}{d.isoformat()}  {wd:<3}{sym:<7}{strat:<17}{en:<6}{ex:<5} {why:<7}₹{pnl:>+8,.0f} ₹{cum:>+9,.0f}")

if __name__ == "__main__":
    main()
