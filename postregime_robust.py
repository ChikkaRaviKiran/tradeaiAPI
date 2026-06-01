#!/usr/bin/env python3
"""Robustness / walk-forward validation for post-regime selling schedule.

Data source: option_candles table → populated by Dhan /v2/charts/rollingoption.

Splits Apr 14 → May 26 into TRAIN (Apr 14 → May 5) and TEST (May 6 → May 26).
For each (weekday × index × strategy × entry × exit × SL%) candidate, demands:
  - Profitable in TRAIN
  - Profitable in TEST
  - >= 2 trades in each half
  - Worst trade > -₹7000 (no near-blowups)
Then ranks survivors by combined total + min(half)*0.5 (stability bonus)."""
from collections import defaultdict
from datetime import date
from statistics import mean, median, pstdev
from phase3a_breakdown_v2 import to_min, load_chain, LOTS, STRIKE_STEP
from strategy_definedrisk_sweep import (find_atm, get_price, find_days,
                                         ENTRIES, EXITS, SL_PCTS,
                                         legs_straddle, legs_strangle,
                                         legs_ironfly, legs_condor)
from postregime_expanded import (legs_jade_lizard, legs_jade_lizard_inv,
                                  legs_broken_wing_call,
                                  build_strategies_expanded, sim)

LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}
MAX_L = 6000.0
REGIME_START = date(2026, 4, 14)
SPLIT_DATE   = date(2026, 5, 5)   # train: ≤ this, test: > this
APR_END      = date(2026, 4, 30)
END_DATE     = date(2026, 5, 26)

NEW_STRATS = {"strangle+3","strangle+4","strangle+5",
              "ironfly_W5","ironfly_W6",
              "condor_S1W4","condor_S2W5","condor_S2W6","condor_S3W5","condor_S3W6",
              "jade_lizard","jade_lizard_inv","bwfly_call"}

def stats(trades):
    """trades = list of (date, pnl_rs)"""
    pnls = [p for _,p in trades]
    if not pnls: return None
    return {
        "n": len(pnls),
        "tot": sum(pnls),
        "avg": mean(pnls),
        "med": median(pnls),
        "std": pstdev(pnls) if len(pnls) > 1 else 0,
        "min": min(pnls),
        "max": max(pnls),
        "wins": sum(1 for p in pnls if p > 0),
        "sharpe_like": (mean(pnls)/pstdev(pnls)) if len(pnls) > 1 and pstdev(pnls) > 0 else 0,
    }

def main():
    print("Data source: option_candles ← DhanHQ /v2/charts/rollingoption (1-min OHLC)")
    print("Loading post-regime chains (Apr 14 → May 26)...")
    near = find_days()
    cache = {}
    for (d, sym), exps in near.items():
        if d < REGIME_START or d > END_DATE: continue
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = bars
    print(f"  cache: {len(cache)} (date,sym) days loaded\n")

    train_days = [d for (d,_) in cache if d <= SPLIT_DATE]
    test_days  = [d for (d,_) in cache if d  > SPLIT_DATE]
    print(f"  TRAIN: {min(train_days).isoformat()} → {max(train_days).isoformat()} "
          f"({len(set(train_days))} unique dates)")
    print(f"  TEST:  {min(test_days).isoformat()} → {max(test_days).isoformat()} "
          f"({len(set(test_days))} unique dates)\n")

    by_bucket = defaultdict(list)
    for (d, sym), bars in cache.items():
        by_bucket[(d.strftime("%a"), sym)].append((d, bars))

    print("Walk-forward search (21 strategies × 4 entries × 5 exits × 4 SLs × 10 buckets)...")
    survivors = []   # passed both halves
    rejected_reasons = defaultdict(int)

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
                        all_trades = []
                        for d, bars in days:
                            atm = find_atm(bars, em, sym)
                            if atm is None: continue
                            legs = leg_fn(atm)
                            r = sim(bars, legs, em, xm, sl, lot)
                            if r is None: continue
                            all_trades.append((d, r[0]))
                        if len(all_trades) < 4:
                            rejected_reasons["too_few_trades"] += 1; continue
                        tr_t = [(d,p) for d,p in all_trades if d <= SPLIT_DATE]
                        te_t = [(d,p) for d,p in all_trades if d  > SPLIT_DATE]
                        if len(tr_t) < 2 or len(te_t) < 2:
                            rejected_reasons["uneven_split"] += 1; continue
                        tr_s = stats(tr_t); te_s = stats(te_t)
                        if tr_s["tot"] <= 0:
                            rejected_reasons["train_loss"] += 1; continue
                        if te_s["tot"] <= 0:
                            rejected_reasons["test_loss"] += 1; continue
                        if min(tr_s["min"], te_s["min"]) <= -7000:
                            rejected_reasons["bad_worst_trade"] += 1; continue
                        full_s = stats(all_trades)
                        # stability score = total + 0.5 * smaller half
                        stability = full_s["tot"] + 0.5 * min(tr_s["tot"], te_s["tot"])
                        survivors.append({
                            "wd": wd, "sym": sym, "strat": sname,
                            "en": e, "ex": x, "sl": sl,
                            "train": tr_s, "test": te_s, "full": full_s,
                            "stab": stability,
                            "trades": all_trades,
                        })

    print(f"  survivors (profitable in BOTH halves, worst > -₹7k, n≥4): {len(survivors)}")
    print(f"  rejections: {dict(rejected_reasons)}\n")

    # Top survivors per (weekday × index)
    by_bk = defaultdict(list)
    for s in survivors:
        by_bk[(s["wd"], s["sym"])].append(s)
    for k in by_bk:
        by_bk[k].sort(key=lambda c: -c["stab"])

    print("══════════ ROBUST TOP-3 per (weekday × index) — passed BOTH halves ══════════")
    print(f"  {'WD':<4}{'Sym':<7}{'Rk':<3}{'Strategy':<17}{'En':<7}{'→Ex':<7}{'SL%':>4}"
          f"{'Train':>10}{'Test':>10}{'Full':>11}{'Worst':>9}  {'Sharpe':>7} W/N")
    order = sorted(by_bk.keys(), key=lambda k: -by_bk[k][0]["stab"] if by_bk[k] else 0)
    for k in order:
        if not by_bk[k]:
            print(f"  {k[0]:<4}{k[1]:<7}— no candidate passed both halves —")
            continue
        for rank, c in enumerate(by_bk[k][:3], 1):
            tag = " *NEW" if c["strat"] in NEW_STRATS else ""
            wn = f"{c['full']['wins']}/{c['full']['n']}"
            print(f"  {k[0]:<4}{k[1]:<7}{rank:<3}{c['strat']:<17}{c['en']:<6}{c['ex']:<6}{c['sl']:>4}"
                  f"  ₹{c['train']['tot']:>+7,.0f} ₹{c['test']['tot']:>+7,.0f}"
                  f"  ₹{c['full']['tot']:>+8,.0f} ₹{c['full']['min']:>+6,.0f}"
                  f"  {c['full']['sharpe_like']:>+6.2f}  {wn}{tag}")

    # ── Build ROBUST schedule (best surviving per weekday across both indices) ──
    print("\n══════════ ROBUST RECOMMENDATION — best surviving per weekday ══════════")
    best_per_wd = {}
    for s in survivors:
        wd = s["wd"]
        if wd not in best_per_wd or s["stab"] > best_per_wd[wd]["stab"]:
            best_per_wd[wd] = s

    tot = tr_tot = te_tot = 0; ntr = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        s = best_per_wd.get(wd)
        if not s:
            print(f"  {wd}: ── NO ROBUST CANDIDATE (skip this day) ──"); continue
        tag = " *NEW" if s["strat"] in NEW_STRATS else ""
        print(f"  {wd}: {s['sym']:<7}{s['strat']:<17}{s['en']}→{s['ex']}  SL={s['sl']}%   "
              f"Train=₹{s['train']['tot']:+,.0f}  Test=₹{s['test']['tot']:+,.0f}  "
              f"Full=₹{s['full']['tot']:+,.0f}  worst=₹{s['full']['min']:+,.0f}  "
              f"Sharpe≈{s['full']['sharpe_like']:+.2f}  W/N={s['full']['wins']}/{s['full']['n']}{tag}")
        tot += s["full"]["tot"]; tr_tot += s["train"]["tot"]; te_tot += s["test"]["tot"]; ntr += s["full"]["n"]
    print(f"\n  ROBUST TOTAL: ₹{tot:+,.0f}   Train=₹{tr_tot:+,.0f}  Test=₹{te_tot:+,.0f}   trades={ntr}")
    print(f"  (Test = pure out-of-sample. Should be ≥ Train for healthy strategy.)")

if __name__ == "__main__":
    main()
