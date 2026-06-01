#!/usr/bin/env python3
"""Phase-3a SL-ONLY — same 8 strategies, NO TP gate. SL=-₹6,000 + SL% + TIME.
Fine entry/exit grid, one index per day, walk-forward validation.

Strategies: straddle, strangle+1, strangle+2,
            ironfly_W2, ironfly_W3, ironfly_W4,
            condor_S1W3, condor_S2W4

Exit logic (priority):
  1. (TP DISABLED — let winners run)
  2. MTM ≤ -₹6,000  → SL_RS
  3. MTM ≤ -SL% of credit → SL_PCT (defined-risk safety)
  4. Time gate reached → TIME
"""
from collections import defaultdict
from datetime import date
from statistics import mean, pstdev
from phase3a_breakdown_v2 import to_min, load_chain, LOTS, STRIKE_STEP
from strategy_definedrisk_sweep import (find_atm, get_price, find_days,
                                         legs_straddle, legs_strangle,
                                         legs_ironfly, legs_condor)

LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}
TP_RS = +1e12  # DISABLED — let winners run to TIME exit
SL_RS = -6000.0
REGIME_START = date(2026, 4, 14)
SPLIT_DATE   = date(2026, 5, 5)
APR_END      = date(2026, 4, 30)
END_DATE     = date(2026, 5, 29)

# FINER GRID
ENTRIES = ["09:20","09:25","09:30","09:35","09:45","10:00","10:15","10:30"]
EXITS   = ["11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]
SL_PCTS = [30, 50, 75, 100]

STRATS_BUILDERS = lambda sym, step: {
    "straddle":    lambda atm: legs_straddle(atm, step),
    "strangle+1":  lambda atm: legs_strangle(atm, step, 1),
    "strangle+2":  lambda atm: legs_strangle(atm, step, 2),
    "ironfly_W2":  lambda atm: legs_ironfly(atm, step, 2),
    "ironfly_W3":  lambda atm: legs_ironfly(atm, step, 3),
    "ironfly_W4":  lambda atm: legs_ironfly(atm, step, 4),
    "condor_S1W3": lambda atm: legs_condor(atm, step, 1, 3),
    "condor_S2W4": lambda atm: legs_condor(atm, step, 2, 4),
}

def sim_tp_sl(bars, legs, em, xm, sl_pct, lot):
    """Simulate with TP +₹6000 / SL -₹6000 / SL% / time exit."""
    ent = []
    for k, side, qty in legs:
        e = get_price(bars, k, side, em)
        if e is None: return None
        ent.append((k, side, qty, e))
    credit = sum((-q) * e for _, _, q, e in ent)
    if credit <= 0: return None
    sl_pts = credit * sl_pct / 100.0
    N = lot * LOTS
    last = None
    for m in range(em + 1, xm + 1):
        ok = True; mtm = 0.0
        for k, side, qty, e in ent:
            p = get_price(bars, k, side, m)
            if p is None: ok = False; break
            if qty < 0: mtm += abs(qty) * (e - p)
            else:       mtm += abs(qty) * (p - e)
        if not ok: continue
        rs = mtm * N
        if rs >= TP_RS:    return rs, "TP"
        if rs <= SL_RS:    return rs, "SL_RS"
        if mtm <= -sl_pts: return mtm * N, "SL_PCT"
        last = mtm
    if last is None: return None
    return last * N, "TIME"

def stats(trades):
    pnls = [p for _,p in trades]
    if not pnls: return None
    return {
        "n": len(pnls), "tot": sum(pnls),
        "avg": mean(pnls), "std": pstdev(pnls) if len(pnls)>1 else 0,
        "min": min(pnls), "max": max(pnls),
        "wins": sum(1 for p in pnls if p > 0),
        "sharpe": (mean(pnls)/pstdev(pnls)) if len(pnls)>1 and pstdev(pnls)>0 else 0,
    }

def main():
    print("Data source: option_candles ← DhanHQ /v2/charts/rollingoption (1-min OHLC)")
    print(f"Window:      {REGIME_START} → {END_DATE}   (post-regime, Apr 14 expiry shift)")
    print(f"Gates:       TP=DISABLED  SL=−₹{abs(SL_RS):,.0f}  + SL%-of-credit safety  +  TIME exit")
    print(f"Lots:        {LOTS} (NIFTY={LOT_SIZE['NIFTY']}×{LOTS}={LOT_SIZE['NIFTY']*LOTS}, "
          f"SENSEX={LOT_SIZE['SENSEX']}×{LOTS}={LOT_SIZE['SENSEX']*LOTS})")
    print(f"Grid:        {len(ENTRIES)} entries × {len(EXITS)} exits × {len(SL_PCTS)} SL% × "
          f"8 strategies\n")

    near = find_days()
    cache = {}
    for (d, sym), exps in near.items():
        if d < REGIME_START or d > END_DATE: continue
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = bars
    print(f"Loaded {len(cache)} (date,sym) days from Dhan-sourced DB\n")

    by_bk = defaultdict(list)
    for (d, sym), bars in cache.items():
        by_bk[(d.strftime("%a"), sym)].append((d, bars))

    print("Sweeping fine grid with walk-forward validation...")
    survivors = []
    rej = defaultdict(int)
    for (wd, sym), days in by_bk.items():
        step = STRIKE_STEP[sym]; lot = LOT_SIZE[sym]
        for sname, leg_fn in STRATS_BUILDERS(sym, step).items():
            for e in ENTRIES:
                em = to_min(e)
                for x in EXITS:
                    xm = to_min(x)
                    if xm <= em + 30: continue   # require >=30 min hold
                    for sl in SL_PCTS:
                        all_t = []
                        for d, bars in days:
                            atm = find_atm(bars, em, sym)
                            if atm is None: continue
                            legs = leg_fn(atm)
                            r = sim_tp_sl(bars, legs, em, xm, sl, lot)
                            if r is None: continue
                            all_t.append((d, r[0]))
                        if len(all_t) < 4: rej["n<4"] += 1; continue
                        tr_t = [(d,p) for d,p in all_t if d <= SPLIT_DATE]
                        te_t = [(d,p) for d,p in all_t if d  > SPLIT_DATE]
                        if len(tr_t) < 2 or len(te_t) < 2: rej["unsplit"] += 1; continue
                        tr_s = stats(tr_t); te_s = stats(te_t)
                        if tr_s["tot"] <= 0: rej["train_loss"] += 1; continue
                        if te_s["tot"] <= 0: rej["test_loss"] += 1; continue
                        full_s = stats(all_t)
                        # stability: total + 0.5 * smaller half
                        stab = full_s["tot"] + 0.5 * min(tr_s["tot"], te_s["tot"])
                        survivors.append({
                            "wd": wd, "sym": sym, "strat": sname,
                            "en": e, "ex": x, "sl": sl,
                            "train": tr_s, "test": te_s, "full": full_s,
                            "stab": stab, "trades": all_t,
                        })
    print(f"  survivors: {len(survivors)}   rejections: {dict(rej)}\n")

    # ── TOP-5 per (weekday × index) ──
    bk = defaultdict(list)
    for s in survivors: bk[(s["wd"], s["sym"])].append(s)
    for k in bk: bk[k].sort(key=lambda c: -c["stab"])

    print("══════════ TOP-5 per (weekday × index) — PHASE 3a SL-ONLY (no TP) ══════════")
    print(f"  {'WD':<4}{'Sym':<7}{'Rk':<3}{'Strategy':<13}{'En':<6}{'→Ex':<7}{'SL%':>4}"
          f"{'Train':>9}{'Test':>9}{'Full':>10}{'Worst':>9}{'Sh':>6}  W/N")
    order = sorted(bk.keys(), key=lambda k: -bk[k][0]["stab"] if bk[k] else 0)
    for k in order:
        if not bk[k]:
            print(f"  {k[0]:<4}{k[1]:<7} — NO survivor —"); continue
        for rk, c in enumerate(bk[k][:5], 1):
            wn = f"{c['full']['wins']}/{c['full']['n']}"
            print(f"  {k[0]:<4}{k[1]:<7}{rk:<3}{c['strat']:<13}{c['en']:<6}{c['ex']:<6}{c['sl']:>4}"
                  f" ₹{c['train']['tot']:>+6,.0f} ₹{c['test']['tot']:>+6,.0f}"
                  f" ₹{c['full']['tot']:>+7,.0f} ₹{c['full']['min']:>+6,.0f}"
                  f" {c['full']['sharpe']:>+5.2f}  {wn}")

    # ── Pick best surviving per weekday (one index per day) ──
    print("\n══════════ FINAL RECOMMENDATION — PHASE 3a SL-ONLY (SL=-₹6k + TIME, one index/day) ══════════")
    best_wd = {}
    for s in survivors:
        wd = s["wd"]
        if wd not in best_wd or s["stab"] > best_wd[wd]["stab"]:
            best_wd[wd] = s
    tot=ap=my=0; n=0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        s = best_wd.get(wd)
        if not s:
            print(f"  {wd}: ── NO ROBUST CANDIDATE — skip ──"); continue
        print(f"  {wd}: {s['sym']:<7}{s['strat']:<13}{s['en']}→{s['ex']}  SL={s['sl']}%   "
              f"Train=₹{s['train']['tot']:+,.0f}  Test=₹{s['test']['tot']:+,.0f}  "
              f"Full=₹{s['full']['tot']:+,.0f}  worst=₹{s['full']['min']:+,.0f}  "
              f"Sh={s['full']['sharpe']:+.2f}  W/N={s['full']['wins']}/{s['full']['n']}")
        tot += s["full"]["tot"]; n += s["full"]["n"]
        ap += sum(p for d,p in s["trades"] if d <= APR_END)
        my += sum(p for d,p in s["trades"] if d  > APR_END)
    print(f"\n  PHASE-3a TOTAL: ₹{tot:+,.0f}   Apr=₹{ap:+,.0f}  May=₹{my:+,.0f}   trades={n}")

    # ── Trade-by-trade for the final schedule ──
    print("\n══════════ Trade-by-trade ══════════")
    flat = []
    for wd, s in best_wd.items():
        atm_strat = (s["sym"], s["strat"], s["en"], s["ex"], s["sl"])
        # re-simulate to recover exit reasons
        step = STRIKE_STEP[s["sym"]]; lot = LOT_SIZE[s["sym"]]
        leg_fn = STRATS_BUILDERS(s["sym"], step)[s["strat"]]
        em = to_min(s["en"]); xm = to_min(s["ex"])
        for d, bars in by_bk[(wd, s["sym"])]:
            atm = find_atm(bars, em, s["sym"])
            if atm is None: continue
            r = sim_tp_sl(bars, leg_fn(atm), em, xm, s["sl"], lot)
            if r is None: continue
            flat.append((d, wd, s["sym"], s["strat"], s["en"], s["ex"], r[1], r[0]))
    cum = 0
    print(f"  {'#':<3}{'Date':<11} {'WD':<4}{'Sym':<7}{'Strategy':<13}{'En':<6}{'→Ex':<6}"
          f"{'Why':<8}{'PnL':>10}{'Cum':>11}")
    for i, (d, wd, sym, strat, en, ex, why, pnl) in enumerate(sorted(flat), 1):
        cum += pnl
        print(f"  {i:<3}{d.isoformat()}  {wd:<4}{sym:<7}{strat:<13}{en:<6}{ex:<5} "
              f"{why:<7} ₹{pnl:>+8,.0f} ₹{cum:>+9,.0f}")

if __name__ == "__main__":
    main()
