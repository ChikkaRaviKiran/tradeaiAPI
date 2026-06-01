#!/usr/bin/env python3
"""DEFINED-RISK SWEEP with stop-loss & take-profit (6 months).

Strategies (all delta-neutral / defined-risk, no naked single legs):
  - straddle (ATM)
  - strangle +1, +2 (OTM both sides)
  - ironfly W2, W3, W4  (short ATM straddle + long ±W strangle)
  - condor S1W3, S2W4   (short strangle + long farther strangle)

Per-trade exits:
  - Scheduled exit time
  - Loss-based stoploss: if MTM loss >= SL% of entry credit → exit
  - Take-profit:         if MTM profit >= TP% of entry credit → exit
SL grid: 30, 50, 75, 100 (% of credit) ; TP grid: None, 30, 50 (% of credit)

For each (weekday × index) finds best combo by 6-mo cum P&L, with regime split.
"""
from collections import defaultdict
from datetime import datetime, date
import json
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, LOT_SIZE, LOTS, STRIKE_STEP)

START = "2025-12-01"
END   = "2026-05-26"
REGIME_SPLIT = date(2026, 4, 15)

ENTRIES = ["09:20", "09:30", "09:45", "10:00"]
EXITS   = ["12:30", "13:30", "14:30", "15:00", "15:15"]
SL_PCTS = [30, 50, 75, 100]
TP_PCTS = [None, 30, 50]
MIN_SAMPLE = 8
COVERAGE   = 0.70

# --------------- helpers ---------------
def find_atm(bars, t_min, sym):
    step = STRIKE_STEP[sym]
    strikes = sorted({k[0] for k in bars.keys()})
    best=None; bd=None
    for s in strikes:
        ce=pe=None
        for m in range(t_min, t_min+5):
            if ce is None: ce = bars.get((s,"CE"),{}).get(m)
            if pe is None: pe = bars.get((s,"PE"),{}).get(m)
            if ce is not None and pe is not None: break
        if ce is None or pe is None: continue
        d = abs(ce-pe)
        if best is None or d<bd: bd=d; best=s
    if best is None: return None
    return int(round(best/step)*step) if best % step else best

def get_price(bars, k, side, m):
    """Get price at minute m, allow 4-min slack forward."""
    s = bars.get((k, side))
    if not s: return None
    for mm in range(m, m+5):
        if mm in s: return s[mm]
    return None

# --------------- strategy leg builders ---------------
def legs_straddle(atm, step):
    return [(atm, "CE", -1), (atm, "PE", -1)]  # (strike, side, qty: -1=short, +1=long)

def legs_strangle(atm, step, off):
    return [(atm + off*step, "CE", -1), (atm - off*step, "PE", -1)]

def legs_ironfly(atm, step, W):
    return [(atm, "CE", -1), (atm, "PE", -1),
            (atm + W*step, "CE", +1), (atm - W*step, "PE", +1)]

def legs_condor(atm, step, S, W):
    return [(atm + S*step, "CE", -1), (atm - S*step, "PE", -1),
            (atm + W*step, "CE", +1), (atm - W*step, "PE", +1)]

def build_strategies(sym):
    step = STRIKE_STEP[sym]
    return {
        "straddle":     lambda atm: legs_straddle(atm, step),
        "strangle+1":   lambda atm: legs_strangle(atm, step, 1),
        "strangle+2":   lambda atm: legs_strangle(atm, step, 2),
        "ironfly_W2":   lambda atm: legs_ironfly(atm, step, 2),
        "ironfly_W3":   lambda atm: legs_ironfly(atm, step, 3),
        "ironfly_W4":   lambda atm: legs_ironfly(atm, step, 4),
        "condor_S1W3":  lambda atm: legs_condor(atm, step, 1, 3),
        "condor_S2W4":  lambda atm: legs_condor(atm, step, 2, 4),
    }

# --------------- per-trade simulator with TP/SL ---------------
def simulate(bars, legs, e_min, x_min, sl_pct, tp_pct, lotsize):
    """Return (pnl_rupees, exit_min, exit_reason) or None if no prices.

    Credit (points) = sum_over_short(entry_premium) - sum_over_long(entry_premium)
    MTM at minute m = sum_over_legs( qty * (entry_px - current_px) )
                    = credit_received - current_close_cost
    For SL/TP triggers we scan every minute (with 4-min slack) between entry and exit.
    """
    entries = []
    for k, side, qty in legs:
        e = get_price(bars, k, side, e_min)
        if e is None: return None
        entries.append((k, side, qty, e))
    # credit (in points, positive if net credit)
    credit = sum(qty * -e if qty == -1 else qty * -e for qty, e in
                 [(q, ep) for _,_,q,ep in entries])
    # qty*-e for short (qty=-1) gives +e (receive premium); for long (qty=+1) gives -e (pay)
    # Net credit_points = +e_short_total - e_long_total
    if credit <= 0:
        return None  # debit spread — skip
    sl_amt = credit * sl_pct / 100.0
    tp_amt = credit * tp_pct / 100.0 if tp_pct else None

    # iterate minute by minute
    for m in range(e_min + 1, x_min + 1):
        # get current price for each leg
        currents = []
        ok = True
        for k, side, qty, _ in entries:
            p = get_price(bars, k, side, m)
            if p is None:
                ok = False; break
            currents.append(p)
        if not ok:
            continue  # skip minute if missing; will exit at x_min anyway
        # MTM (in points). For each leg: qty*(entry - current)? No.
        # short pnl = entry - current (receive entry, pay current to close)
        # long pnl = current - entry
        mtm = 0.0
        for (k, side, qty, e), cur in zip(entries, currents):
            if qty == -1: mtm += (e - cur)
            else:         mtm += (cur - e)
        if tp_amt is not None and mtm >= tp_amt:
            return (mtm * lotsize * LOTS, m, "TP")
        if mtm <= -sl_amt:
            return (mtm * lotsize * LOTS, m, "SL")
    # scheduled exit
    currents = []
    for k, side, qty, _ in entries:
        p = get_price(bars, k, side, x_min)
        if p is None: return None
        currents.append(p)
    mtm = 0.0
    for (k, side, qty, e), cur in zip(entries, currents):
        if qty == -1: mtm += (e - cur)
        else:         mtm += (cur - e)
    return (mtm * lotsize * LOTS, x_min, "TIME")

# --------------- main ---------------
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

    print("Loading chains...", flush=True)
    cache = {}
    items = list(near.items())
    for i, ((d, sym), exps) in enumerate(items, 1):
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = (bars, (ex-d).days, exp_s)
        if i % 40 == 0: print(f"  [{i}/{len(items)}] cached={len(cache)}", flush=True)
    print(f"Cached chains: {len(cache)}\n", flush=True)

    by_bucket = defaultdict(list)
    for (d, sym), (bars, dte, _) in cache.items():
        by_bucket[(d.strftime("%a"), sym)].append((d, bars, dte))

    results = []
    for (wd, sym), days in by_bucket.items():
        if len(days) < MIN_SAMPLE: continue
        strat_map = build_strategies(sym)
        lot = LOT_SIZE[sym]
        # pre-compute ATM per (day, entry)
        atm_cache = {}
        for d, bars, _ in days:
            for e in ENTRIES:
                atm_cache[(d, e)] = find_atm(bars, to_min(e), sym)
        for strat_name, leg_fn in strat_map.items():
            for e in ENTRIES:
                em = to_min(e)
                for x in EXITS:
                    xm = to_min(x)
                    if xm <= em: continue
                    for sl in SL_PCTS:
                        for tp in TP_PCTS:
                            pnls = []
                            exits = defaultdict(int)
                            for d, bars, _ in days:
                                atm = atm_cache.get((d, e))
                                if atm is None: continue
                                legs = leg_fn(atm)
                                r = simulate(bars, legs, em, xm, sl, tp, lot)
                                if r is None: continue
                                pnl, ex_m, why = r
                                pnls.append((d, pnl, why))
                                exits[why] += 1
                            n = len(pnls)
                            if n < len(days) * COVERAGE: continue
                            cum = sum(p for _, p, _ in pnls)
                            w   = sum(1 for _, p, _ in pnls if p > 0)
                            pre = sum(p for d, p, _ in pnls if d < REGIME_SPLIT)
                            post = sum(p for d, p, _ in pnls if d >= REGIME_SPLIT)
                            pre_n = sum(1 for d, _, _ in pnls if d < REGIME_SPLIT)
                            post_n = n - pre_n
                            post_w = sum(1 for d, p, _ in pnls if d >= REGIME_SPLIT and p > 0)
                            worst = min(p for _, p, _ in pnls)
                            results.append({
                                "wd": wd, "sym": sym, "strat": strat_name,
                                "entry": e, "exit": x, "sl": sl, "tp": tp,
                                "n": n, "win": w, "cum": cum,
                                "pre": pre, "pre_n": pre_n,
                                "post": post, "post_n": post_n, "post_win": post_w,
                                "worst": worst,
                                "tp_hits": exits["TP"], "sl_hits": exits["SL"], "time_hits": exits["TIME"],
                            })
    print(f"Total result rows: {len(results)}\n", flush=True)

    # ===== TOP 5 per (wd, sym), filtered for regime robustness =====
    print("══════════ TOP-5 per (WEEKDAY, INDEX) — DEFINED-RISK + SL/TP ══════════")
    by_ws = defaultdict(list)
    for r in results: by_ws[(r["wd"], r["sym"])].append(r)
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            rs = by_ws.get((wd, sym), [])
            # Filter: post-regime must be profitable and have >=4 samples
            stable = [r for r in rs if r["post_n"] >= 4 and r["post"] > 0]
            if not stable: stable = rs
            stable.sort(key=lambda r: -r["cum"])
            if not stable: continue
            print(f"\n  {wd} {sym}:")
            print(f"    {'Strategy':14} {'En':5}→{'Ex':5} {'SL%':>3} {'TP%':>4}  {'n':>3} {'Win':>4}  {'Cum':>11}  {'Post':>10}  {'Worst':>9}  TP/SL/Time")
            for r in stable[:5]:
                tp = f"{r['tp']}" if r['tp'] else "-"
                print(f"    {r['strat']:14} {r['entry']}→{r['exit']} {r['sl']:>3} {tp:>4}  "
                      f"{r['n']:>3} {int(100*r['win']/r['n']):>3}%  {fmt_rs(r['cum']):>11}  "
                      f"{fmt_rs(r['post']):>10}  {fmt_rs(r['worst']):>9}  "
                      f"{r['tp_hits']}/{r['sl_hits']}/{r['time_hits']}")

    # ===== PHASE-3a SCHEDULE (one index/day) =====
    print("\n══════════ PHASE-3a SCHEDULE (one index/day, defined-risk + SL/TP) ══════════")
    by_wd = defaultdict(list)
    for r in results: by_wd[r["wd"]].append(r)
    chosen_3a = {}
    grand = ntot = wtot = 0; pre_g = post_g = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        opts = by_wd.get(wd, [])
        stable = [r for r in opts if r["post_n"] >= 4 and r["post"] > 0]
        if not stable: stable = opts
        stable.sort(key=lambda r: -r["cum"])
        if not stable: continue
        r = stable[0]
        chosen_3a[wd] = r
        tp = f"{r['tp']}%" if r['tp'] else "none"
        print(f"  {wd}: {r['sym']:6} {r['strat']:14} {r['entry']}→{r['exit']} "
              f"SL={r['sl']}% TP={tp:>5}  n={r['n']:2d} win={int(100*r['win']/r['n']):3d}%  "
              f"cum={fmt_rs(r['cum']):>10}  post={fmt_rs(r['post']):>9}  worst={fmt_rs(r['worst'])}")
        grand += r["cum"]; ntot += r["n"]; wtot += r["win"]
        pre_g += r["pre"]; post_g += r["post"]
    if ntot:
        print(f"\n  PHASE-3a TOTAL: {fmt_rs(grand)} / {ntot} trades / {int(100*wtot/ntot)}% win  ~{fmt_rs(grand/6)}/mo")
        print(f"  Pre-regime:  {fmt_rs(pre_g)}   Post-regime: {fmt_rs(post_g)} (≈ {fmt_rs(post_g/1.5)}/mo)")

    # ===== PHASE-3b SCHEDULE (both indexes/day) =====
    print("\n══════════ PHASE-3b SCHEDULE (BOTH NIFTY + SENSEX each day) ══════════")
    chosen_3b = {}
    grand = ntot = wtot = 0; pre_g = post_g = 0
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            opts = [r for r in by_ws.get((wd, sym), [])]
            stable = [r for r in opts if r["post_n"] >= 4 and r["post"] > 0]
            if not stable: stable = opts
            stable.sort(key=lambda r: -r["cum"])
            if not stable: continue
            r = stable[0]
            chosen_3b[(wd,sym)] = r
            tp = f"{r['tp']}%" if r['tp'] else "none"
            print(f"  {wd} {sym:6}: {r['strat']:14} {r['entry']}→{r['exit']} SL={r['sl']}% TP={tp:>5}  "
                  f"n={r['n']:2d} win={int(100*r['win']/r['n']):3d}% cum={fmt_rs(r['cum']):>10} post={fmt_rs(r['post']):>9} worst={fmt_rs(r['worst'])}")
            grand += r["cum"]; ntot += r["n"]; wtot += r["win"]
            pre_g += r["pre"]; post_g += r["post"]
    if ntot:
        print(f"\n  PHASE-3b TOTAL: {fmt_rs(grand)} / {ntot} trades / {int(100*wtot/ntot)}% win  ~{fmt_rs(grand/6)}/mo")
        print(f"  Pre-regime:  {fmt_rs(pre_g)}   Post-regime: {fmt_rs(post_g)} (≈ {fmt_rs(post_g/1.5)}/mo)")

    # ===== Monthly replay of Phase-3a winner =====
    print("\n══════════ MONTHLY P&L of PHASE-3a winner ══════════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0,"tp":0,"sl":0,"tm":0})
    for (d, sym), (bars, _, _) in cache.items():
        wd = d.strftime("%a")
        if wd not in chosen_3a: continue
        r = chosen_3a[wd]
        if sym != r["sym"]: continue
        atm = find_atm(bars, to_min(r["entry"]), sym)
        if atm is None: continue
        legs = build_strategies(sym)[r["strat"]](atm)
        sim = simulate(bars, legs, to_min(r["entry"]), to_min(r["exit"]),
                       r["sl"], r["tp"], LOT_SIZE[sym])
        if sim is None: continue
        p, _, why = sim
        m = d.strftime("%Y-%m")
        by_m[m]["pnl"] += p; by_m[m]["n"] += 1
        if p > 0: by_m[m]["w"] += 1
        by_m[m]["tp"] += (why == "TP"); by_m[m]["sl"] += (why == "SL"); by_m[m]["tm"] += (why == "TIME")
    print(f"  {'Month':10} {'Days':>5} {'Win%':>5}  {'PnL':>12}  TP/SL/Time")
    gp=gn=gw=0
    for m in sorted(by_m):
        v=by_m[m]; gp+=v["pnl"]; gn+=v["n"]; gw+=v["w"]
        print(f"  {m:10} {v['n']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>12}  {v['tp']}/{v['sl']}/{v['tm']}")
    if gn:
        print(f"  {'TOTAL':10} {gn:>5} {int(100*gw/gn):>4}%  {fmt_rs(gp):>12}")

    with open("/tmp/strategy_definedrisk.json","w") as f:
        json.dump({"window":[START,END], "phase3a":
                   {wd:{k:(v if not isinstance(v,float) else round(v,2)) for k,v in r.items()}
                    for wd,r in chosen_3a.items()},
                   "phase3b":
                   {f"{wd}_{sym}":{k:(v if not isinstance(v,float) else round(v,2)) for k,v in r.items()}
                    for (wd,sym),r in chosen_3b.items()}}, f, indent=2, default=str)
    print("\nWrote /tmp/strategy_definedrisk.json")

if __name__ == "__main__":
    main()
