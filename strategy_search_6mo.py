#!/usr/bin/env python3
"""6-MONTH STRATEGY SEARCH (Dec 2025 – May 26, 2026)

Reuses helpers from phase3a_breakdown_v2 (psql, parse_expiry, load_chain, etc.)
and ultimate_sweep (pnl_for for straddle/strangle).

Adds new strategy families:
  - Short straddle (offset 0)
  - Short strangle (offset 1, 2, 3)
  - Iron fly  : short ATM straddle + long ±W strangle (defined-risk variant)
  - Iron condor: short ±S strangle + long ±W strangle  (S < W)
  - Naked CE / Naked PE   (single-leg directional)
  - 1x2 Ratio (sell 1 ATM call/put + buy 2 OTM same side) -- credit version

For each (weekday × index × strategy × offset/width × expiry-rank × entry × exit)
report best by total PnL with min-sample filter, then build PHASE-3a (one index/day).

Regime split: also report Pre (Dec 1 - Apr 14) vs Post (Apr 15 - May 26).
"""
from collections import defaultdict
from datetime import datetime, date
import json, sys
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, LOT_SIZE, LOTS, STRIKE_STEP)

START = "2025-12-01"
END   = "2026-05-26"
REGIME_SPLIT = date(2026, 4, 15)

ENTRIES = ["09:20", "09:30", "09:45", "10:00", "10:30", "11:00"]
EXITS   = ["10:30", "11:00", "11:30", "12:00", "12:30",
           "13:00", "13:30", "14:00", "14:30", "15:00", "15:15"]
MIN_SAMPLE = 8        # >=8 days per (wd, sym, ...) bucket required
COVERAGE   = 0.70     # require >=70% of days have valid prices

# Strategy families
#   straddle:   short CE(atm)+PE(atm)
#   strangle_N: short CE(atm+N) + PE(atm-N), N in {1,2,3}
#   ironfly_W:  short straddle (atm) + long CE(atm+W) + long PE(atm-W) (W>=2)
#   ironcondor_S_W: short CE(atm+S)+PE(atm-S) + long CE(atm+W)+PE(atm-W) (W>S)
#   naked_ce:   short CE only (atm or otm)  -- directional bearish
#   naked_pe:   short PE only (atm or otm)  -- directional bullish
#   ratio_call: short 1 ATM CE, long 2 (atm+W) CE
#   ratio_put : short 1 ATM PE, long 2 (atm-W) PE

def price_at(bars, strike, opt, t_min):
    s = bars.get((strike, opt))
    if not s: return None
    for m in range(t_min, t_min+5):
        if m in s: return s[m]
    return None

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

# ---------- per-strategy PnL ----------
def _short(entry_px, exit_px): return entry_px - exit_px       # sell, buy back
def _long(entry_px, exit_px):  return exit_px - entry_px       # buy, sell

def pnl_straddle(bars, sym, atm, e_min, x_min, off=0):
    step = STRIKE_STEP[sym]
    ce_k = atm + off*step; pe_k = atm - off*step
    ce_e = price_at(bars, ce_k, "CE", e_min); pe_e = price_at(bars, pe_k, "PE", e_min)
    ce_x = price_at(bars, ce_k, "CE", x_min); pe_x = price_at(bars, pe_k, "PE", x_min)
    if None in (ce_e, pe_e, ce_x, pe_x): return None
    pts = _short(ce_e, ce_x) + _short(pe_e, pe_x)
    return pts * LOT_SIZE[sym] * LOTS

def pnl_ironfly(bars, sym, atm, e_min, x_min, W):
    step = STRIKE_STEP[sym]
    sc, sp = atm, atm                              # short straddle
    lc, lp = atm + W*step, atm - W*step            # long wings
    p = []
    for k, side in [(sc,"CE"), (sp,"PE")]:
        e = price_at(bars, k, side, e_min); x = price_at(bars, k, side, x_min)
        if None in (e,x): return None
        p.append(_short(e,x))
    for k, side in [(lc,"CE"), (lp,"PE")]:
        e = price_at(bars, k, side, e_min); x = price_at(bars, k, side, x_min)
        if None in (e,x): return None
        p.append(_long(e,x))
    return sum(p) * LOT_SIZE[sym] * LOTS

def pnl_ironcondor(bars, sym, atm, e_min, x_min, S, W):
    step = STRIKE_STEP[sym]
    sc, sp = atm + S*step, atm - S*step
    lc, lp = atm + W*step, atm - W*step
    p = []
    for k, side in [(sc,"CE"), (sp,"PE")]:
        e = price_at(bars, k, side, e_min); x = price_at(bars, k, side, x_min)
        if None in (e,x): return None
        p.append(_short(e,x))
    for k, side in [(lc,"CE"), (lp,"PE")]:
        e = price_at(bars, k, side, e_min); x = price_at(bars, k, side, x_min)
        if None in (e,x): return None
        p.append(_long(e,x))
    return sum(p) * LOT_SIZE[sym] * LOTS

def pnl_naked(bars, sym, atm, e_min, x_min, side, off):
    """Short single leg, side='CE' or 'PE', off positive => OTM."""
    step = STRIKE_STEP[sym]
    k = atm + off*step if side=="CE" else atm - off*step
    e = price_at(bars, k, side, e_min); x = price_at(bars, k, side, x_min)
    if None in (e,x): return None
    return _short(e,x) * LOT_SIZE[sym] * LOTS

def pnl_ratio(bars, sym, atm, e_min, x_min, side, W):
    """Short 1 ATM, long 2 (atm±W) same side. W>=2. Credit if 2*long < short."""
    step = STRIKE_STEP[sym]
    short_k = atm
    long_k  = atm + W*step if side=="CE" else atm - W*step
    se = price_at(bars, short_k, side, e_min); sx = price_at(bars, short_k, side, x_min)
    le = price_at(bars, long_k,  side, e_min); lx = price_at(bars, long_k,  side, x_min)
    if None in (se,sx,le,lx): return None
    pts = _short(se,sx) + 2*_long(le,lx)
    return pts * LOT_SIZE[sym] * LOTS

# ---------- data loading ----------
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

# ---------- catalog of strategies ----------
def build_catalog(sym):
    """Yield (strategy_name, fn(bars, atm, e, x))."""
    cat = []
    # straddle / strangles
    for off in [0,1,2,3]:
        name = "straddle" if off==0 else f"strangle+{off}"
        cat.append((name, lambda b,a,e,x, o=off: pnl_straddle(b, sym, a, e, x, o)))
    # iron flies
    for W in [2,3,4,5]:
        cat.append((f"ironfly_W{W}", lambda b,a,e,x, W=W: pnl_ironfly(b, sym, a, e, x, W)))
    # iron condors
    for S,W in [(1,3),(1,4),(2,4),(2,5),(3,5),(3,6)]:
        cat.append((f"condor_S{S}W{W}", lambda b,a,e,x, S=S,W=W: pnl_ironcondor(b, sym, a, e, x, S, W)))
    # naked legs
    for side in ["CE","PE"]:
        for off in [0,1,2,3]:
            tag = "ATM" if off==0 else f"+{off}"
            cat.append((f"naked_{side}_{tag}", lambda b,a,e,x, side=side, off=off: pnl_naked(b, sym, a, e, x, side, off)))
    # 1x2 ratios
    for side in ["CE","PE"]:
        for W in [2,3,4]:
            cat.append((f"ratio_{side}_W{W}", lambda b,a,e,x, side=side, W=W: pnl_ratio(b, sym, a, e, x, side, W)))
    return cat

# ---------- main ----------
def main():
    near = find_days()
    print(f"(date,sym) combos: {len(near)}   window {START} → {END}", flush=True)

    print("Loading chains (nearest weekly only — keeps it manageable)...", flush=True)
    cache = {}
    items = list(near.items())
    for i, ((d, sym), exps) in enumerate(items, 1):
        ex, exp_s = exps[0]
        bars = load_chain(d, sym, exp_s)
        if bars: cache[(d, sym)] = (bars, (ex-d).days, exp_s)
        if i % 40 == 0:
            print(f"  [{i}/{len(items)}] cached={len(cache)}", flush=True)
    print(f"Cached chains: {len(cache)}\n", flush=True)

    # Group by weekday × sym
    by_bucket = defaultdict(list)   # (wd, sym) -> [(d, bars, dte)]
    for (d, sym), (bars, dte, _) in cache.items():
        by_bucket[(d.strftime("%a"), sym)].append((d, bars, dte))

    # Per (wd, sym), iterate strategies & windows
    # Track: best per (wd, sym, strat) regime-aware
    results = []   # list of dicts
    for (wd, sym), days in by_bucket.items():
        if len(days) < MIN_SAMPLE: continue
        catalog = build_catalog(sym)
        # pre-compute ATM per day at every entry
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
                    pnls_pre = []
                    pnls_post = []
                    for d, bars, _ in days:
                        atm = atm_cache.get((d, e))
                        if atm is None: continue
                        p = fn(bars, atm, em, xm)
                        if p is None: continue
                        pnls.append((d, p))
                        if d < REGIME_SPLIT: pnls_pre.append(p)
                        else: pnls_post.append(p)
                    n = len(pnls)
                    if n < len(days) * COVERAGE: continue
                    cum = sum(p for _,p in pnls)
                    w   = sum(1 for _,p in pnls if p > 0)
                    worst = min(p for _,p in pnls)
                    bestd = max(p for _,p in pnls)
                    results.append({
                        "wd": wd, "sym": sym, "strat": strat_name,
                        "entry": e, "exit": x,
                        "n": n, "win": w, "cum": cum,
                        "worst": worst, "best_day": bestd,
                        "pre_cum": sum(pnls_pre), "pre_n": len(pnls_pre),
                        "pre_win": sum(1 for p in pnls_pre if p>0),
                        "post_cum": sum(pnls_post), "post_n": len(pnls_post),
                        "post_win": sum(1 for p in pnls_post if p>0),
                    })

    print(f"Total result rows: {len(results)}", flush=True)

    # ========= REPORT 1: Top-5 per (wd, sym) overall =========
    print("\n══════════ TOP-5 per (WEEKDAY, INDEX) — FULL 6-MO ══════════")
    by_ws = defaultdict(list)
    for r in results:
        by_ws[(r["wd"], r["sym"])].append(r)
    order_wd = ["Mon","Tue","Wed","Thu","Fri"]
    for wd in order_wd:
        for sym in ["NIFTY","SENSEX"]:
            rs = by_ws.get((wd, sym), [])
            if not rs: continue
            rs.sort(key=lambda r: -r["cum"])
            print(f"\n  {wd} {sym}:")
            print(f"    {'Strategy':18} {'Win':3} {'En':5}→{'Ex':5}  {'Cum':>11}  {'Worst':>10}  {'PrePost':>16}")
            for r in rs[:5]:
                pre = fmt_rs(r['pre_cum']) if r['pre_n'] else "n/a"
                post = fmt_rs(r['post_cum']) if r['post_n'] else "n/a"
                wpct = int(100*r['win']/r['n'])
                print(f"    {r['strat']:18} {wpct:>2}% {r['entry']}→{r['exit']}  {fmt_rs(r['cum']):>11}  {fmt_rs(r['worst']):>10}  pre={pre} post={post}")

    # ========= REPORT 2: PHASE-3a (one index per day, best strategy) =========
    print("\n══════════ NEW PHASE-3a (one index/day, ANY strategy) ══════════")
    by_wd = defaultdict(list)
    for r in results:
        by_wd[r["wd"]].append(r)
    chosen_3a = {}
    grand = ntot = wtot = 0
    pre_grand = post_grand = 0
    for wd in order_wd:
        opts = by_wd.get(wd, [])
        if not opts: continue
        # filter: require post-regime n>=4 and post_cum>0 (regime-stable)
        stable = [r for r in opts if r["post_n"] >= 4 and r["post_cum"] > 0]
        if not stable: stable = opts  # fallback
        stable.sort(key=lambda r: -r["cum"])
        r = stable[0]
        chosen_3a[wd] = r
        print(f"  {wd}: {r['sym']:6} {r['strat']:18} {r['entry']}→{r['exit']}  "
              f"n={r['n']:2d} win={int(100*r['win']/r['n']):3d}%  cum={fmt_rs(r['cum']):>10}  "
              f"pre={fmt_rs(r['pre_cum']):>10} post={fmt_rs(r['post_cum']):>9}  worst={fmt_rs(r['worst'])}")
        grand += r["cum"]; ntot += r["n"]; wtot += r["win"]
        pre_grand += r["pre_cum"]; post_grand += r["post_cum"]
    if ntot:
        print(f"\n  PHASE-3a TOTAL: {fmt_rs(grand)} over {ntot} trades, {int(100*wtot/ntot)}% win  ~{fmt_rs(grand/6)}/mo")
        print(f"  Pre-regime  cum: {fmt_rs(pre_grand)}")
        print(f"  Post-regime cum: {fmt_rs(post_grand)}  (≈ {fmt_rs(post_grand/1.5)}/mo if Apr15+ holds)")

    # ========= REPORT 3: Monthly replay of chosen 3a =========
    print("\n══════════ MONTHLY P&L of PHASE-3a winner ══════════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for (d, sym), (bars, _, _) in cache.items():
        wd = d.strftime("%a")
        if wd not in chosen_3a: continue
        r = chosen_3a[wd]
        if sym != r["sym"]: continue
        # need to re-run the strategy fn for this (d, sym)
        atm = find_atm(bars, to_min(r["entry"]), sym)
        if atm is None: continue
        catalog = dict(build_catalog(sym))
        p = catalog[r["strat"]](bars, atm, to_min(r["entry"]), to_min(r["exit"]))
        if p is None: continue
        m = d.strftime("%Y-%m")
        by_m[m]["pnl"] += p; by_m[m]["n"] += 1
        if p > 0: by_m[m]["w"] += 1
    print(f"  {'Month':10} {'Days':>5} {'Win%':>5}  {'PnL':>12}")
    gp = gn = gw = 0
    for m in sorted(by_m):
        v = by_m[m]; gp += v["pnl"]; gn += v["n"]; gw += v["w"]
        print(f"  {m:10} {v['n']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>12}")
    if gn:
        print(f"  {'TOTAL':10} {gn:>5} {int(100*gw/gn):>4}%  {fmt_rs(gp):>12}")

    # Persist JSON for offline review
    out = {"window":[START,END], "regime_split":str(REGIME_SPLIT),
           "phase3a": {wd: {k:(v if not isinstance(v,float) else round(v,2)) for k,v in r.items()} for wd,r in chosen_3a.items()}}
    with open("/tmp/strategy_search_6mo.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/strategy_search_6mo.json")

if __name__ == "__main__":
    main()
