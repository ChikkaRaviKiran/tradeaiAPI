#!/usr/bin/env python3
"""ULTIMATE SWEEP: optimise (entry, exit, offset, expiry) per (weekday, sym).

For each weekday × sym, find the combination of:
  - entry time (09:20..11:00)
  - exit time (10:30..15:15)
  - strike offset (0=ATM, 1/2/3 OTM strangle)
  - expiry rank (0=nearest weekly, 1=next weekly)
that maximizes profit across all sample days.

Then build the best Phase-3a (one index/day) and Phase-3b (both) schedules.
"""
from collections import defaultdict
from datetime import datetime
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, LOT_SIZE, LOTS, STRIKE_STEP)

START = "2026-01-01"
END   = "2026-05-22"

ENTRIES = ["09:20","09:30","09:45","10:00","10:30","11:00"]
EXITS   = ["10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:15"]
OFFSETS = [0, 1, 2, 3]    # 0=ATM straddle, N=±N strangle
MIN_SAMPLE = 6            # require >=6 days per (wd, sym, offset, exprank) bucket

def find_days():
    """Per (date, sym) return list of (expiry_date, expiry_str) sorted nearest first."""
    rows = psql(f"""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '{START}' AND date <= '{END}' ORDER BY date, instrument;""")
    by_day = defaultdict(list)
    for date_s, inst, exp_s in rows:
        try: d = datetime.strptime(date_s, "%Y-%m-%d").date()
        except Exception: continue
        ex = parse_expiry(exp_s)
        if ex is None or ex < d: continue
        by_day[(d, inst)].append((ex, exp_s))
    out = {}
    for k, lst in by_day.items():
        lst.sort()
        out[k] = lst
    return out

def find_atm(bars, t_min):
    strikes = sorted({k[0] for k in bars.keys()})
    best=None; bd=None
    for s in strikes:
        ce=pe=None
        for m in range(t_min, t_min+5):
            if ce is None: ce = bars.get((s,"CE"),{}).get(m)
            if pe is None: pe = bars.get((s,"PE"),{}).get(m)
            if ce is not None and pe is not None: break
        if ce is None or pe is None: continue
        diff = abs(ce-pe)
        if best is None or diff<bd: bd=diff; best=s
    return best

def price_at(bars, strike, opt, t_min):
    s = bars.get((strike, opt))
    if not s: return None
    for m in range(t_min, t_min+5):
        if m in s: return s[m]
    return None

def pnl_for(bars, sym, e_min, x_min, offset):
    atm = find_atm(bars, e_min)
    if atm is None: return None
    step = STRIKE_STEP[sym]
    ce_k = atm + offset*step
    pe_k = atm - offset*step
    ce_e = price_at(bars, ce_k, "CE", e_min)
    pe_e = price_at(bars, pe_k, "PE", e_min)
    ce_x = price_at(bars, ce_k, "CE", x_min)
    pe_x = price_at(bars, pe_k, "PE", x_min)
    if None in (ce_e, pe_e, ce_x, pe_x): return None
    return ((ce_e + pe_e) - (ce_x + pe_x)) * LOT_SIZE[sym] * LOTS

def main():
    near = find_days()
    print(f"Total (date, sym) combos: {len(near)}\n")

    # Pre-load chains for nearest + next expiry per day
    # cache: (date, sym, exprank) -> bars
    print("Loading chains (nearest + next weekly)...")
    cache = {}
    items = list(near.items())
    for i, ((d, sym), exps) in enumerate(items, 1):
        for rank, (ex, exp_s) in enumerate(exps[:2]):
            bars = load_chain(d, sym, exp_s)
            if bars:
                cache[(d, sym, rank)] = (bars, (ex-d).days, exp_s)
        if i % 30 == 0:
            print(f"  [{i}/{len(items)}] cached={len(cache)}")
    print(f"Cached chains: {len(cache)}\n")

    # Group by (wd, sym, exprank)
    by_bucket = defaultdict(list)  # (wd, sym, exprank) -> [(date, bars, dte), ...]
    for (d, sym, rank), (bars, dte, exp_s) in cache.items():
        wd = d.strftime("%a")
        by_bucket[(wd, sym, rank)].append((d, bars, dte))

    # Pre-compute pnl per (date, sym, rank, offset, entry, exit)
    # Sweep best window per (wd, sym, offset, rank)
    print("Sweeping entry × exit × offset × expiry...")
    best = {}   # (wd, sym, offset, rank) -> (cum, e, x, n, w, worst, best_pnl, dte_med)
    for k, days in by_bucket.items():
        if len(days) < MIN_SAMPLE: continue
        wd, sym, rank = k
        dtes = sorted([dte for _,_,dte in days])
        dte_med = dtes[len(dtes)//2]
        for off in OFFSETS:
            results = []
            for e in ENTRIES:
                em = to_min(e)
                for x in EXITS:
                    xm = to_min(x)
                    if xm <= em: continue
                    pnls = []
                    for _, bars, _ in days:
                        p = pnl_for(bars, sym, em, xm, off)
                        if p is not None: pnls.append(p)
                    if len(pnls) < len(days) * 0.7: continue
                    n = len(pnls); w = sum(1 for p in pnls if p>0); cum = sum(pnls)
                    results.append((cum, e, x, n, w, min(pnls), max(pnls)))
            if not results: continue
            results.sort(reverse=True)
            cum, e, x, n, w, mn, mx = results[0]
            best[(wd, sym, off, rank)] = (cum, e, x, n, w, mn, mx, dte_med)

    # Show top per (wd, sym): the very best combination
    print("\n══════ ABSOLUTE BEST per (WD, SYM) — across all offset & expiry ══════")
    overall = defaultdict(list)
    for (wd, sym, off, rank), v in best.items():
        overall[(wd, sym)].append((v[0], off, rank, v))
    for (wd, sym), opts in sorted(overall.items()):
        opts.sort(reverse=True)
        cum, off, rank, (_, e, x, n, w, mn, mx, dte) = opts[0]
        strat = "ATM" if off==0 else f"+{off}OTM"
        exptag = "nearWK" if rank==0 else "nextWK"
        print(f"  {wd} {sym:6}  {strat:6} {exptag} dte={dte:2d}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10}  worst={fmt_rs(mn):>9}")

    # Build NEW Phase-3a (one index/day) and Phase-3b (both)
    print("\n══════ NEW PHASE-3a (one best index/day) ══════")
    by_wd = defaultdict(list)
    for (wd, sym, off, rank), v in best.items():
        by_wd[wd].append((v[0], sym, off, rank, v))
    grand = ntot = wtot = 0
    chosen_3a = {}
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        opts = by_wd.get(wd, [])
        if not opts: continue
        opts.sort(reverse=True)
        cum, sym, off, rank, (_, e, x, n, w, mn, mx, dte) = opts[0]
        chosen_3a[wd] = (sym, off, rank, e, x)
        strat = "ATM" if off==0 else f"+{off}OTM"
        exptag = "nearWK" if rank==0 else "nextWK"
        print(f"  {wd}: {sym:6} {strat:6} {exptag} dte={dte:2d}  {e}→{x}  n={n:2d}  win={int(100*w/n):3d}%  cum={fmt_rs(cum):>10}  worst={fmt_rs(mn)}")
        grand += cum; ntot += n; wtot += w
    print(f"  3a TOTAL: {fmt_rs(grand)}  ({ntot} trades, {int(100*wtot/ntot)}% win)  ~{fmt_rs(grand/5)}/mo")

    print("\n══════ NEW PHASE-3b (both NIFTY + SENSEX/day, each at its best) ══════")
    grand = ntot = 0
    chosen_3b = {}
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        line = f"  {wd}: "
        for sym in ["NIFTY","SENSEX"]:
            opts = [(v[0], off, rank, v) for (w2,s,off,rank),v in best.items() if w2==wd and s==sym]
            if not opts: continue
            opts.sort(reverse=True)
            cum, off, rank, (_, e, x, n, w, mn, mx, dte) = opts[0]
            chosen_3b[(wd,sym)] = (off, rank, e, x)
            strat = "ATM" if off==0 else f"+{off}"
            exptag = "near" if rank==0 else "next"
            line += f"{sym}({strat},{exptag},dte{dte},{e}→{x}, {fmt_rs(cum)})  "
            grand += cum; ntot += n
        print(line)
    print(f"  3b TOTAL: {fmt_rs(grand)}  (~{fmt_rs(grand/5)}/mo)")

    # Replay chosen 3a schedule month-by-month
    print("\n══════ Phase-3a MONTHLY (replay best schedule) ══════")
    print(f"{'Month':10} {'Days':>5} {'Win%':>5}  {'PnL':>13}")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for (d, sym, rank), (bars, dte, exp_s) in cache.items():
        wd = d.strftime("%a")
        if wd not in chosen_3a: continue
        s2, off, rk, e, x = chosen_3a[wd]
        if sym != s2 or rank != rk: continue
        p = pnl_for(bars, sym, to_min(e), to_min(x), off)
        if p is None: continue
        m = d.strftime("%Y-%m")
        by_m[m]["pnl"] += p; by_m[m]["n"] += 1
        if p>0: by_m[m]["w"] += 1
    g = gn = gw = 0
    for m in sorted(by_m):
        v = by_m[m]; g += v["pnl"]; gn += v["n"]; gw += v["w"]
        print(f"{m:10} {v['n']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>13}")
    if gn:
        print(f"{'TOTAL':10} {gn:>5} {int(100*gw/gn):>4}%  {fmt_rs(g):>13}")

    # Replay chosen 3b
    print("\n══════ Phase-3b MONTHLY (replay both-index schedule) ══════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for (d, sym, rank), (bars, dte, exp_s) in cache.items():
        wd = d.strftime("%a")
        if (wd,sym) not in chosen_3b: continue
        off, rk, e, x = chosen_3b[(wd,sym)]
        if rank != rk: continue
        p = pnl_for(bars, sym, to_min(e), to_min(x), off)
        if p is None: continue
        m = d.strftime("%Y-%m")
        by_m[m]["pnl"] += p; by_m[m]["n"] += 1
        if p>0: by_m[m]["w"] += 1
    g = gn = gw = 0
    print(f"{'Month':10} {'Trades':>7} {'Win%':>5}  {'PnL':>13}")
    for m in sorted(by_m):
        v = by_m[m]; g += v["pnl"]; gn += v["n"]; gw += v["w"]
        print(f"{m:10} {v['n']:>7} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>13}")
    if gn:
        print(f"{'TOTAL':10} {gn:>7} {int(100*gw/gn):>4}%  {fmt_rs(g):>13}")

if __name__ == "__main__":
    main()
