#!/usr/bin/env python3
"""Compare ATM straddle vs OTM strangles (+1, +2, +3 strikes) on same schedule."""
from collections import defaultdict
from datetime import datetime
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   find_trading_days, load_chain,
                                   LOT_SIZE, LOTS, STRIKE_STEP)

SCHEDULE = {
    "Mon": ("NIFTY",  "09:20", "15:15"),
    "Tue": ("NIFTY",  "09:30", "13:30"),
    "Wed": ("SENSEX", "10:00", "15:15"),
    "Thu": ("SENSEX", "09:20", "11:30"),
    "Fri": ("NIFTY",  "09:45", "14:30"),
}

def find_atm(bars, t_min, sym):
    """PCP-based ATM strike at t_min."""
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

def simulate_strangle(bars, sym, entry, exit_t, offset_steps):
    """offset_steps=0 => ATM straddle. offset_steps=N => sell CE@ATM+N*step, PE@ATM-N*step."""
    e_min = to_min(entry); x_min = to_min(exit_t)
    atm = find_atm(bars, e_min, sym)
    if atm is None: return None
    step = STRIKE_STEP[sym]
    ce_strike = atm + offset_steps*step
    pe_strike = atm - offset_steps*step
    ce_e = price_at(bars, ce_strike, "CE", e_min)
    pe_e = price_at(bars, pe_strike, "PE", e_min)
    ce_x = price_at(bars, ce_strike, "CE", x_min)
    pe_x = price_at(bars, pe_strike, "PE", x_min)
    if None in (ce_e, pe_e, ce_x, pe_x): return None
    credit = ce_e + pe_e
    debit  = ce_x + pe_x
    return (credit - debit) * LOT_SIZE[sym] * LOTS

def main():
    near = find_trading_days()
    print(f"Loading bars for all schedule days...")
    days = []
    for (d, sym), (ex, exp_s) in sorted(near.items()):
        wd = d.strftime("%a")
        if wd not in SCHEDULE: continue
        plan_sym, entry, exit_t = SCHEDULE[wd]
        if sym != plan_sym: continue
        bars = load_chain(d, sym, exp_s)
        if bars: days.append((d, wd, sym, entry, exit_t, bars))
    print(f"Loaded {len(days)} days\n")

    strategies = {
        "ATM straddle  (0)": 0,
        "Strangle +1   ":    1,
        "Strangle +2   ":    2,
        "Strangle +3   ":    3,
        "Strangle +4   ":    4,
        "Strangle +5   ":    5,
    }

    print(f"{'Strategy':22} {'Trades':>7} {'Wins':>5} {'Win%':>5}  {'PnL':>13}  {'Avg/day':>10}  {'Worst':>10}  {'Best':>10}")
    print("─" * 105)

    per_strategy = {}
    for label, off in strategies.items():
        trades = []
        for d, wd, sym, e, x, bars in days:
            pnl = simulate_strangle(bars, sym, e, x, off)
            if pnl is not None:
                trades.append((d, wd, sym, pnl))
        if not trades: continue
        n = len(trades); w = sum(1 for _,_,_,p in trades if p>0)
        pnls = [p for _,_,_,p in trades]
        cum = sum(pnls)
        per_strategy[label] = trades
        print(f"{label:22} {n:>7} {w:>5} {int(100*w/n):>4}%  {fmt_rs(cum):>13}  {fmt_rs(cum/n):>10}  {fmt_rs(min(pnls)):>10}  {fmt_rs(max(pnls)):>10}")

    # Monthly breakdown for each
    print("\n══════ MONTHLY PnL by strategy ══════")
    months = sorted({t[0].strftime("%Y-%m") for trades in per_strategy.values() for t in trades})
    header = f"{'Strategy':22}" + "".join(f"{m[5:]:>10}" for m in months) + f"{'TOTAL':>13}"
    print(header)
    print("─" * len(header))
    for label, trades in per_strategy.items():
        by_m = defaultdict(float)
        for d, wd, sym, p in trades:
            by_m[d.strftime("%Y-%m")] += p
        cells = "".join(f"{fmt_rs(by_m[m]):>10}" for m in months)
        total = sum(by_m.values())
        print(f"{label:22}{cells}{fmt_rs(total):>13}")

    # Win-rate breakdown
    print("\n══════ Win rate by strategy ══════")
    for label, trades in per_strategy.items():
        by_m = defaultdict(lambda: [0,0])
        for d, wd, sym, p in trades:
            m = d.strftime("%Y-%m"); by_m[m][0] += 1
            if p>0: by_m[m][1] += 1
        cells = "".join(f"{int(100*by_m[m][1]/by_m[m][0]):>4}%   " if by_m[m][0] else "  --   " for m in months)
        print(f"{label:22}{cells}")

if __name__ == "__main__":
    main()
