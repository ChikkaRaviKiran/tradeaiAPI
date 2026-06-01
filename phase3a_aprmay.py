#!/usr/bin/env python3
"""Replay Phase-3a ULTIMATE schedule for April + May 2026, day-by-day."""
from datetime import datetime
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, find_atm, LOT_SIZE, LOTS, STRIKE_STEP)

START = "2026-04-01"
END   = "2026-05-22"

# Phase-3a ULTIMATE schedule per weekday: (sym, offset, entry, exit, expiry_rank)
# offset 0=ATM straddle; N=±N OTM strangle (NIFTY step 50, SENSEX step 100)
# expiry_rank: 0=nearest weekly, 1=next weekly
SCHED = {
    0: ("NIFTY",  3, "09:20", "11:00", 0),  # Mon
    1: ("NIFTY",  0, "09:30", "14:30", 0),  # Tue
    2: ("SENSEX", 0, "09:45", "15:00", 0),  # Wed
    3: ("SENSEX", 0, "10:00", "15:15", 0),  # Thu
    4: ("NIFTY",  1, "10:00", "13:00", 0),  # Fri
}

WD = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


def find_expiries(date, sym):
    """Return sorted (expiry_date, expiry_str) for date+sym, nearest first."""
    rows = psql(f"""SELECT DISTINCT expiry FROM option_candles
                   WHERE date='{date}' AND instrument='{sym}';""")
    out = []
    for (ex_s,) in rows:
        ex = parse_expiry(ex_s)
        if ex is None or ex < date: continue
        out.append((ex, ex_s))
    out.sort()
    return out


def simulate(bars, atm, offset, t_in, t_out, sym):
    step = STRIKE_STEP[sym]
    if offset == 0:
        ce_strike = pe_strike = atm
    else:
        ce_strike = atm + offset * step
        pe_strike = atm - offset * step

    def price(strike, side, t):
        for m in range(t, t+5):
            p = bars.get((strike, side), {}).get(m)
            if p is not None: return p
        return None

    ce_in = price(ce_strike, "CE", t_in)
    pe_in = price(pe_strike, "PE", t_in)
    ce_out = price(ce_strike, "CE", t_out)
    pe_out = price(pe_strike, "PE", t_out)
    if None in (ce_in, pe_in, ce_out, pe_out): return None
    # short straddle/strangle: receive premium at entry, buy back at exit
    pnl_per_unit = (ce_in - ce_out) + (pe_in - pe_out)
    return pnl_per_unit * LOT_SIZE[sym] * LOTS, ce_strike, pe_strike


def main():
    # all trading days in range
    dates = psql(f"""SELECT DISTINCT date FROM option_candles
                   WHERE date >= '{START}' AND date <= '{END}' ORDER BY date;""")
    days = []
    for (ds,) in dates:
        try: d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception: continue
        days.append(d)

    print(f"\n══════ Phase-3a ULTIMATE — DAY-BY-DAY Apr+May 2026 ══════")
    print(f"{'Date':12} {'WD':4} {'Sym':7} {'Strike':18} {'Win':4} {'Entry→Exit':14} {'DTE':>3}  {'PnL':>11}  {'Cum':>11}")
    print("-"*110)

    monthly = {}  # month -> {pnl, n, wins, days_list}
    cum_total = 0
    for d in days:
        wd = d.weekday()
        if wd not in SCHED: continue
        sym, off, e_s, x_s, exprank = SCHED[wd]
        exps = find_expiries(d, sym)
        if len(exps) <= exprank: continue
        ex_date, ex_str = exps[exprank]
        dte = (ex_date - d).days

        bars = load_chain(d, sym, ex_str)
        if not bars: continue
        t_in = to_min(e_s); t_out = to_min(x_s)
        atm = find_atm(bars, t_in, sym)
        if atm is None: continue
        res = simulate(bars, atm, off, t_in, t_out, sym)
        if res is None: continue
        pnl, ce_k, pe_k = res
        cum_total += pnl

        mkey = d.strftime("%Y-%m")
        m = monthly.setdefault(mkey, {"pnl":0, "n":0, "wins":0, "rows":[]})
        m["pnl"] += pnl; m["n"] += 1
        if pnl > 0: m["wins"] += 1
        m["rows"].append((d, wd, sym, off, ce_k, pe_k, e_s, x_s, dte, pnl))

    # print per-month sections
    grand_pnl=0; grand_n=0; grand_w=0
    for mkey in sorted(monthly.keys()):
        m = monthly[mkey]
        print(f"\n── {mkey} ──")
        cum=0
        for (d, wd, sym, off, ce_k, pe_k, e_s, x_s, dte, pnl) in m["rows"]:
            cum += pnl
            if off == 0:
                strike_s = f"ATM {ce_k}"
            else:
                strike_s = f"+{off} {pe_k}/{ce_k}"
            tag = "WIN " if pnl > 0 else "LOSS"
            print(f"{d!s:12} {WD[wd]:4} {sym:7} {strike_s:18} {tag:4} {e_s}→{x_s:6}  {dte:>3}  {fmt_rs(pnl):>11}  {fmt_rs(cum):>11}")
        winpct = int(100*m["wins"]/m["n"]) if m["n"] else 0
        print(f"  {mkey} TOTAL: {fmt_rs(m['pnl'])}  ({m['n']} trades, {winpct}% win, avg {fmt_rs(m['pnl']/max(1,m['n']))}/trade)")
        grand_pnl += m["pnl"]; grand_n += m["n"]; grand_w += m["wins"]

    print(f"\n══════ APR+MAY GRAND TOTAL: {fmt_rs(grand_pnl)}  ({grand_n} trades, {int(100*grand_w/max(1,grand_n))}% win) ══════")
    print(f"  Monthly summary:")
    print(f"  {'Month':10} {'Trades':>6} {'Win%':>5} {'PnL':>12} {'Avg/trade':>12}")
    for mkey in sorted(monthly.keys()):
        m = monthly[mkey]
        winpct = int(100*m["wins"]/m["n"]) if m["n"] else 0
        avg = m["pnl"]/m["n"] if m["n"] else 0
        print(f"  {mkey:10} {m['n']:>6} {winpct:>4}% {fmt_rs(m['pnl']):>12} {fmt_rs(avg):>12}")

if __name__ == "__main__":
    main()
