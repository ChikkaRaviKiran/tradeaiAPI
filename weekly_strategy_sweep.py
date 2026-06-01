#!/usr/bin/env python3
"""Weekly-adaptive strategy comparison.

Compares two ideas against the fixed Phase-3a baseline:

  (A) ORACLE — best-strategy-per-week chosen in hindsight (upper bound).
      For each ISO week, evaluate N candidate weekly schedules; pick the
      week's winner. Sum across all weeks.

  (B) SIGNAL — pick the next week's schedule using a signal computable
      BEFORE the week starts. Two signals tested:
        S1 = Monday 09:30 ATM straddle premium (vol proxy):
             - high vol  -> tighter strikes (closer to ATM) catch premium decay
             - low vol   -> wider strangle still pays, fewer touches
        S2 = Previous week's PnL leader (momentum):
             - use whatever schedule won last week
"""
from collections import defaultdict
from datetime import datetime, date, timedelta
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, find_atm, LOT_SIZE, LOTS, STRIKE_STEP)

START = "2026-01-01"
END   = "2026-05-22"

# Phase-3a baseline (per weekday: sym, offset, entry, exit, exprank)
SCHED_3A = {
    0: ("NIFTY",  3, "09:20", "11:00", 0),
    1: ("NIFTY",  0, "09:30", "14:30", 0),
    2: ("SENSEX", 0, "09:45", "15:00", 0),
    3: ("SENSEX", 0, "10:00", "15:15", 0),
    4: ("NIFTY",  1, "10:00", "13:00", 0),
}

# Candidate WEEKLY schedules. Each is a full 5-day schedule (Mon-Fri).
# We vary the OFFSET only and keep entry/exit/sym/exprank from 3a — so
# every comparison is apples-to-apples (same windows, same indexes,
# only the strike offset rotates per week).
CANDIDATES = {
    "3a (fixed)": SCHED_3A,                                       # what we ship today
    "ALL ATM":    {wd: (s[0], 0, s[2], s[3], s[4]) for wd, s in SCHED_3A.items()},
    "ALL +1":     {wd: (s[0], 1, s[2], s[3], s[4]) for wd, s in SCHED_3A.items()},
    "ALL +2":     {wd: (s[0], 2, s[2], s[3], s[4]) for wd, s in SCHED_3A.items()},
    "ALL +3":     {wd: (s[0], 3, s[2], s[3], s[4]) for wd, s in SCHED_3A.items()},
}

WD_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


# ─────────────────────── data helpers ───────────────────────

def find_expiries(d, sym):
    rows = psql(f"""SELECT DISTINCT expiry FROM option_candles
                   WHERE date='{d}' AND instrument='{sym}';""")
    out = []
    for (ex_s,) in rows:
        ex = parse_expiry(ex_s)
        if ex is None or ex < d: continue
        out.append((ex, ex_s))
    out.sort()
    return out


def simulate(bars, atm, offset, t_in, t_out, sym):
    step = STRIKE_STEP[sym]
    ce_k = atm + offset * step
    pe_k = atm - offset * step
    def price(k, side, t):
        for m in range(t, t+5):
            p = bars.get((k, side), {}).get(m)
            if p is not None: return p
        return None
    ce_in = price(ce_k, "CE", t_in)
    pe_in = price(pe_k, "PE", t_in)
    ce_out = price(ce_k, "CE", t_out)
    pe_out = price(pe_k, "PE", t_out)
    if None in (ce_in, pe_in, ce_out, pe_out): return None
    return ((ce_in - ce_out) + (pe_in - pe_out)) * LOT_SIZE[sym] * LOTS


def day_pnl(d, sym, offset, e_s, x_s, exprank, _cache):
    """Cached per-day PnL for a (sym, offset, e, x, exprank) combo."""
    key = (d, sym, offset, e_s, x_s, exprank)
    if key in _cache: return _cache[key]
    exps = find_expiries(d, sym)
    if len(exps) <= exprank:
        _cache[key] = None; return None
    ex_date, ex_str = exps[exprank]
    bars = load_chain(d, sym, ex_str)
    if not bars:
        _cache[key] = None; return None
    t_in = to_min(e_s); t_out = to_min(x_s)
    atm = find_atm(bars, t_in, sym)
    if atm is None:
        _cache[key] = None; return None
    res = simulate(bars, atm, offset, t_in, t_out, sym)
    _cache[key] = res
    return res


def schedule_week_pnl(week_days, sched, cache):
    """Sum daily PnL for the schedule on every day in a week."""
    pnl = 0.0; n = 0; wins = 0
    for d in week_days:
        wd = d.weekday()
        if wd not in sched: continue
        sym, off, e, x, rk = sched[wd]
        p = day_pnl(d, sym, off, e, x, rk, cache)
        if p is None: continue
        pnl += p; n += 1
        if p > 0: wins += 1
    return pnl, n, wins


def signal_monday_atm_premium(monday, cache_other):
    """S1 vol proxy = NIFTY Mon 09:30 ATM CE+PE premium (per-unit)."""
    exps = find_expiries(monday, "NIFTY")
    if not exps: return None
    _, ex_str = exps[0]
    bars = load_chain(monday, "NIFTY", ex_str)
    if not bars: return None
    t = to_min("09:30")
    atm = find_atm(bars, t, "NIFTY")
    if atm is None: return None
    ce = bars.get((atm, "CE"), {})
    pe = bars.get((atm, "PE"), {})
    ce_p = pe_p = None
    for m in range(t, t+5):
        if ce_p is None: ce_p = ce.get(m)
        if pe_p is None: pe_p = pe.get(m)
    if ce_p is None or pe_p is None: return None
    return ce_p + pe_p


# ─────────────────────── main ───────────────────────

def main():
    # All trading days
    rows = psql(f"""SELECT DISTINCT date FROM option_candles
                   WHERE date >= '{START}' AND date <= '{END}' ORDER BY date;""")
    days = []
    for (ds,) in rows:
        try: d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception: continue
        days.append(d)

    # Group by ISO (year, week)
    weeks = defaultdict(list)
    for d in days:
        y, w, _ = d.isocalendar()
        weeks[(y, w)].append(d)
    week_keys = sorted(weeks.keys())

    cache = {}

    # Pre-compute per-week PnL for every candidate
    week_results = {}  # (y,w) -> {cand_name: (pnl, n, wins)}
    for k in week_keys:
        week_results[k] = {}
        for name, sched in CANDIDATES.items():
            week_results[k][name] = schedule_week_pnl(weeks[k], sched, cache)

    # ─── Print weekly table ───
    print("\n══════ Weekly PnL by candidate schedule ══════")
    hdr = f"{'Week':12} " + " ".join(f"{n:>11}" for n in CANDIDATES.keys()) + f"  {'BEST':>11}  vs 3a"
    print(hdr); print("-"*len(hdr))
    totals = {n: 0.0 for n in CANDIDATES.keys()}
    oracle_total = 0.0
    win_count_per_candidate = defaultdict(int)
    for k in week_keys:
        wk_label = f"{k[0]}-W{k[1]:02d}"
        cells = []
        best_name = None; best_pnl = -1e18
        for n in CANDIDATES.keys():
            p, _, _ = week_results[k][n]
            cells.append(f"{fmt_rs(p):>11}")
            totals[n] += p
            if p > best_pnl: best_pnl = p; best_name = n
        oracle_total += best_pnl
        win_count_per_candidate[best_name] += 1
        diff = best_pnl - week_results[k]["3a (fixed)"][0]
        print(f"{wk_label:12} " + " ".join(cells) + f"  {fmt_rs(best_pnl):>11} ({best_name})  d={fmt_rs(diff)}")

    print("-"*len(hdr))
    tot_cells = " ".join(f"{fmt_rs(totals[n]):>11}" for n in CANDIDATES.keys())
    print(f"{'TOTAL':12} {tot_cells}  {fmt_rs(oracle_total):>11} (ORACLE)")

    # ─── Oracle stats ───
    base = totals["3a (fixed)"]
    print(f"\n══════ ORACLE upper bound (in-hindsight weekly pick) ══════")
    print(f"  ORACLE total : {fmt_rs(oracle_total)}")
    print(f"  3a fixed     : {fmt_rs(base)}")
    print(f"  Delta        : {fmt_rs(oracle_total - base)}  (+{100*(oracle_total-base)/abs(base):.1f}%)")
    print(f"  Weeks where each candidate WON:")
    for n in CANDIDATES.keys():
        print(f"    {n:12} : {win_count_per_candidate[n]:>3} weeks")

    # ─── Signal-driven strategies ───
    # S1: pick by Monday 09:30 ATM premium (proxy for IV).
    # Bucket weeks into terciles by premium; assign best-historical-tercile-winner.
    print(f"\n══════ SIGNAL S1: Monday 09:30 NIFTY ATM premium tercile ══════")
    monday_prem = {}
    for k in week_keys:
        mon = weeks[k][0]
        if mon.weekday() != 0: continue   # only ISO weeks starting Monday
        p = signal_monday_atm_premium(mon, cache)
        if p is not None: monday_prem[k] = p
    if len(monday_prem) >= 6:
        sorted_p = sorted(monday_prem.values())
        n3 = len(sorted_p) // 3
        lo_thr = sorted_p[n3]; hi_thr = sorted_p[2*n3]
        # Walk-forward: for each week with a signal, pick the candidate that
        # had highest mean PnL in the same tercile across PRIOR weeks.
        tercile_stats = {"LO": defaultdict(list), "MID": defaultdict(list), "HI": defaultdict(list)}
        s1_total = 0.0; s1_decisions = []
        for k in week_keys:
            if k not in monday_prem: continue
            prem = monday_prem[k]
            terc = "LO" if prem <= lo_thr else "HI" if prem >= hi_thr else "MID"
            # Pick using only PRIOR data
            best_for_terc = None; best_avg = -1e18
            for n, pnls in tercile_stats[terc].items():
                if len(pnls) >= 2:
                    avg = sum(pnls) / len(pnls)
                    if avg > best_avg: best_avg = avg; best_for_terc = n
            chosen = best_for_terc or "3a (fixed)"   # fallback until enough history
            chosen_pnl = week_results[k][chosen][0]
            s1_total += chosen_pnl
            s1_decisions.append((k, prem, terc, chosen, chosen_pnl, week_results[k]["3a (fixed)"][0]))
            # Update tercile stats with TRUE winner of this week (oracle bookkeeping)
            for n in CANDIDATES.keys():
                tercile_stats[terc][n].append(week_results[k][n][0])
        print(f"  premium terciles (CE+PE): LO<={lo_thr:.1f}  HI>={hi_thr:.1f}")
        for (k, prem, terc, chosen, pnl, base_p) in s1_decisions:
            d = pnl - base_p
            print(f"  {k[0]}-W{k[1]:02d}  prem={prem:6.1f} {terc:3}  pick={chosen:12}  pnl={fmt_rs(pnl):>10}  vs3a {fmt_rs(d):>10}")
        print(f"  S1 TOTAL: {fmt_rs(s1_total)}   vs 3a {fmt_rs(s1_total - base)}")
    else:
        print("  (not enough Mondays in sample)")

    # S2: pick last week's winner (momentum)
    print(f"\n══════ SIGNAL S2: pick last week's winning candidate (momentum) ══════")
    s2_total = 0.0
    prev_winner = "3a (fixed)"
    for k in week_keys:
        pnl_chosen = week_results[k][prev_winner][0]
        s2_total += pnl_chosen
        # update for next week
        best_n = None; best_v = -1e18
        for n, (p, _, _) in week_results[k].items():
            if p > best_v: best_v = p; best_n = n
        prev_winner = best_n
    print(f"  S2 TOTAL: {fmt_rs(s2_total)}   vs 3a {fmt_rs(s2_total - base)}")

    # ─── Final comparison ───
    print(f"\n══════ FINAL COMPARISON ══════")
    print(f"  3a fixed      : {fmt_rs(base)}")
    for n in CANDIDATES.keys():
        if n == "3a (fixed)": continue
        print(f"  {n:14}: {fmt_rs(totals[n])}  ({'+' if totals[n]>=base else ''}{fmt_rs(totals[n]-base)} vs 3a)")
    print(f"  S1 signal     : {fmt_rs(s1_total) if 's1_total' in dir() and len(monday_prem)>=6 else 'n/a'}")
    print(f"  S2 signal     : {fmt_rs(s2_total)}  ({'+' if s2_total>=base else ''}{fmt_rs(s2_total-base)} vs 3a)")
    print(f"  ORACLE max    : {fmt_rs(oracle_total)}  (+{fmt_rs(oracle_total-base)} vs 3a — upper bound only)")


if __name__ == "__main__":
    main()
