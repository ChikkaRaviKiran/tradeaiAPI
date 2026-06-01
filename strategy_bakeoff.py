#!/usr/bin/env python3
"""Strategy bake-off: test 15+ schedule variants vs Phase-3a baseline.

Each variant is a `decide(date, weekday, history) -> trade_spec | None`.
We iterate all trading days once, ask every variant for a trade (or skip),
look up the (cached) PnL, accumulate per variant. Final ranked table.
"""
from collections import defaultdict
from datetime import datetime, date
from phase3a_breakdown_v2 import (psql, parse_expiry, to_min, fmt_rs,
                                   load_chain, find_atm, LOT_SIZE, LOTS, STRIKE_STEP)

START = "2026-01-01"
END   = "2026-05-22"

# Baseline 3a: (sym, offset, entry, exit, exprank)
B = {
    0: ("NIFTY",  3, "09:20", "11:00", 0),  # Mon
    1: ("NIFTY",  0, "09:30", "14:30", 0),  # Tue
    2: ("SENSEX", 0, "09:45", "15:00", 0),  # Wed
    3: ("SENSEX", 0, "10:00", "15:15", 0),  # Thu
    4: ("NIFTY",  1, "10:00", "13:00", 0),  # Fri
}
WD = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
_pnl_cache = {}
_exp_cache = {}


def find_expiries(d, sym):
    key = (d, sym)
    if key in _exp_cache: return _exp_cache[key]
    rows = psql(f"""SELECT DISTINCT expiry FROM option_candles
                   WHERE date='{d}' AND instrument='{sym}';""")
    out = []
    for (ex_s,) in rows:
        ex = parse_expiry(ex_s)
        if ex is None or ex < d: continue
        out.append((ex, ex_s))
    out.sort()
    _exp_cache[key] = out
    return out


def day_pnl(d, spec):
    """spec=(sym, offset, e_s, x_s, exprank). Returns PnL or None."""
    sym, off, e_s, x_s, rk = spec
    key = (d, sym, off, e_s, x_s, rk)
    if key in _pnl_cache: return _pnl_cache[key]
    exps = find_expiries(d, sym)
    if len(exps) <= rk:
        _pnl_cache[key] = None; return None
    _, ex_str = exps[rk]
    bars = load_chain(d, sym, ex_str)
    if not bars:
        _pnl_cache[key] = None; return None
    t_in = to_min(e_s); t_out = to_min(x_s)
    atm = find_atm(bars, t_in, sym)
    if atm is None:
        _pnl_cache[key] = None; return None
    step = STRIKE_STEP[sym]
    ce_k = atm + off*step; pe_k = atm - off*step
    def price(k, side, t):
        for m in range(t, t+5):
            p = bars.get((k, side), {}).get(m)
            if p is not None: return p
        return None
    ce_in = price(ce_k,"CE",t_in); pe_in = price(pe_k,"PE",t_in)
    ce_out = price(ce_k,"CE",t_out); pe_out = price(pe_k,"PE",t_out)
    if None in (ce_in,pe_in,ce_out,pe_out):
        _pnl_cache[key] = None; return None
    pnl = ((ce_in-ce_out) + (pe_in-pe_out)) * LOT_SIZE[sym] * LOTS
    _pnl_cache[key] = pnl
    return pnl


def is_last_week_of_month(d, all_days):
    # last week = no later trading day in same month
    return not any(x.month == d.month and x > d for x in all_days)


def is_first_week_of_month(d, all_days):
    return not any(x.month == d.month and x < d for x in all_days)


def is_expiry_week(d, sym):
    """Sym has its weekly expiry this week."""
    exps = find_expiries(d, sym)
    if not exps: return False
    ex = exps[0][0]
    # Same ISO week?
    return d.isocalendar()[:2] == ex.isocalendar()[:2]


# ──────────────── VARIANTS ────────────────
# Each: decide(d, wd, history, all_days) -> list of specs to trade (may be 0..N)
# history = list of (date, variant_pnl) for this variant so far

def v_3a(d, wd, h, ad):
    return [B[wd]] if wd in B else []

def v_swap_idx(d, wd, h, ad):
    """Swap NIFTY↔SENSEX where data exists, keep same offset/window."""
    if wd not in B: return []
    sym, off, e, x, rk = B[wd]
    new = "SENSEX" if sym == "NIFTY" else "NIFTY"
    return [(new, off, e, x, rk)]

def v_drop_thu(d, wd, h, ad):
    if wd == 3: return []
    return v_3a(d, wd, h, ad)

def v_drop_mon(d, wd, h, ad):
    if wd == 0: return []
    return v_3a(d, wd, h, ad)

def v_tue_only(d, wd, h, ad):
    if wd != 1: return []
    return [B[1]]

def v_tue_wed(d, wd, h, ad):
    if wd in (1,2): return [B[wd]]
    return []

def v_tue_double(d, wd, h, ad):
    """Tue: trade both NIFTY ATM AND SENSEX ATM. Other days: 3a."""
    if wd == 1:
        return [B[1], ("SENSEX", 0, "09:30", "15:00", 0)]
    return v_3a(d, wd, h, ad)

def v_all_atm(d, wd, h, ad):
    if wd not in B: return []
    sym, _, e, x, rk = B[wd]
    return [(sym, 0, e, x, rk)]

def v_all_p1(d, wd, h, ad):
    if wd not in B: return []
    sym, _, e, x, rk = B[wd]
    return [(sym, 1, e, x, rk)]

def v_expiry_tight(d, wd, h, ad):
    """In a week where the index used today has its expiry, force ATM.
    Otherwise use +1 OTM."""
    if wd not in B: return []
    sym, _, e, x, rk = B[wd]
    off = 0 if is_expiry_week(d, sym) else 1
    return [(sym, off, e, x, rk)]

def v_lastwk_wider(d, wd, h, ad):
    """Last week of month → +2 OTM defensive; else 3a."""
    if wd not in B: return []
    if is_last_week_of_month(d, ad):
        sym, _, e, x, rk = B[wd]
        return [(sym, 2, e, x, rk)]
    return [B[wd]]

def v_firstwk_atm(d, wd, h, ad):
    """First week of month → ATM; else 3a."""
    if wd not in B: return []
    if is_first_week_of_month(d, ad):
        sym, _, e, x, rk = B[wd]
        return [(sym, 0, e, x, rk)]
    return [B[wd]]

def v_mon_loss_widen(d, wd, h, ad):
    """If Monday this week was a loss, widen Tue-Fri to +3."""
    if wd not in B: return []
    # find Monday's pnl this ISO week
    mon_pnl = None
    iso = d.isocalendar()[:2]
    for hd, hp in h:
        if hd.isocalendar()[:2] == iso and hd.weekday() == 0:
            mon_pnl = hp; break
    if wd > 0 and mon_pnl is not None and mon_pnl < 0:
        sym, _, e, x, rk = B[wd]
        return [(sym, 3, e, x, rk)]
    return [B[wd]]

def v_mon_loss_skip_week(d, wd, h, ad):
    """If Mon lost, skip rest of week."""
    if wd not in B: return []
    iso = d.isocalendar()[:2]
    if wd == 0: return [B[0]]
    for hd, hp in h:
        if hd.isocalendar()[:2] == iso and hd.weekday() == 0 and hp < 0:
            return []
    return [B[wd]]

def v_wtd_stop(d, wd, h, ad):
    """Stop trading rest of week if week-to-date < -10000."""
    if wd not in B: return []
    iso = d.isocalendar()[:2]
    wtd = sum(hp for hd,hp in h if hd.isocalendar()[:2] == iso)
    if wtd < -10000: return []
    return [B[wd]]

def v_skip_last3(d, wd, h, ad):
    """Skip the last 3 trading days of every month (event risk)."""
    if wd not in B: return []
    later = sum(1 for x in ad if x.month == d.month and x > d)
    if later < 3: return []
    return [B[wd]]

def v_hybrid_best(d, wd, h, ad):
    """Hybrid: keep 3a for high-conviction days (Tue/Wed), use +1 on
    historically weaker days (Mon/Thu/Fri)."""
    if wd not in B: return []
    sym, off, e, x, rk = B[wd]
    if wd in (1, 2):                  # Tue, Wed — keep 3a
        return [(sym, off, e, x, rk)]
    return [(sym, max(1, off), e, x, rk)]   # widen Mon/Thu/Fri to at least +1

def v_phase3b_tue(d, wd, h, ad):
    """3a + on Tue also trade SENSEX (Phase-3b style for Tue only)."""
    if wd not in B: return []
    base = [B[wd]]
    if wd == 1:
        base.append(("SENSEX", 0, "09:30", "15:00", 0))
    return base


VARIANTS = [
    ("3a baseline",         v_3a),
    ("swap NIFTY<->SENSEX", v_swap_idx),
    ("drop Thu",            v_drop_thu),
    ("drop Mon",            v_drop_mon),
    ("Tue only",            v_tue_only),
    ("Tue+Wed only",        v_tue_wed),
    ("Tue DOUBLE (N+S)",    v_tue_double),
    ("ALL ATM",             v_all_atm),
    ("ALL +1",              v_all_p1),
    ("expiry-wk -> ATM",    v_expiry_tight),
    ("last-wk -> +2 wider", v_lastwk_wider),
    ("first-wk -> ATM",     v_firstwk_atm),
    ("Mon loss -> widen +3",v_mon_loss_widen),
    ("Mon loss -> SKIP wk", v_mon_loss_skip_week),
    ("WTD stop -10K",       v_wtd_stop),
    ("skip last-3 of mo",   v_skip_last3),
    ("hybrid: +1 weak days",v_hybrid_best),
    ("3a + Tue add SENSEX", v_phase3b_tue),
]


def main():
    rows = psql(f"""SELECT DISTINCT date FROM option_candles
                   WHERE date >= '{START}' AND date <= '{END}' ORDER BY date;""")
    days = []
    for (ds,) in rows:
        try: d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception: continue
        days.append(d)

    print(f"Days in sample: {len(days)}  ({days[0]} → {days[-1]})\n")
    print("Running variants (sequential, with per-spec PnL cache)...")

    results = {}  # name -> list of (date, pnl)
    for name, fn in VARIANTS:
        hist = []
        for d in days:
            wd = d.weekday()
            specs = fn(d, wd, hist, days)
            day_total = 0.0; got = False
            for sp in specs:
                p = day_pnl(d, sp)
                if p is not None:
                    day_total += p; got = True
            if got:
                hist.append((d, day_total))
        results[name] = hist

    # ── stats per variant ──
    base_total = sum(p for _, p in results["3a baseline"])
    base_wins  = sum(1 for _, p in results["3a baseline"] if p > 0)
    base_n     = len(results["3a baseline"])

    print(f"\n══════ Baseline 3a: {fmt_rs(base_total)} / {base_n} trade-days / {int(100*base_wins/base_n)}% win ══════\n")

    stats = []
    for name, _ in VARIANTS:
        h = results[name]
        n = len(h)
        if n == 0:
            stats.append((name, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)); continue
        tot = sum(p for _,p in h)
        wins = sum(1 for _,p in h if p > 0)
        worst = min(p for _,p in h)
        best  = max(p for _,p in h)
        avg = tot/n
        win_pct = 100*wins/n
        stats.append((name, n, wins, win_pct, tot, avg, worst, best))

    # Sort by total desc
    stats.sort(key=lambda r: -r[4])

    print(f"{'Variant':<26}{'Trades':>7}{'Win%':>6}{'Total':>14}{'Avg/day':>11}{'Worst':>11}{'Best':>11}{'vs 3a':>13}")
    print("-"*99)
    for name, n, wins, wp, tot, avg, worst, best in stats:
        diff = tot - base_total
        marker = " ★" if tot > base_total else ""
        print(f"{name:<26}{n:>7}{int(wp):>5}%{fmt_rs(tot):>14}{fmt_rs(avg):>11}{fmt_rs(worst):>11}{fmt_rs(best):>11}{fmt_rs(diff):>13}{marker}")

    # ── monthly breakdown of TOP 5 ──
    print("\n══════ MONTHLY breakdown — top 5 variants ══════")
    top5 = [s[0] for s in stats[:5]]
    months = sorted({d.strftime("%Y-%m") for d in days})
    print(f"{'Variant':<26}" + "".join(f"{m:>14}" for m in months) + f"{'TOTAL':>14}")
    print("-"*(26 + 14*(len(months)+1)))
    for name in top5:
        h = results[name]
        by_mo = defaultdict(float)
        for d, p in h: by_mo[d.strftime("%Y-%m")] += p
        cells = "".join(f"{fmt_rs(by_mo.get(m,0)):>14}" for m in months)
        tot = sum(by_mo.values())
        print(f"{name:<26}{cells}{fmt_rs(tot):>14}")

    # Min-monthly check (every month positive?)
    print(f"\n══════ EVERY MONTH POSITIVE? (top 5) ══════")
    for name in top5:
        h = results[name]
        by_mo = defaultdict(float)
        for d, p in h: by_mo[d.strftime("%Y-%m")] += p
        mins = min(by_mo.values()) if by_mo else 0
        bad = [m for m,v in by_mo.items() if v < 0]
        status = "✓ ALL POSITIVE" if not bad else f"✗ negative in {','.join(bad)}"
        print(f"  {name:<26} min-mo={fmt_rs(mins):>12}   {status}")


if __name__ == "__main__":
    main()
