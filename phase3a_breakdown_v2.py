#!/usr/bin/env python3
"""Phase-3a daily PnL breakdown — handles post-Apr regime (NULL spot, numeric expiry)."""
import subprocess
from collections import defaultdict
from datetime import datetime, date

LOTS = 3
LOT_SIZE = {"NIFTY": 75, "SENSEX": 20}
STRIKE_STEP = {"NIFTY": 50, "SENSEX": 100}

SCHEDULE = {
    "Mon": ("NIFTY",  "09:20", "14:00"),
    "Tue": ("NIFTY",  "09:20", "15:15"),
    "Wed": ("SENSEX", "09:20", "15:00"),
    "Thu": ("SENSEX", "09:20", "12:30"),
    "Fri": ("SENSEX", "09:20", "12:30"),
}

def psql(sql):
    cmd = ["docker","exec","-i","tradeai-postgres","psql","-U","tradeai","-d","tradeai","-t","-A","-F","\t","-c",sql]
    return [line.split("\t") for line in subprocess.check_output(cmd, text=True).strip().split("\n") if line.strip()]

def parse_expiry(s):
    try: return datetime.strptime(s, "%d%b%y").date()
    except Exception: pass
    # Numeric YYMDD (or YYMMDD): 26423 -> 2026-04-23; 26514 -> 2026-05-14
    if s.isdigit() and 5 <= len(s) <= 6:
        try:
            day = int(s[-2:])
            ym  = s[:-2]
            yy  = int(ym[:2])
            mm  = int(ym[2:])
            return date(2000+yy, mm, day)
        except Exception: return None
    return None

def to_min(t):
    h,m=t.split(":"); return int(h)*60+int(m)

def fmt_rs(x):
    return f"₹{'+' if x>=0 else '-'}{abs(int(x)):,}"

def find_trading_days():
    rows = psql("""SELECT DISTINCT date, instrument, expiry FROM option_candles
                   WHERE date >= '2026-01-01' AND date <= '2026-05-22' ORDER BY date, instrument;""")
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
        out[k] = lst[0]
    return out

def load_chain(d, sym, exp_s):
    """Load ALL strikes for the day; return {(strike,opt): {minute: close}}."""
    rows = psql(f"""SELECT strike, option_type, timestamp, close FROM option_candles
                    WHERE instrument='{sym}' AND date='{d}' AND expiry='{exp_s}'
                      AND close IS NOT NULL ORDER BY timestamp;""")
    bars = defaultdict(dict)
    for strike, opt, ts, close in rows:
        try:
            dt = datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
            m = dt.hour*60+dt.minute
            bars[(int(strike), opt)][m] = float(close)
        except Exception: continue
    return bars

def find_atm(bars, t_min, sym):
    """ATM = strike where |CE - PE| is minimum near t_min (put-call parity)."""
    step = STRIKE_STEP[sym]
    # collect strikes with both CE and PE at t_min (allow up to +4 min slack)
    strikes = sorted({k[0] for k in bars.keys()})
    best = None; best_diff = None
    for s in strikes:
        ce = pe = None
        for m in range(t_min, t_min+5):
            if ce is None: ce = bars.get((s,"CE"),{}).get(m)
            if pe is None: pe = bars.get((s,"PE"),{}).get(m)
            if ce is not None and pe is not None: break
        if ce is None or pe is None: continue
        diff = abs(ce - pe)
        if best is None or diff < best_diff:
            best_diff = diff; best = s
    if best is None: return None
    # snap to step grid
    return int(round(best/step)*step) if best % step == 0 else best

def straddle(bars, atm, t_min):
    ce = pe = None
    for m in range(t_min, t_min+5):
        if ce is None: ce = bars.get((atm,"CE"),{}).get(m)
        if pe is None: pe = bars.get((atm,"PE"),{}).get(m)
        if ce is not None and pe is not None: break
    if ce is None or pe is None: return None
    return ce + pe

def simulate(bars, sym, entry, exit_t):
    e_min, x_min = to_min(entry), to_min(exit_t)
    atm = find_atm(bars, e_min, sym)
    if atm is None: return None, None
    c = straddle(bars, atm, e_min)
    d = straddle(bars, atm, x_min)
    if c is None or d is None: return None, atm
    return (c - d) * LOT_SIZE[sym] * LOTS, atm

def main():
    near = find_trading_days()
    print(f"Total (date, sym) combos: {len(near)}")
    trades = []
    skipped = []
    for (d, sym), (ex, exp_s) in sorted(near.items()):
        wd = d.strftime("%a")
        if wd not in SCHEDULE: continue
        plan_sym, entry, exit_t = SCHEDULE[wd]
        if sym != plan_sym: continue
        actual_dte = (ex - d).days
        bars = load_chain(d, sym, exp_s)
        if not bars:
            skipped.append((d, sym, "no-bars")); continue
        pnl, atm = simulate(bars, sym, entry, exit_t)
        if pnl is None:
            skipped.append((d, sym, f"no-pnl atm={atm}")); continue
        trades.append((d, wd, sym, actual_dte, entry, exit_t, atm, pnl))

    print(f"Matched trades: {len(trades)}   Skipped: {len(skipped)}")
    if skipped:
        for s in skipped: print(f"  SKIP {s}")

    print("\n══════ PER-DAY PnL ══════")
    print(f"{'Date':12} {'WD':4} {'Sym':7} {'DTE':>3} {'ATM':>7} {'Window':14}  {'PnL':>12}  Cum")
    cum = 0
    for d, wd, sym, dte, e, x, atm, pnl in trades:
        cum += pnl
        print(f"{str(d):12} {wd:4} {sym:7} {dte:>3} {atm:>7} {e}→{x}  {fmt_rs(pnl):>12}  {fmt_rs(cum)}")

    print("\n══════ MONTHLY TOTALS ══════")
    by_m = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for d, wd, sym, dte, e, x, atm, pnl in trades:
        key = d.strftime("%Y-%m")
        by_m[key]["pnl"] += pnl; by_m[key]["n"] += 1
        if pnl>0: by_m[key]["w"] += 1
    print(f"{'Month':10} {'Days':>5} {'Wins':>5} {'Win%':>5}  {'PnL':>13}  {'Avg/day':>10}")
    g_pnl=g_n=g_w=0
    for m in sorted(by_m):
        v = by_m[m]; g_pnl += v["pnl"]; g_n += v["n"]; g_w += v["w"]
        print(f"{m:10} {v['n']:>5} {v['w']:>5} {int(100*v['w']/v['n']):>4}%  {fmt_rs(v['pnl']):>13}  {fmt_rs(v['pnl']/v['n']):>10}")
    print(f"{'TOTAL':10} {g_n:>5} {g_w:>5} {int(100*g_w/g_n):>4}%  {fmt_rs(g_pnl):>13}  {fmt_rs(g_pnl/g_n):>10}")

    print("\n══════ BY WEEKDAY ══════")
    by_wd = defaultdict(lambda: {"pnl":0,"n":0,"w":0})
    for d, wd, sym, dte, e, x, atm, pnl in trades:
        by_wd[wd]["pnl"] += pnl; by_wd[wd]["n"] += 1
        if pnl>0: by_wd[wd]["w"] += 1
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        v = by_wd[wd]
        if v["n"]==0: continue
        sym, e, x = SCHEDULE[wd]
        print(f"  {wd} {sym:6} {e}→{x}  n={v['n']:2d}  win={int(100*v['w']/v['n']):3d}%  cum={fmt_rs(v['pnl'])}  avg={fmt_rs(v['pnl']/v['n'])}")

if __name__ == "__main__":
    main()
