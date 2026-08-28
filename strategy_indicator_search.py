#!/usr/bin/env python3
"""INDICATOR-BASED entry/exit search (Dec 2025 – May 26 2026).

Goal: answer "does an indicator-triggered entry/exit beat the fixed-time
schedule from strategy_search_6mo?" using the SAME days and SAME chains.

Inputs reused from phase3a_breakdown_v2:
  psql, parse_expiry, to_min, load_chain, LOT_SIZE, LOTS, STRIKE_STEP

Approach
--------
For each cached (date, sym) chain we:
  1. Build a per-minute SYNTHETIC SPOT series via put-call-parity at the ATM
     strike (spot ≈ K + CE − PE).
  2. Build a per-minute STRADDLE-PREMIUM series anchored at the 09:20 ATM
     strike (proxy for implied vol / variance perception).
  3. Compute indicators on the spot series:
       EMA9, EMA20, cumulative-VWAP proxy (running mean since 09:15),
       RSI(14) on 1-min returns, BB(20, 2σ), realised-vol bandwidth.
     And on the premium series: premium EMA9, % drop from session open.

Strategies tested (= the 6-mo winners per weekday/index):
  Mon NIFTY  naked_PE_ATM         (long-bias bet)
  Tue NIFTY  straddle             (mean-revert / IV crush)
  Wed SENSEX straddle             (mean-revert / IV crush)
  Thu SENSEX naked_CE_ATM         (short-bias bet)
  Fri NIFTY  naked_CE_ATM         (short-bias bet)

For each (wd, sym, strategy):
  We sweep ENTRY ∈ {T0..T5} × EXIT ∈ {X0..X5} on the SAME days the chain
  exists, and compare to the BASELINE (T0 = fixed entry, X0 = fixed exit
  from the 6-mo schedule).  We only enter once per day; if entry trigger
  never fires before LAST_ENTRY (13:00) the day is a no-trade.

Entries (search until 13:00):
  T0 baseline   – fixed time (per baseline schedule)
  T1 iv_crush   – first minute after 09:25 where premium < EMA9(prem)
                   AND premium dropped ≥0.4 % from 09:20 open
  T2 vwap_align – spot on the "right" side of VWAP for direction:
                   naked_CE: spot < VWAP and EMA9<EMA20  (bear bias)
                   naked_PE: spot > VWAP and EMA9>EMA20  (bull bias)
                   straddle: |spot−VWAP| < 0.15% (mean-revert)
  T3 rsi_revert – RSI(14) crosses back through 50 from extreme:
                   naked_CE: RSI peaks ≥65 then crosses ↓50
                   naked_PE: RSI bottoms ≤35 then crosses ↑50
                   straddle: |RSI−50| ≤ 5 with BB width < 0.4%
  T4 bb_squeeze – Bollinger width (4σ/spot) below 0.6% AND spot inside band
                   (any strategy – low-vol entry)
  T5 momentum   – directional momentum confirmation
                   naked_CE: EMA9 cross below EMA20
                   naked_PE: EMA9 cross above EMA20
                   straddle: same as T4 (fallback)

Exits (evaluated each minute; whichever fires first; force exit 15:15):
  X0 baseline   – fixed scheduled exit time
  X1 tp30       – take-profit when MTM gain ≥ 30 % of entry credit
  X2 tp50_sl100 – TP 50 %, hard SL at 100 % of entry credit
  X3 vwap_adv   – spot crosses VWAP against the position (bearish for short PE,
                   bullish for short CE, |spot−VWAP|>0.2% for straddle)
  X4 rsi_flip   – RSI crosses against the direction (>55 for naked_CE,
                   <45 for naked_PE, |RSI−50|>20 for straddle)
  X5 vol_pop    – 5-min realised vol > 2× session-mean vol up to that minute
                   (volatility expansion blows out short-vol trades)

We also force a 100 % entry-credit hard SL on every exit-variant so all
configs share the same tail-risk cap.

Output: prints per-(wd,sym,strategy) ranking + winner overlay vs baseline,
and writes ~/strategy_indicator_search.log and /tmp/strategy_indicator.json.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, date
from statistics import mean, pstdev
import json
import math

from phase3a_breakdown_v2 import (
    psql, parse_expiry, to_min, fmt_rs,
    STRIKE_STEP,
)
import subprocess

# Override stale phase3a constants: live lot sizes are NIFTY=65, SENSEX=20.
LOT_SIZE = {"NIFTY": 65, "SENSEX": 20}
# Lot count per leg-pair. 4+4 lots multi-index ≈ ₹12L margin (within 13L cap).
LOTS = 4

# ------------------------------------------------------- HLC chain loader ----
def load_chain_hlc(d, sym, exp_s):
    """Like phase3a load_chain but brings HIGH, LOW and VOLUME so that
    intra-minute SL/TP can fire on wicks AND we can build a true
    volume-weighted VWAP from option-chain flow.
    Returns {(strike,opt): {minute: (close, high, low, volume)}}.
    """
    sql = (f"SELECT strike, option_type, timestamp, close, high, low, volume "
           f"FROM option_candles WHERE instrument='{sym}' AND date='{d}' "
           f"AND expiry='{exp_s}' AND close IS NOT NULL ORDER BY timestamp;")
    rows = psql(sql)
    bars = defaultdict(dict)
    for r in rows:
        if len(r) < 7: continue
        strike, opt, ts, close, high, low, vol = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        try:
            dt = datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
            m = dt.hour*60 + dt.minute
            c = float(close)
            h = float(high) if high not in ("", None) else c
            l = float(low)  if low  not in ("", None) else c
            v = int(vol)    if vol  not in ("", None) else 0
            bars[(int(strike), opt)][m] = (c, h, l, v)
        except Exception:
            continue
    return bars

START = "2026-04-15"
END   = "2026-06-01"
REGIME_SPLIT = date(2026, 4, 15)   # entire window is post-regime

OPEN_MIN  = to_min("09:15")
ENTRY_OPEN= to_min("09:20")
LAST_ENT  = to_min("13:00")
EOD_MIN   = to_min("15:15")

# Live-system hard ₹ gates (match gated_run.py / postregime_search.py / phase3a_actual_6k.py)
# NO take-profit — SL only.
SL_RS = 6000.0          # hard stop — max ₹ loss per trade

# Baseline ENTRY/EXIT times per weekday (from strategy_search_6mo schedule).
# We sweep STRATS × ENTRY × EXIT – *no naked* (only straddle / strangle).
BASELINE_TIMES = {
    # Primary index per weekday (from strategy_search_6mo schedule):
    ("Mon","NIFTY")  : ("09:20", "14:30"),
    ("Tue","NIFTY")  : ("09:30", "14:30"),
    ("Wed","SENSEX") : ("09:45", "15:15"),
    ("Thu","SENSEX") : ("09:30", "15:15"),
    ("Fri","NIFTY")  : ("11:00", "15:15"),
    # Secondary index per weekday — added for 13L multi-index sweep.
    ("Mon","SENSEX") : ("09:20", "15:15"),
    ("Tue","SENSEX") : ("09:30", "15:15"),
    ("Wed","NIFTY")  : ("09:45", "15:15"),
    ("Thu","NIFTY")  : ("09:30", "15:15"),
    ("Fri","SENSEX") : ("11:00", "15:15"),
}
# The weekday→(primary) index used by single-index 4.4L plan:
PRIMARY_INDEX = {"Mon":"NIFTY","Tue":"NIFTY","Wed":"SENSEX","Thu":"SENSEX","Fri":"NIFTY"}
# Strategies to sweep (defined-risk style: straddle or strangle, no naked legs).
STRATS = ["straddle", "strangle_1", "strangle_2"]

# ---------------------------------------------------------------- helpers ----
def _ohlc(bars, strike, opt, t_min, slack=5):
    """Return (close, high, low, volume) tuple at/near t_min, or None."""
    s = bars.get((strike, opt))
    if not s: return None
    for m in range(t_min, t_min+slack):
        v = s.get(m)
        if v is not None: return v
    return None

def price_at(bars, strike, opt, t_min, slack=5):
    """Return CLOSE price (used for spot/premium series and entry credit)."""
    v = _ohlc(bars, strike, opt, t_min, slack)
    return v[0] if v else None

def leg_cost_close(bars, legs, m, slack=1):
    c = 0.0
    for k, side, qty in legs:
        v = _ohlc(bars, k, side, m, slack)
        if v is None: return None
        c += qty * v[0]
    return c

def leg_cost_worst(bars, legs, m, slack=1):
    """Worst-case cost-to-buy-back: use HIGH for short legs, LOW for long legs."""
    c = 0.0
    for k, side, qty in legs:
        v = _ohlc(bars, k, side, m, slack)
        if v is None: return None
        _, hi, lo, _ = v
        c += qty * (hi if qty > 0 else lo)
    return c

def leg_cost_best(bars, legs, m, slack=1):
    """Best-case cost-to-buy-back: use LOW for short legs, HIGH for long legs."""
    c = 0.0
    for k, side, qty in legs:
        v = _ohlc(bars, k, side, m, slack)
        if v is None: return None
        _, hi, lo, _ = v
        c += qty * (lo if qty > 0 else hi)
    return c

def find_atm(bars, t_min, sym):
    step = STRIKE_STEP[sym]
    strikes = sorted({k[0] for k in bars.keys()})
    best=None; bd=None
    for s in strikes:
        ce=pe=None
        for m in range(t_min, t_min+5):
            if ce is None:
                v = bars.get((s,"CE"),{}).get(m); ce = v[0] if v else None
            if pe is None:
                v = bars.get((s,"PE"),{}).get(m); pe = v[0] if v else None
            if ce is not None and pe is not None: break
        if ce is None or pe is None: continue
        d = abs(ce-pe)
        if best is None or d<bd: bd=d; best=s
    if best is None: return None
    return int(round(best/step)*step) if best % step else best

def build_chain_volume(bars, mins):
    """Return {minute: total_chain_volume} — sum of every (strike,opt) volume at m."""
    vol = {}
    for (k, opt), series in bars.items():
        for m, tup in series.items():
            vol[m] = vol.get(m, 0) + (tup[3] if len(tup) >= 4 else 0)
    return vol

def build_minute_series(bars, sym):
    """Return (minutes_sorted, spot[m], straddle_prem_anchored[m])
    spot uses per-minute ATM (PCP); premium uses 09:20 ATM strike held constant.
    """
    step = STRIKE_STEP[sym]
    strikes = sorted({k[0] for k in bars.keys()})
    # All minutes that have at least one quote
    mins = sorted({m for s in strikes for opt in ("CE","PE") for m in bars.get((s,opt),{})})
    spot = {}
    for m in mins:
        best=None; bd=None; bspot=None
        for s in strikes:
            cv = bars.get((s,"CE"),{}).get(m); pv = bars.get((s,"PE"),{}).get(m)
            if cv is None or pv is None: continue
            ce, pe = cv[0], pv[0]
            d = abs(ce-pe)
            if best is None or d<bd:
                bd=d; best=s; bspot = s + ce - pe
        if bspot is not None: spot[m] = bspot
    atm0 = find_atm(bars, ENTRY_OPEN, sym)
    prem = {}
    if atm0 is not None:
        for m in mins:
            cv = bars.get((atm0,"CE"),{}).get(m); pv = bars.get((atm0,"PE"),{}).get(m)
            if cv is not None and pv is not None: prem[m] = cv[0] + pv[0]
    return mins, spot, prem, atm0

def ema(series_dict, mins, span):
    k = 2 / (span+1); out = {}; e = None
    for m in mins:
        v = series_dict.get(m)
        if v is None: continue
        e = v if e is None else (v - e)*k + e
        out[m] = e
    return out

def rsi(series_dict, mins, period=14):
    out = {}; gains=[]; losses=[]; prev=None; ag=al=None
    for m in mins:
        v = series_dict.get(m)
        if v is None:
            prev=None; continue
        if prev is not None:
            ch = v-prev
            gains.append(max(ch,0.0)); losses.append(max(-ch,0.0))
            if len(gains) > period: gains.pop(0); losses.pop(0)
            if len(gains) == period:
                ag = mean(gains); al = mean(losses)
                if al == 0: out[m] = 100.0
                else:
                    rs = ag/al; out[m] = 100 - 100/(1+rs)
        prev = v
    return out

def cum_vwap(series_dict, mins, vol_dict=None):
    """Cumulative volume-weighted average. If vol_dict is provided, weights are
    the option-chain aggregate volume per minute (a real activity proxy); otherwise
    falls back to volume-less running mean.
    """
    out = {}; sv = 0.0; sw = 0.0
    for m in mins:
        v = series_dict.get(m)
        if v is None: continue
        w = (vol_dict.get(m, 0) if vol_dict else 1) or 0
        if w <= 0:
            # No flow this minute — carry forward last VWAP, do not poison the mean
            if sw > 0: out[m] = sv / sw
            continue
        sv += v * w
        sw += w
        out[m] = sv / sw
    return out

def bbands(series_dict, mins, period=20, k=2.0):
    """Return (mid, width_pct) per minute. width_pct = 2k·σ / mid."""
    out_mid={}; out_w={}
    buf=[]
    for m in mins:
        v = series_dict.get(m)
        if v is None: continue
        buf.append(v)
        if len(buf) > period: buf.pop(0)
        if len(buf) == period:
            mid = mean(buf); sd = pstdev(buf)
            out_mid[m] = mid
            out_w[m]   = (2*k*sd / mid) if mid else None
    return out_mid, out_w

def realised_vol_5m(series_dict, mins):
    """Rolling 5-min stdev of 1-min log returns."""
    out={}; rets=[]; prev=None
    for m in mins:
        v = series_dict.get(m)
        if v is None or v <= 0:
            prev=None; continue
        if prev is not None and prev > 0:
            rets.append(math.log(v/prev))
            if len(rets) > 5: rets.pop(0)
            if len(rets) == 5: out[m] = pstdev(rets)
        prev = v
    return out

# ---------------------------------------------------------- entry triggers --
def trigger_entry(kind, strat_dir, mins, spot, prem, ema9, ema20, vwap_, rsi_, bb_w):
    """Return the entry minute (or None). strat_dir ∈ {'short_ce','short_pe','straddle'}."""
    if kind == "T0":   # baseline – caller supplies fixed minute
        return None
    open_prem = None
    for m in mins:
        if m >= ENTRY_OPEN and m in prem:
            open_prem = prem[m]; break
    peak_rsi = -1; trough_rsi = 101
    for m in mins:
        if m < ENTRY_OPEN + 5 or m > LAST_ENT: continue
        if m not in spot: continue
        sp = spot[m]
        e9 = ema9.get(m); e20 = ema20.get(m); vw = vwap_.get(m); r = rsi_.get(m); bw = bb_w.get(m)
        pr = prem.get(m); pr_e9 = None  # set below

        if kind == "T1":  # iv_crush
            if pr is None or open_prem is None: continue
            # need premium EMA9 too – computed externally; we get it via ema9 alias? No.
            # We approximate by requiring premium below 09:20 open by ≥0.4 % and
            # premium not making a new high in last 5 min.
            drop = (pr - open_prem) / open_prem if open_prem else 0
            if drop <= -0.004:
                return m
            continue
        if kind == "T2":  # vwap_align
            if vw is None or e9 is None or e20 is None: continue
            if strat_dir == "short_ce":
                if sp < vw and e9 < e20: return m
            elif strat_dir == "short_pe":
                if sp > vw and e9 > e20: return m
            else:  # straddle
                if abs(sp-vw)/sp < 0.0015 and bw is not None and bw < 0.006: return m
            continue
        if kind == "T3":  # rsi_revert
            if r is None: continue
            if strat_dir == "short_ce":
                peak_rsi = max(peak_rsi, r)
                if peak_rsi >= 65 and r < 50: return m
            elif strat_dir == "short_pe":
                trough_rsi = min(trough_rsi, r)
                if trough_rsi <= 35 and r > 50: return m
            else:
                if abs(r-50) <= 5 and bw is not None and bw < 0.005: return m
            continue
        if kind == "T4":  # bb_squeeze
            if bw is None: continue
            if bw < 0.006: return m
            continue
        if kind == "T5":  # momentum cross
            if e9 is None or e20 is None: continue
            # need prior bar
            pm = m-1
            pe9 = ema9.get(pm); pe20 = ema20.get(pm)
            if pe9 is None or pe20 is None: continue
            if strat_dir == "short_ce" and pe9 >= pe20 and e9 < e20: return m
            if strat_dir == "short_pe" and pe9 <= pe20 and e9 > e20: return m
            if strat_dir == "straddle":
                if bw is not None and bw < 0.006: return m
            continue
    return None

# ---------------------------------------------------------- exit triggers ---
def evaluate_exit(kind, strat_dir, e_min, x_baseline, credit, leg_keys, bars, sym,
                  mins, spot, vwap_, rsi_, rv5):
    """Walk minute-by-minute from e_min+1 to EOD_MIN; return (exit_min, reason).
    'kind' selects the trigger menu; X0 returns the baseline minute.
    The hard ₹ SL is always active, including for X0 fixed-time exits.
    """
    rsi_peak = -1; rsi_trough = 101
    base_rv = None
    rv_seen = []
    rs_per_pt = LOT_SIZE[sym] * LOTS
    last_minute = min(EOD_MIN, max(mins)) if mins else EOD_MIN
    if kind == "X0":
        last_minute = min(last_minute, x_baseline)
    for m in range(e_min+1, last_minute+1):
        # ---- Hard ₹ SL check (always on): WORST-case cost (HIGH of short legs).
        worst_cost = leg_cost_worst(bars, leg_keys, m, slack=2)
        if worst_cost is not None:
            worst_mtm_rs = (credit - worst_cost) * rs_per_pt
            if worst_mtm_rs <= -SL_RS:
                return m, "sl_rs"
        if kind == "X3":  # vwap adverse — signal exit at close
            sp = spot.get(m); vw = vwap_.get(m)
            if sp is not None and vw is not None:
                if strat_dir == "short_ce" and sp > vw: return m, "vwap_adv_up"
                if strat_dir == "short_pe" and sp < vw: return m, "vwap_adv_dn"
                if strat_dir == "straddle":
                    if abs(sp-vw)/sp > 0.002: return m, "vwap_drift"
        if kind == "X4":  # rsi flip
            r = rsi_.get(m)
            if r is not None:
                if strat_dir == "short_ce":
                    if r > 55: return m, "rsi_up"
                elif strat_dir == "short_pe":
                    if r < 45: return m, "rsi_dn"
                else:
                    if abs(r-50) > 20: return m, "rsi_extreme"
        if kind == "X5":  # vol pop
            v = rv5.get(m)
            if v is not None:
                rv_seen.append(v)
                base_rv = mean(rv_seen)
                if base_rv and v > 2.0 * base_rv and len(rv_seen) > 30:
                    return m, "vol_pop"
    if kind == "X0":
        return last_minute, "fixed"
    return last_minute, "eod"

# --------------------------------------------------- strategy → leg builder --
def legs_for(strat_name, sym, atm):
    step = STRIKE_STEP[sym]
    if strat_name == "straddle":
        return [(atm,"CE",+1),(atm,"PE",+1)], "straddle"
    if strat_name == "strangle_1":
        return [(atm+step,"CE",+1),(atm-step,"PE",+1)], "straddle"
    if strat_name == "strangle_2":
        return [(atm+2*step,"CE",+1),(atm-2*step,"PE",+1)], "straddle"
    raise ValueError(strat_name)

def credit_at(bars, legs, t_min):
    c = 0.0
    for k, side, qty in legs:
        px = price_at(bars, k, side, t_min, slack=5)
        if px is None: return None
        c += qty * px
    return c

# --------------------------------------------------------- data load ---------
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

# ----------------------------------------------------------------- main ------
def main():
    near = find_days()
    print(f"(date,sym) combos: {len(near)}   window {START} → {END}", flush=True)

    cache = {}
    items = list(near.items())
    for i, ((d, sym), exps) in enumerate(items, 1):
        ex, exp_s = exps[0]
        bars = load_chain_hlc(d, sym, exp_s)
        if bars: cache[(d, sym)] = bars
        if i % 40 == 0:
            print(f"  [{i}/{len(items)}] cached={len(cache)}", flush=True)
    print(f"Cached chains: {len(cache)}\n", flush=True)

    ENTRY_KINDS = ["T0","T1","T2","T3","T4","T5"]
    EXIT_KINDS  = ["X0","X3","X4","X5"]   # no TPs — only time/signal exits + always-on ₹6000 SL
    results = []   # list of dicts

    for (wd_sym, times) in BASELINE_TIMES.items():
        wd, sym = wd_sym
        bl_entry, bl_exit = times
        bl_e_min = to_min(bl_entry); bl_x_min = to_min(bl_exit)
        days = [(d, cache[(d,sym)]) for (d,s), _ in near.items()
                if s == sym and d.strftime("%a") == wd and (d,sym) in cache]
        if not days: continue
        print(f"\n══ {wd} {sym}  baseline window {bl_entry}→{bl_exit}  n={len(days)}", flush=True)

        # precompute indicator series for all days
        prepped = []
        for d, bars in days:
            mins, spot, prem, atm0 = build_minute_series(bars, sym)
            if not mins or atm0 is None: continue
            chain_vol = build_chain_volume(bars, mins)
            e9 = ema(spot, mins, 9); e20 = ema(spot, mins, 20)
            vw = cum_vwap(spot, mins, chain_vol); rs = rsi(spot, mins, 14)
            _, bw = bbands(spot, mins, 20, 2.0)
            rv = realised_vol_5m(spot, mins)
            prepped.append((d, bars, mins, spot, prem, atm0, e9, e20, vw, rs, bw, rv))

        for strat_name in STRATS:
          for ek in ENTRY_KINDS:
            for xk in EXIT_KINDS:
                pnls=[]; pre=[]; post=[]; reasons=defaultdict(int); skipped=0
                trades=[]   # per-trade detail for this (strat, ek, xk)
                for d, bars, mins, spot, prem, atm0, e9, e20, vw, rs, bw, rv in prepped:
                    if ek == "T0":
                        em = bl_e_min
                    else:
                        em = trigger_entry(ek, legs_for(strat_name, sym, atm0)[1],
                                           mins, spot, prem, e9, e20, vw, rs, bw)
                        if em is None: skipped += 1; continue
                    atm = find_atm(bars, em, sym)
                    if atm is None: skipped += 1; continue
                    legs, dir_ = legs_for(strat_name, sym, atm)
                    credit = credit_at(bars, legs, em)
                    if credit is None or credit <= 0: skipped += 1; continue
                    xm, reason = evaluate_exit(xk, dir_, em, bl_x_min, credit, legs,
                                               bars, sym, mins, spot, vw, rs, rv)
                    # Price the exit honestly based on reason:
                    #   sl_rs   = hard ₹ stop fill → pnl = -SL_RS exactly
                    #   all others (fixed / vwap / rsi / vol / eod) → mark to CLOSE
                    rs_per_pt = LOT_SIZE[sym] * LOTS
                    if reason == "sl_rs":
                        pnl = -SL_RS
                    else:
                        exit_cost = leg_cost_close(bars, legs, xm, slack=5)
                        if exit_cost is None:
                            for back in range(xm-1, em, -1):
                                exit_cost = leg_cost_close(bars, legs, back, slack=1)
                                if exit_cost is not None: xm = back; break
                        if exit_cost is None: skipped += 1; continue
                        pts = credit - exit_cost
                        pnl = pts * rs_per_pt
                        # Honour the live ₹ SL floor on signal/time exits too.
                        if pnl < -SL_RS: pnl = -SL_RS
                    pnls.append((d, pnl))
                    reasons[reason] += 1
                    trades.append({
                        "date": str(d),
                        "strat": strat_name,
                        "entry_min": em,
                        "entry_time": f"{em//60:02d}:{em%60:02d}",
                        "atm": atm,
                        "legs": [(k, side) for k,side,_ in legs],
                        "credit_pts": round(credit, 2),
                        "credit_rs":  round(credit * LOT_SIZE[sym] * LOTS, 0),
                        "exit_min": xm,
                        "exit_time": f"{xm//60:02d}:{xm%60:02d}",
                        "reason": reason,
                        "pnl": round(pnl, 0),
                    })
                    if d < REGIME_SPLIT: pre.append(pnl)
                    else: post.append(pnl)
                if len(pnls) < max(3, int(0.5*len(prepped))): continue
                cum = sum(p for _,p in pnls)
                w   = sum(1 for _,p in pnls if p>0)
                worst = min(p for _,p in pnls)
                results.append({
                    "wd":wd,"sym":sym,"strat":strat_name,
                    "entry_kind":ek,"exit_kind":xk,
                    "n":len(pnls),"win":w,"cum":cum,"worst":worst,
                    "pre_cum":sum(pre),"pre_n":len(pre),
                    "post_cum":sum(post),"post_n":len(post),
                    "skipped":skipped,
                    "reasons":dict(reasons),
                    "trades":trades,
                })

    # ============================================================ report ===
    print("\n══════════ TIME-BASED ONLY  (T0/X0, SL ₹6,000 wick-aware, no TP) ══════════")
    print(f"  {'Day':3} {'Sym':6} {'Strat':10} {'n':>3} {'win%':>4} {'Cum':>11} {'Worst':>10}  reasons")
    time_only_totals = defaultdict(float)   # by strategy
    for r in sorted(results, key=lambda r: (["Mon","Tue","Wed","Thu","Fri"].index(r["wd"]),
                                            r["sym"], r["strat"])):
        if r["entry_kind"] != "T0" or r["exit_kind"] != "X0": continue
        wpct = int(100*r['win']/r['n']) if r['n'] else 0
        print(f"  {r['wd']:3} {r['sym']:6} {r['strat']:10} {r['n']:>3} {wpct:>3}% "
              f"{fmt_rs(r['cum']):>11} {fmt_rs(r['worst']):>10}  {r['reasons']}")
        time_only_totals[r["strat"]] += r["cum"]
    print(f"\n  Time-based weekly totals (5 days/wk):")
    for strat, tot in time_only_totals.items():
        print(f"    {strat:10} = {fmt_rs(tot)}")

    print("\n══════════ TOP CONFIGS PER (WEEKDAY, INDEX) ══════════")
    by_ws = defaultdict(list)
    for r in results: by_ws[(r["wd"], r["sym"])].append(r)
    # Pick the stable best per (wd, sym) — used for BOTH the single-index plan
    # (primary index only) and the multi-index plan (both indices summed).
    best_per_ws = {}
    bl_per_ws = {}
    for (wd, sym), rs_ in by_ws.items():
        rs_.sort(key=lambda r: -r["cum"])
        bl = next((r for r in rs_ if r["strat"]=="straddle" and
                                       r["entry_kind"]=="T0" and r["exit_kind"]=="X0"), None)
        bl_per_ws[(wd,sym)] = bl["cum"] if bl else 0
        # Only print top-configs for the primary index to keep the report readable
        if PRIMARY_INDEX.get(wd) == sym:
            print(f"\n  {wd} {sym}  straddle T0/X0 baseline cum = {fmt_rs(bl_per_ws[(wd,sym)])}")
            print(f"    {'Strat':10} {'En':3} {'Ex':3} {'n':>3} {'win%':>4} {'Cum':>11} {'vs BL':>9} {'Worst':>10}  reasons")
            for r in rs_[:10]:
                delta = r["cum"] - bl_per_ws[(wd,sym)]
                wpct = int(100*r['win']/r['n']) if r['n'] else 0
                print(f"    {r['strat']:10} {r['entry_kind']:3} {r['exit_kind']:3} {r['n']:>3} {wpct:>3}% "
                      f"{fmt_rs(r['cum']):>11} {fmt_rs(delta):>9} "
                      f"{fmt_rs(r['worst']):>10}  {r['reasons']}")
        stable = [r for r in rs_ if r["post_n"] >= 4 and r["post_cum"] >= 0]
        best_per_ws[(wd,sym)] = stable[0] if stable else rs_[0]

    # Single-index 4.4L plan = primary index per weekday only
    chosen = {(wd, sym): best_per_ws[(wd, sym)]
              for (wd, sym) in best_per_ws
              if PRIMARY_INDEX.get(wd) == sym}
    grand_total = sum(r["cum"] for r in chosen.values())
    bl_total    = sum(bl_per_ws[k] for k in chosen.keys())

    print("\n══════════ INDICATOR PHASE-3a (best per day) ══════════")
    for (wd,sym), r in chosen.items():
        print(f"  {wd} {sym}: {r['strat']:11} entry={r['entry_kind']} exit={r['exit_kind']}  "
              f"n={r['n']:2d}  win={int(100*r['win']/r['n']):3d}%  cum={fmt_rs(r['cum']):>10}  "
              f"post={fmt_rs(r['post_cum']):>9}  worst={fmt_rs(r['worst'])}")
    print(f"\n  Indicator-schedule total : {fmt_rs(grand_total)}  ({len(chosen)} days/wk)")
    print(f"  Straddle T0/X0 baseline  : {fmt_rs(bl_total)}")
    print(f"  Delta                    : {fmt_rs(grand_total - bl_total)}")

    # ───────────────── MULTI-INDEX 13L PLAN (NIFTY + SENSEX every day) ─────
    print("\n══════════ TIME-BASED MULTI-INDEX  (T0/X0 both indices, no indicators) ══════════")
    print(f"  {'Day':3} {'Sym':6} {'Strat':10} {'n':>3} {'win%':>4} {'Cum':>11} {'Worst':>10}")
    tb_multi_total = 0
    tb_multi_by_day = defaultdict(float)
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            # T0/X0 straddle = the comparable time-based baseline for that (wd,sym)
            r = next((x for x in by_ws.get((wd, sym), [])
                      if x["strat"]=="straddle" and x["entry_kind"]=="T0" and x["exit_kind"]=="X0"),
                     None)
            if r is None:
                print(f"  {wd:3} {sym:6} (no data)")
                continue
            wpct = int(100*r['win']/r['n']) if r['n'] else 0
            print(f"  {wd:3} {sym:6} {r['strat']:10} {r['n']:>3} {wpct:>3}% "
                  f"{fmt_rs(r['cum']):>11} {fmt_rs(r['worst']):>10}")
            tb_multi_total += r["cum"]
            tb_multi_by_day[wd] += r["cum"]
    print(f"\n  Time-based multi-index per-day totals:")
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        print(f"    {wd}: {fmt_rs(tb_multi_by_day[wd])}")
    print(f"  Time-based multi-index weekly : {fmt_rs(tb_multi_total)}")

    print("\n══════════ MULTI-INDEX 13L PLAN  (NIFTY + SENSEX same day, indicator-gated) ══════════")
    print(f"  {'Day':3} {'Sym':6} {'Strat':10} {'En':3} {'Ex':3} {'n':>3} {'win%':>4} "
          f"{'Cum':>11} {'Worst':>10}")
    multi_total = 0
    multi_by_day = defaultdict(float)
    multi_chosen = {}
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            r = best_per_ws.get((wd, sym))
            if r is None:
                print(f"  {wd:3} {sym:6} (no data)")
                continue
            wpct = int(100*r['win']/r['n']) if r['n'] else 0
            print(f"  {wd:3} {sym:6} {r['strat']:10} {r['entry_kind']:3} {r['exit_kind']:3} "
                  f"{r['n']:>3} {wpct:>3}% {fmt_rs(r['cum']):>11} {fmt_rs(r['worst']):>10}")
            multi_total += r["cum"]
            multi_by_day[wd] += r["cum"]
            multi_chosen[(wd, sym)] = r
    print(f"\n  Per-day totals (NIFTY + SENSEX combined):")
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        print(f"    {wd}: {fmt_rs(multi_by_day[wd])}")
    print(f"\n  Multi-index 13L weekly total  : {fmt_rs(multi_total)}  (indicator-gated)")
    print(f"  Multi-index time-based weekly : {fmt_rs(tb_multi_total)}  (T0/X0 straddle both)")
    print(f"  Indicator vs time-based       : {fmt_rs(multi_total - tb_multi_total)}  "
          f"({(multi_total/tb_multi_total - 1)*100:+.1f}%)")
    print(f"  Single-index 4.4L weekly total: {fmt_rs(grand_total)}  (indicator, primary only)")
    print(f"  Multi-index uplift vs single  : {fmt_rs(multi_total - grand_total)}  "
          f"({(multi_total/grand_total - 1)*100:+.1f}% on PnL)")
    print(f"  Capital: single ≈ ₹4.4L (3 lots, 1 index) | multi ≈ ₹{LOTS*2.2:.1f}L ({LOTS}+{LOTS} lots)")

    # ───────────────── per-trade detail for the chosen schedule ─────────────
    print("\n══════════ FULL TRADE LIST (single-index 4.4L schedule) ══════════")
    for (wd,sym), r in chosen.items():
        print(f"\n  ── {wd} {sym}  {r['strat']}  entry={r['entry_kind']} exit={r['exit_kind']} ──")
        print(f"     {'Date':10} {'Entry':5} {'ATM':>6} {'Legs':22} {'Credit₹':>9} {'Exit':5} {'Reason':12} {'PnL':>9}")
        for t in r.get("trades", []):
            legs_s = ",".join(f"{k}{s}" for k,s in t["legs"])
            print(f"     {t['date']:10} {t['entry_time']:5} {t['atm']:>6} {legs_s:22} "
                  f"{int(t['credit_rs']):>9,} {t['exit_time']:5} {t['reason']:12} "
                  f"{fmt_rs(t['pnl']):>9}")

    print("\n══════════ FULL TRADE LIST (multi-index 13L schedule) ══════════")
    for wd in ["Mon","Tue","Wed","Thu","Fri"]:
        for sym in ["NIFTY","SENSEX"]:
            r = multi_chosen.get((wd, sym))
            if r is None: continue
            print(f"\n  ── {wd} {sym}  {r['strat']}  entry={r['entry_kind']} exit={r['exit_kind']} ──")
            print(f"     {'Date':10} {'Entry':5} {'ATM':>6} {'Legs':22} {'Credit₹':>9} {'Exit':5} {'Reason':12} {'PnL':>9}")
            for t in r.get("trades", []):
                legs_s = ",".join(f"{k}{s}" for k,s in t["legs"])
                print(f"     {t['date']:10} {t['entry_time']:5} {t['atm']:>6} {legs_s:22} "
                      f"{int(t['credit_rs']):>9,} {t['exit_time']:5} {t['reason']:12} "
                      f"{fmt_rs(t['pnl']):>9}")

    out = {
        "window":[START,END],
        "regime_split": str(REGIME_SPLIT),
        "baseline_times": {f"{wd}_{sym}": list(v) for (wd,sym),v in BASELINE_TIMES.items()},
        "strats_tested": STRATS,
        "chosen": {f"{wd}_{sym}": {k:(v if not isinstance(v,float) else round(v,2))
                                    for k,v in r.items() if k!="reasons"}
                   for (wd,sym),r in chosen.items()},
        "indicator_total": round(grand_total,2),
        "baseline_total":  round(bl_total,2),
    }
    with open("/tmp/strategy_indicator.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nWrote /tmp/strategy_indicator.json")

if __name__ == "__main__":
    main()
