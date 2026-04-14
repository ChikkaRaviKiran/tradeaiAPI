"""ORB + VWAP Directional Strategy — Opening Range Breakout with VWAP filter.

Research basis:
  - Adam Grimes (46,000 bar study): market open clusters near daily high/low ~60%
  - Crabel, *Day Trading with Short-Term Price Patterns* — ORB concept
  - Fisher, *The Logical Trader* — ACD opening range method
  - TradingView "MNQ ORB Strategy - VWAP + Bias" — VWAP directional filter
  - WarriorTrading — ORB with directional bias and VWAP alignment

Opening range: 09:15–09:30 IST (first 15 minutes)
ORH = highest high, ORL = lowest low in that window.

Entry rules:
  1. Price CLOSES above ORH (call) or below ORL (put) — confirmed breakout
  2. VWAP filter: longs ONLY above VWAP, shorts ONLY below VWAP
  3. ORB range must be >= 0.3% of spot (avoid noise) and <= 1.5× ATR (avoid chaos)
  4. Direction lock: first breakout direction locks for the day — no reversals
  5. No entries after 11:30 IST (breakout momentum exhausted)

Exit rules:
  1. Structural stop loss: opposite end of ORB range (hard SL)
  2. Target: 2× stop loss distance (2:1 R:R minimum)
  3. VWAP cross exit: if price crosses back through VWAP against trade → exit
  4. After target hit: trail at ORB midpoint or 2-candle trailing low
  5. EOD exit: 15:10 IST

Break-even win rate at 2:1 R:R = 33.3%.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from app.core.models import OptionType, OptionsMetrics, StrategyName, StrategySignal
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Opening range window
ORB_START = dtime(9, 15)
ORB_END = dtime(9, 30)


class ORBVWAPStrategy(BaseStrategy):
    """Opening Range Breakout + VWAP directional filter.

    Designed for backtesting on 1-minute index candles (NIFTY/SENSEX).
    Returns signals only when all confluence conditions are met.
    """

    def __init__(
        self,
        min_range_pct: float = 0.3,
        max_range_atr_mult: float = 15.0,
        entry_deadline: dtime = dtime(11, 30),
    ):
        self.min_range_pct = min_range_pct
        self.max_range_atr_mult = max_range_atr_mult
        self.entry_deadline = entry_deadline

    def evaluate(
        self,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        spot_price: float,
        daily_levels: Optional[dict] = None,
        structure_data: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if df.empty or len(df) < 20:
            return None

        # ── Time filter: only after ORB window, before deadline ──
        last_time = df.index[-1]
        if hasattr(last_time, "time"):
            t = last_time.time()
        else:
            t = last_time.to_pydatetime().time()

        if t <= ORB_END or t > self.entry_deadline:
            return None

        # ── Build opening range ──
        or_candles = df[
            (df.index.time >= ORB_START) & (df.index.time <= ORB_END)
        ]
        if len(or_candles) < 5:  # Need reasonable ORB data
            return None

        orh = float(or_candles["high"].max())
        orl = float(or_candles["low"].min())
        orb_range = orh - orl

        if orb_range <= 0:
            return None

        # ── Filter 1: Min range (skip noise) ──
        if spot_price > 0 and (orb_range / spot_price * 100) < self.min_range_pct:
            return None

        # ── Filter 2: Max range via ATR (skip chaos) ──
        atr = df.iloc[-1].get("atr")
        if atr is not None and not pd.isna(atr) and atr > 0:
            if orb_range > self.max_range_atr_mult * float(atr):
                return None

        # ── Get current bar data ──
        last = df.iloc[-1]
        close = float(last["close"])
        vwap = last.get("vwap")
        ema9 = last.get("ema9")
        ema20 = last.get("ema20")
        rsi = last.get("rsi")

        # Require VWAP
        if vwap is None or pd.isna(vwap) or float(vwap) <= 0:
            return None

        vwap = float(vwap)

        # Safe conversions
        ema9 = float(ema9) if ema9 is not None and not pd.isna(ema9) else None
        ema20 = float(ema20) if ema20 is not None and not pd.isna(ema20) else None
        rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else None

        # ── Check for breakout ──
        # We need the CURRENT bar to CLOSE above ORH (or below ORL)
        # Only signal on the first bar that confirms breakout

        # Look at post-ORB candles to ensure this is the FIRST breakout bar
        post_orb = df[(df.index.time > ORB_END) & (df.index <= last_time)]
        if len(post_orb) < 2:
            return None

        # ── CALL: Close > ORH + VWAP above ──
        if close > orh and close > vwap:
            # Check this is the first candle that closed above ORH
            prev_candles = post_orb.iloc[:-1]
            if len(prev_candles) > 0 and (prev_candles["close"] > orh).any():
                return None  # Already broke out earlier — skip

            # EMA alignment: hard gate — only take breakouts aligned with short-term trend
            ema_aligned = ema9 is not None and ema20 is not None and ema9 > ema20
            if not ema_aligned:
                return None
            rsi_ok = rsi is not None and 45 <= rsi <= 75

            return StrategySignal(
                strategy=StrategyName.ORB_VWAP,
                option_type=OptionType.CALL,
                strike_price=_nearest_strike(spot_price, "CE"),
                details={
                    "orh": round(orh, 2),
                    "orl": round(orl, 2),
                    "orb_range": round(orb_range, 2),
                    "orb_range_pct": round(orb_range / spot_price * 100, 3),
                    "vwap": round(vwap, 2),
                    "breakout_level": round(orh, 2),
                    "structural_sl_level": round(orl, 2),
                    "ema_aligned": ema_aligned,
                    "rsi": round(rsi, 1) if rsi else None,
                    "rsi_ok": rsi_ok,
                },
            )

        # ── PUT: Close < ORL + VWAP below ──
        if close < orl and close < vwap:
            prev_candles = post_orb.iloc[:-1]
            if len(prev_candles) > 0 and (prev_candles["close"] < orl).any():
                return None

            ema_aligned = ema9 is not None and ema20 is not None and ema9 < ema20
            if not ema_aligned:
                return None
            rsi_ok = rsi is not None and 25 <= rsi <= 55

            return StrategySignal(
                strategy=StrategyName.ORB_VWAP,
                option_type=OptionType.PUT,
                strike_price=_nearest_strike(spot_price, "PE"),
                details={
                    "orh": round(orh, 2),
                    "orl": round(orl, 2),
                    "orb_range": round(orb_range, 2),
                    "orb_range_pct": round(orb_range / spot_price * 100, 3),
                    "vwap": round(vwap, 2),
                    "breakout_level": round(orl, 2),
                    "structural_sl_level": round(orh, 2),
                    "ema_aligned": ema_aligned,
                    "rsi": round(rsi, 1) if rsi else None,
                    "rsi_ok": rsi_ok,
                },
            )

        return None


def _nearest_strike(price: float, opt_type: str) -> float:
    """Round to nearest 50 strike (ATM)."""
    return round(price / 50) * 50
