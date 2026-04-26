"""Strategy testing engine — runs all strategies on historical or live data.

Core of the analysis system. StrategyTester runs every strategy on a day's
data with a uniform exit framework so entry quality is compared fairly.

Uniform Exit Rules (applied to ALL strategies):
  - SL  = entry_price × 0.80 (20% risk = 1R)
  - T1  = entry_price + 1R   (20% reward)
  - T2  = entry_price + 2R   (40% reward)
  - After T1 hit: SL → breakeven + 0.5%
  - After T2 hit: SL → lock 1R profit
  - EOD = 15:10 IST

TimeWindow buckets for time-of-day analysis:
  EARLY_OPEN   09:15–09:45
  POST_ORB     09:45–10:15
  MID_MORNING  10:15–11:00
  LATE_MORNING 11:00–12:00
  AFTERNOON    12:00–13:30
  LATE_SESSION 13:30–15:15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as dtime, datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from app.core.models import DayType, OptionsMetrics, StrategySignal
from app.engine.feature_engine import FeatureEngine

# Strategy imports
from app.strategies.orb import ORBStrategy
from app.strategies.orb_vwap import ORBVWAPStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.range_breakout import SL_PCT as RB_SL_PCT, MIN_ENTRY_PREMIUM as RB_MIN_ENTRY_PREMIUM
from app.strategies.ema_breakout import EMABreakoutStrategy
from app.strategies.rsi_extreme import RSIExtremeStrategy
from app.strategies.momentum_option_buying import MomentumOptionBuyingStrategy

logger = logging.getLogger(__name__)

# ── Strategy registry (all 10 testable strategies) ───────────────────

STRATEGIES = {
    "ORB": ORBStrategy,
    "ORB_VWAP": ORBVWAPStrategy,
    "VWAP_RECLAIM": VWAPReclaimStrategy,
    "TREND_PULLBACK": TrendPullbackStrategy,
    "LIQUIDITY_SWEEP": LiquiditySweepStrategy,
    "MOMENTUM_BREAKOUT": MomentumBreakoutStrategy,
    "RANGE_BREAKOUT": RangeBreakoutStrategy,
    "EMA_BREAKOUT": EMABreakoutStrategy,
    "RSI_EXTREME": RSIExtremeStrategy,
    "MOMENTUM_OPTION_BUYING": MomentumOptionBuyingStrategy,
}

EOD_EXIT_TIME = dtime(15, 10)
MIN_WARMUP_BARS = 30


# ── Time Window ──────────────────────────────────────────────────────

class TimeWindow:
    EARLY_OPEN = "09:15-09:45"
    POST_ORB = "09:45-10:15"
    MID_MORNING = "10:15-11:00"
    LATE_MORNING = "11:00-12:00"
    AFTERNOON = "12:00-13:30"
    LATE_SESSION = "13:30-15:15"

    ALL = [
        "09:15-09:45", "09:45-10:15", "10:15-11:00",
        "11:00-12:00", "12:00-13:30", "13:30-15:15",
    ]

    @staticmethod
    def from_time(t: dtime) -> str:
        if t < dtime(9, 45):
            return TimeWindow.EARLY_OPEN
        if t < dtime(10, 15):
            return TimeWindow.POST_ORB
        if t < dtime(11, 0):
            return TimeWindow.MID_MORNING
        if t < dtime(12, 0):
            return TimeWindow.LATE_MORNING
        if t < dtime(13, 30):
            return TimeWindow.AFTERNOON
        return TimeWindow.LATE_SESSION


# ── Trade Result ─────────────────────────────────────────────────────

@dataclass
class TradeResult:
    date: str
    instrument: str
    strategy: str
    direction: str          # CE or PE
    entry_time: str         # HH:MM
    exit_time: str          # HH:MM
    entry_price: float
    exit_price: float
    strike: float
    pnl: float              # absolute PnL for 1 lot
    pnl_pct: float          # PnL as % of entry
    exit_reason: str
    hold_minutes: int
    r_multiple: float       # PnL / 1R (positive = profit)
    time_window: str        # TimeWindow value
    day_type: str           # Realtime classification
    day_type_hindsight: str # End-of-day ground truth
    lot_size: int = 0


# ── Strategy Tester ──────────────────────────────────────────────────

class StrategyTester:
    """Run all strategies on day data with uniform exit logic.

    Used by:
      - strategy_analyzer.py (backtest: test all strategies on all days)
      - Live orchestrator (test recommended strategies on today's data)
    """

    def __init__(self, fe: FeatureEngine = None, strategy_filter: List[str] = None):
        self.fe = fe or FeatureEngine()
        if strategy_filter:
            self.strategies = {
                k: cls() for k, cls in STRATEGIES.items() if k in strategy_filter
            }
        else:
            self.strategies = {k: cls() for k, cls in STRATEGIES.items()}

    def test_day(
        self,
        df: pd.DataFrame,
        instrument_symbol: str,
        strike_interval: int,
        lot_size: int,
        option_cache: Dict,
        oi_snapshots: Optional[List] = None,
        day_type: str = "unclear",
        day_type_hindsight: str = "unclear",
        date_str: str = "",
    ) -> List[TradeResult]:
        """Run ALL strategies on one day. Returns list of trade results.

        Args:
            df: Raw 1-min OHLCV DataFrame (not yet enriched).
            instrument_symbol: e.g. "NIFTY"
            strike_interval: e.g. 50 for NIFTY
            lot_size: e.g. 65 for NIFTY
            option_cache: {(inst, date, strike, type): DataFrame}
            oi_snapshots: [(time, OptionsMetrics), ...] for this date
            day_type: realtime classification string
            day_type_hindsight: hindsight classification string
            date_str: "YYYY-MM-DD"
        """
        if df is None or df.empty or len(df) < MIN_WARMUP_BARS:
            return []

        df_enriched = self.fe.compute_indicators(df.copy())
        trades = []
        used_strategies = set()

        n = len(df_enriched)
        for i in range(MIN_WARMUP_BARS, n):
            bar_ts = df_enriched.index[i]
            try:
                bar_time = bar_ts.time()
            except Exception:
                continue

            if bar_time >= EOD_EXIT_TIME:
                break

            spot = float(df_enriched.iloc[i]["close"])
            partial_df = df_enriched.iloc[: i + 1]

            om = self._get_oi(oi_snapshots, bar_time)

            for strat_name, strategy in self.strategies.items():
                if strat_name in used_strategies:
                    continue

                try:
                    signal = strategy.evaluate(partial_df, om, spot)
                except Exception:
                    continue

                if signal is None:
                    continue

                trade = self._simulate_trade(
                    df_enriched, i, signal, strat_name,
                    instrument_symbol, strike_interval, lot_size,
                    option_cache, date_str, day_type, day_type_hindsight,
                )
                if trade is not None:
                    trades.append(trade)
                    used_strategies.add(strat_name)

            if len(used_strategies) >= len(self.strategies):
                break

        return trades

    # ── Trade simulation with uniform exit ────────────────────────────

    def _simulate_trade(
        self, df, entry_bar_idx, signal, strat_name,
        instrument, strike_interval, lot_size,
        option_cache, date_str, day_type, day_type_hindsight,
    ) -> Optional[TradeResult]:
        """Simulate entry at signal bar, exit via uniform SL/T1/T2/EOD."""
        entry_bar = df.iloc[entry_bar_idx]
        spot = float(entry_bar["close"])
        opt_type = signal.option_type.value  # "CE" or "PE"

        strike = round(spot / strike_interval) * strike_interval
        opt_key = (instrument, date_str, float(strike), opt_type)
        opt_df = option_cache.get(opt_key)

        if opt_df is None or len(opt_df) < 10:
            return None

        entry_ts = entry_bar.name
        if hasattr(entry_ts, "tz_localize") and entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize("Asia/Kolkata")

        # Find entry in option candles
        entry_opt_idx = None
        for idx in range(len(opt_df)):
            if opt_df.index[idx] >= entry_ts:
                entry_opt_idx = idx
                break
        if entry_opt_idx is None:
            return None

        entry_price = float(opt_df.iloc[entry_opt_idx]["close"])
        if entry_price <= 0:
            return None

        # Strategy-specific entry filter: skip if option premium below minimum
        if strat_name == "RANGE_BREAKOUT" and RB_MIN_ENTRY_PREMIUM > 0 and entry_price < RB_MIN_ENTRY_PREMIUM:
            return None

        # Uniform exit: SL (1R), T1 at +1R, T2 at +2R
        # RANGE_BREAKOUT uses RB_SL_PCT (env-driven); all others use 20%
        sl_pct = RB_SL_PCT if strat_name == "RANGE_BREAKOUT" else 0.20
        one_r = entry_price * sl_pct
        sl = max(entry_price - one_r, 1.0)
        t1 = entry_price + one_r
        t2 = entry_price + 2.0 * one_r

        exit_price = entry_price
        exit_reason = "eod_close"
        exit_idx = entry_opt_idx
        t1_hit = False
        t2_hit = False

        for j in range(entry_opt_idx + 1, len(opt_df)):
            bar = opt_df.iloc[j]
            bart = opt_df.index[j]
            bar_low = float(bar["low"])
            bar_high = float(bar["high"])
            bar_close = float(bar["close"])

            try:
                bt = bart.time()
            except Exception:
                continue

            # EOD exit
            if bt >= EOD_EXIT_TIME:
                exit_price = bar_close
                exit_reason = "eod_close"
                exit_idx = j
                break

            # SL check
            if bar_low <= sl:
                exit_price = sl
                exit_reason = "trailing_sl" if (t1_hit or t2_hit) else "stoploss"
                exit_idx = j
                break

            # T2 check
            if not t2_hit and bar_high >= t2:
                t2_hit = True
                t1_hit = True
                sl = max(sl, entry_price + one_r)  # lock 1R profit
                exit_price = t2
                exit_reason = "target2"
                exit_idx = j
                break

            # T1 check — move SL to breakeven
            if not t1_hit and bar_high >= t1:
                t1_hit = True
                sl = max(sl, entry_price * 1.005)
        else:
            # Fell through all bars
            if len(opt_df) > entry_opt_idx:
                exit_price = float(opt_df.iloc[-1]["close"])
                exit_idx = len(opt_df) - 1

        pnl_per_unit = exit_price - entry_price
        pnl = pnl_per_unit * lot_size
        pnl_pct = (pnl_per_unit / entry_price * 100) if entry_price > 0 else 0
        r_multiple = pnl_per_unit / one_r if one_r > 0 else 0

        entry_time_str = entry_ts.strftime("%H:%M") if hasattr(entry_ts, "strftime") else ""
        exit_ts = opt_df.index[exit_idx]
        exit_time_str = exit_ts.strftime("%H:%M") if hasattr(exit_ts, "strftime") else ""

        try:
            hold_minutes = int((exit_ts - entry_ts).total_seconds() / 60)
        except Exception:
            hold_minutes = 0

        try:
            tw = TimeWindow.from_time(entry_ts.time())
        except Exception:
            tw = TimeWindow.MID_MORNING

        return TradeResult(
            date=date_str,
            instrument=instrument,
            strategy=strat_name,
            direction=opt_type,
            entry_time=entry_time_str,
            exit_time=exit_time_str,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            strike=strike,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
            hold_minutes=hold_minutes,
            r_multiple=round(r_multiple, 2),
            time_window=tw,
            day_type=day_type,
            day_type_hindsight=day_type_hindsight,
            lot_size=lot_size,
        )

    @staticmethod
    def _get_oi(snapshots, bar_time):
        """Get most recent OI snapshot at or before bar_time."""
        if not snapshots:
            return OptionsMetrics()
        best = None
        for snap_time, om in snapshots:
            if isinstance(snap_time, str):
                try:
                    snap_time = datetime.strptime(snap_time, "%H:%M:%S").time()
                except Exception:
                    continue
            try:
                st = snap_time if isinstance(snap_time, dtime) else snap_time.time()
            except Exception:
                continue
            if st <= bar_time:
                best = om
            else:
                break
        return best if best else OptionsMetrics()
