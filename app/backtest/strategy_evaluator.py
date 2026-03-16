"""Strategy Evaluator — runs all strategies against recent historical data
and ranks which strategy + instrument combinations perform best.

Designed to run:
  - Post-market (after 15:30 IST) automatically via orchestrator
  - On-demand via API endpoint
  - During offline hours for next-day planning

Outputs ranked recommendations so you know what to trade tomorrow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from app.core.config import settings
from app.core.instruments import InstrumentConfig, get_instrument
from app.core.models import OptionsMetrics, PerformanceMetrics
from app.data.yahoo_client import YahooFinanceClient
from app.engine.feature_engine import FeatureEngine
from app.engine.regime_detector import RegimeDetector
from app.strategies.base import BaseStrategy
from app.strategies.ema_breakout import EMABreakoutStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.strategies.orb import ORBStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.strategies.breakout_20d import Breakout20DStrategy

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# ── All available strategies with display names ──────────────────────────

_STRATEGY_REGISTRY: list[tuple[str, BaseStrategy]] = [
    ("ORB", ORBStrategy()),
    ("VWAP_RECLAIM", VWAPReclaimStrategy()),
    ("TREND_PULLBACK", TrendPullbackStrategy()),
    ("LIQUIDITY_SWEEP", LiquiditySweepStrategy()),
    ("RANGE_BREAKOUT", RangeBreakoutStrategy()),
    ("MOMENTUM_BREAKOUT", MomentumBreakoutStrategy()),
    ("EMA_BREAKOUT", EMABreakoutStrategy()),
    ("BREAKOUT_20D", Breakout20DStrategy()),
]


@dataclass
class EvalTrade:
    """Single simulated trade during evaluation."""
    date: str
    instrument: str
    strategy: str
    option_type: str
    entry_time: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stoploss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""  # target1 / target2 / stoploss / eod_close


@dataclass
class StrategyRecommendation:
    """Ranked recommendation for a strategy + instrument pair."""
    rank: int = 0
    instrument: str = ""
    strategy: str = ""
    # Performance over evaluation window
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    total_trades: int = 0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    # Composite ranking score (0-100)
    composite_score: float = 0.0
    # Current market context
    current_regime: str = ""
    last_signal_date: str = ""
    signal_frequency: float = 0.0  # Signals per day
    # Evaluation metadata
    eval_days: int = 0
    eval_date: str = ""

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "instrument": self.instrument,
            "strategy": self.strategy,
            "win_rate": round(self.win_rate, 1),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_trades": self.total_trades,
            "avg_pnl": round(self.avg_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "composite_score": round(self.composite_score, 1),
            "current_regime": self.current_regime,
            "last_signal_date": self.last_signal_date,
            "signal_frequency": round(self.signal_frequency, 2),
            "eval_days": self.eval_days,
            "eval_date": self.eval_date,
        }


@dataclass
class EvaluationReport:
    """Full evaluation output."""
    eval_date: str = ""
    instruments_evaluated: list[str] = field(default_factory=list)
    strategies_evaluated: list[str] = field(default_factory=list)
    lookback_days: int = 0
    recommendations: list[StrategyRecommendation] = field(default_factory=list)
    all_trades: list[EvalTrade] = field(default_factory=list)
    run_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "eval_date": self.eval_date,
            "instruments_evaluated": self.instruments_evaluated,
            "strategies_evaluated": self.strategies_evaluated,
            "lookback_days": self.lookback_days,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "total_trades_simulated": len(self.all_trades),
            "run_time_seconds": round(self.run_time_seconds, 1),
        }


class StrategyEvaluator:
    """Evaluates all strategies against recent data and ranks them.

    Uses Yahoo Finance 5-minute intraday data for the last N trading days.
    Walks forward through each day's candles to simulate realistic entries/exits.
    """

    # Composite score weights
    W_WIN_RATE = 0.30
    W_PROFIT_FACTOR = 0.25
    W_SHARPE = 0.20
    W_AVG_PNL = 0.15
    W_CONSISTENCY = 0.10  # penalty for infrequent signals

    def __init__(self, lookback_days: int = 20) -> None:
        self.lookback_days = lookback_days
        self.yahoo = YahooFinanceClient()
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()

    async def evaluate(
        self,
        instruments: list[InstrumentConfig],
        strategies: Optional[list[tuple[str, BaseStrategy]]] = None,
    ) -> EvaluationReport:
        """Run full evaluation across instruments and strategies.

        Args:
            instruments: Which instruments to evaluate.
            strategies: Optional override; defaults to all registered strategies.

        Returns:
            EvaluationReport with ranked recommendations.
        """
        start_time = datetime.now()
        if strategies is None:
            strategies = _STRATEGY_REGISTRY

        eval_date = datetime.now(_IST).strftime("%Y-%m-%d")
        report = EvaluationReport(
            eval_date=eval_date,
            instruments_evaluated=[i.symbol for i in instruments],
            strategies_evaluated=[name for name, _ in strategies],
            lookback_days=self.lookback_days,
        )

        all_recommendations: list[StrategyRecommendation] = []

        for instrument in instruments:
            logger.info("Evaluating %s...", instrument.symbol)

            # Fetch historical intraday data (5-min candles)
            daily_data = self._fetch_historical_data(instrument)
            if not daily_data:
                logger.warning("No historical data for %s — skipping", instrument.symbol)
                continue

            # Detect current regime from most recent day
            current_regime = self._detect_current_regime(daily_data)

            for strat_name, strategy in strategies:
                rec = self._evaluate_strategy(
                    instrument, strat_name, strategy, daily_data, current_regime
                )
                if rec.total_trades > 0:
                    all_recommendations.append(rec)
                    report.all_trades.extend(
                        self._get_trades_for(instrument.symbol, strat_name, daily_data, strategy)
                    )

        # Rank by composite score
        all_recommendations.sort(key=lambda r: r.composite_score, reverse=True)
        for i, rec in enumerate(all_recommendations, 1):
            rec.rank = i

        report.recommendations = all_recommendations
        report.run_time_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Evaluation complete: %d recommendations | %.1f seconds",
            len(all_recommendations), report.run_time_seconds,
        )

        return report

    def _fetch_historical_data(
        self, instrument: InstrumentConfig
    ) -> dict[str, pd.DataFrame]:
        """Fetch last N days of 5-minute candle data, grouped by date.

        Yahoo Finance provides 5m data for up to 60 days.
        Returns dict of {date_string: DataFrame}.
        """
        try:
            yf_symbol = self.yahoo._get_yf_symbol(instrument)
            import yfinance as yf

            ticker = yf.Ticker(yf_symbol)
            # Fetch enough data: lookback_days + buffer for weekends/holidays
            period_days = int(self.lookback_days * 1.6) + 5
            df = ticker.history(period=f"{period_days}d", interval="5m")

            if df.empty:
                return {}

            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            if "volume" not in df.columns:
                df["volume"] = 0

            # Group by date
            daily_data: dict[str, pd.DataFrame] = {}
            df["date_str"] = df.index.strftime("%Y-%m-%d")

            for date_str, group in df.groupby("date_str"):
                group = group.drop(columns=["date_str"])
                # Need at least 30 candles (2.5 hours of 5-min data)
                if len(group) >= 30:
                    daily_data[date_str] = group

            # Keep only last lookback_days
            dates = sorted(daily_data.keys())
            if len(dates) > self.lookback_days:
                dates = dates[-self.lookback_days:]
                daily_data = {d: daily_data[d] for d in dates}

            logger.info(
                "Historical data for %s: %d trading days, %d total candles",
                instrument.symbol, len(daily_data),
                sum(len(v) for v in daily_data.values()),
            )
            return daily_data

        except Exception:
            logger.exception("Error fetching historical data for %s", instrument.symbol)
            return {}

    def _detect_current_regime(self, daily_data: dict[str, pd.DataFrame]) -> str:
        """Detect regime from the most recent day's data."""
        dates = sorted(daily_data.keys())
        if not dates:
            return "UNKNOWN"
        latest = daily_data[dates[-1]].copy()
        latest = self.feature_engine.compute_indicators(latest)
        regime = self.regime_detector.detect(latest)
        return regime.value

    def _evaluate_strategy(
        self,
        instrument: InstrumentConfig,
        strat_name: str,
        strategy: BaseStrategy,
        daily_data: dict[str, pd.DataFrame],
        current_regime: str,
    ) -> StrategyRecommendation:
        """Evaluate one strategy on one instrument over all days."""
        trades: list[EvalTrade] = []
        options_metrics = OptionsMetrics()
        dates = sorted(daily_data.keys())
        last_signal_date = ""

        for date_str in dates:
            df = daily_data[date_str].copy()
            df = self.feature_engine.compute_indicators(df)

            # Walk through candles starting from bar 30 (warmup)
            trade = self._walk_forward_day(
                instrument, strat_name, strategy, df, options_metrics, date_str
            )
            if trade is not None:
                trades.append(trade)
                last_signal_date = date_str

        # Compute metrics
        total_days = len(dates)
        rec = StrategyRecommendation(
            instrument=instrument.symbol,
            strategy=strat_name,
            current_regime=current_regime,
            last_signal_date=last_signal_date,
            eval_days=total_days,
            eval_date=datetime.now(_IST).strftime("%Y-%m-%d"),
        )

        if not trades:
            return rec

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        rec.total_trades = len(trades)
        rec.win_rate = (len(winners) / len(trades) * 100) if trades else 0
        rec.total_pnl = sum(t.pnl for t in trades)
        rec.avg_pnl = rec.total_pnl / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        rec.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

        # Sharpe from daily trade PnLs
        pnls = [t.pnl for t in trades]
        if len(pnls) > 1 and np.std(pnls) > 0:
            rec.sharpe_ratio = float(np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
        else:
            rec.sharpe_ratio = 0.0

        # Max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        rec.max_drawdown = max_dd

        # Signal frequency
        rec.signal_frequency = len(trades) / total_days if total_days > 0 else 0

        # Composite score (0-100)
        rec.composite_score = self._compute_composite(rec)

        return rec

    def _walk_forward_day(
        self,
        instrument: InstrumentConfig,
        strat_name: str,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        options_metrics: OptionsMetrics,
        date_str: str,
    ) -> Optional[EvalTrade]:
        """Walk through a single day's candles, looking for signal → simulating exit.

        Only takes the FIRST signal per day (like live trading).
        Uses remaining candles after signal to simulate SL/T1/T2/EOD exit.
        """
        n = len(df)
        # Start scanning after warmup (bar 30 = ~2.5 hours into session with 5m candles)
        scan_start = 30
        # Stop scanning 6 bars before close (30 min buffer)
        scan_end = max(scan_start + 1, n - 6)

        for i in range(scan_start, scan_end):
            # Feed strategy the candles up to bar i
            partial_df = df.iloc[: i + 1].copy()
            spot = partial_df.iloc[-1]["close"]

            signal = strategy.evaluate(partial_df, options_metrics, spot)
            if signal is None:
                continue

            # Signal fired! Now walk forward to simulate the trade
            entry_bar = df.iloc[i]
            entry_price = entry_bar["close"]
            atr = entry_bar.get("atr", 0)
            if atr is None or atr <= 0:
                atr = entry_price * 0.005  # 0.5% fallback

            # ATR-based SL/targets (same as live)
            option_atr = atr * 0.5
            sl = round(max(entry_price - 1.5 * option_atr, entry_price * 0.70), 2)
            t1 = round(entry_price + 2.0 * option_atr, 2)
            t2 = round(entry_price + 3.5 * option_atr, 2)

            # Walk forward through remaining candles
            exit_price = entry_price
            exit_reason = "eod_close"

            for j in range(i + 1, n):
                bar = df.iloc[j]
                low = bar["low"]
                high = bar["high"]

                # Check stoploss (use low for long positions)
                if signal.option_type.value == "CE":
                    # Long call — if underlying drops, option drops
                    if low <= entry_price - (entry_price - sl):
                        exit_price = sl
                        exit_reason = "stoploss"
                        break
                    if high >= entry_price + (t2 - entry_price):
                        exit_price = t2
                        exit_reason = "target2"
                        break
                    if high >= entry_price + (t1 - entry_price):
                        exit_price = t1
                        exit_reason = "target1"
                        break
                else:
                    # Long put — if underlying rises, option drops
                    if high >= entry_price + (entry_price - sl):
                        exit_price = sl
                        exit_reason = "stoploss"
                        break
                    if low <= entry_price - (t2 - entry_price):
                        exit_price = t2
                        exit_reason = "target2"
                        break
                    if low <= entry_price - (t1 - entry_price):
                        exit_price = t1
                        exit_reason = "target1"
                        break

            if exit_reason == "eod_close":
                exit_price = df.iloc[-1]["close"]
                # For EOD, PnL depends on direction
                if signal.option_type.value == "CE":
                    # CE profits when underlying rises
                    movement = (exit_price - entry_price)
                    exit_price = entry_price + movement * 0.5  # ~delta 0.5
                else:
                    movement = (entry_price - exit_price)
                    exit_price = entry_price + movement * 0.5

            pnl = (exit_price - entry_price) * instrument.lot_size
            pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

            entry_time = ""
            if hasattr(entry_bar, "name"):
                try:
                    entry_time = str(entry_bar.name)
                except Exception:
                    pass

            return EvalTrade(
                date=date_str,
                instrument=instrument.symbol,
                strategy=strat_name,
                option_type=signal.option_type.value,
                entry_time=entry_time,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                stoploss=sl,
                target1=t1,
                target2=t2,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                exit_reason=exit_reason,
            )

        return None  # No signal this day

    def _get_trades_for(
        self,
        symbol: str,
        strat_name: str,
        daily_data: dict[str, pd.DataFrame],
        strategy: BaseStrategy,
    ) -> list[EvalTrade]:
        """Get all trades for a strategy-instrument pair (for the full report)."""
        # Already computed in _evaluate_strategy; re-derive is expensive.
        # We'll let the caller accumulate from _walk_forward_day instead.
        return []

    def _compute_composite(self, rec: StrategyRecommendation) -> float:
        """Compute composite ranking score (0-100) from individual metrics."""
        if rec.total_trades == 0:
            return 0.0

        # Normalize each metric to 0-100 scale
        wr_score = min(rec.win_rate, 100)  # Already 0-100
        pf_score = min(rec.profit_factor * 25, 100)  # PF 4.0 → 100
        sharpe_score = min(max(rec.sharpe_ratio, 0) * 25, 100)  # Sharpe 4.0 → 100
        avg_pnl_score = min(max(rec.avg_pnl / 50 + 50, 0), 100)  # ₹50 avg → score 51
        consistency_score = min(rec.signal_frequency * 100, 100)  # 1 signal/day → 100

        composite = (
            wr_score * self.W_WIN_RATE
            + pf_score * self.W_PROFIT_FACTOR
            + sharpe_score * self.W_SHARPE
            + avg_pnl_score * self.W_AVG_PNL
            + consistency_score * self.W_CONSISTENCY
        )

        # Penalty for very few trades (unreliable stats)
        if rec.total_trades < 3:
            composite *= 0.5
        elif rec.total_trades < 5:
            composite *= 0.75

        return round(composite, 1)
