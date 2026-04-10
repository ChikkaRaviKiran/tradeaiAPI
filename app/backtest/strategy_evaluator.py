"""Strategy Evaluator — runs all strategies against recent historical data,
classifies each day's market conditions, and computes per-condition performance.

Designed to run:
  - Pre-market (08:00 IST) automatically via scheduler
  - Post-market (after 15:30 IST)
  - On-demand via API endpoint

Data source: IndexCandle table in PostgreSQL (populated by IndexCandleCollector).
Falls back to AngelOne API if DB has insufficient data.

Outputs:
  1. Ranked strategy recommendations (global, like before)
  2. Per-condition performance stats → stored in strategy_condition_performance table
     for the StrategySelector to use during live trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.instruments import InstrumentConfig, get_instrument
from app.core.models import DayType, MarketRegime, MarketSnapshot, OptionsMetrics, PerformanceMetrics
from app.engine.feature_engine import FeatureEngine
from app.engine.regime_detector import RegimeDetector
from app.strategies.base import BaseStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.strategies.orb import ORBStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.trading.smart_exit import SmartExitEngine, ExitResult
from app.core.models import DayType as DayTypeEnum, OptionType, StrategyName, Trade, TradeStatus, TechnicalIndicators
from app.engine.feature_engine import compute_market_structure
import app.trading.smart_exit as _se_mod
import app.trading.risk_manager as _rm_mod

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# ── All available strategies ─────────────────────────────────────────────

_STRATEGY_REGISTRY: list[tuple[str, BaseStrategy]] = [
    ("TREND_PULLBACK", TrendPullbackStrategy()),
    ("MOMENTUM_BREAKOUT", MomentumBreakoutStrategy()),
    ("ORB", ORBStrategy()),
    ("VWAP_RECLAIM", VWAPReclaimStrategy()),
    ("LIQUIDITY_SWEEP", LiquiditySweepStrategy()),
    ("RANGE_BREAKOUT", RangeBreakoutStrategy()),
]

# Multi-window weighting for blended composite score
# Prevents overfitting by balancing stability (90d) with adaptability (7d)
_MULTI_WINDOW_WEIGHTS: list[tuple[int, float]] = [
    (90, 0.50),   # 50% weight — long-term stability
    (30, 0.30),   # 30% weight — medium-term trend
    (7, 0.20),    # 20% weight — recent performance
]


@dataclass
class DayCondition:
    """Classified market conditions for a single day (used for condition tagging)."""
    date: str = ""
    gap_pct: float = 0.0
    vix: float = 0.0
    day_type: str = "unknown"       # trend/range/volatile/unclear
    gap_bucket: str = "flat"        # flat/small/medium/large
    vix_bucket: str = "unknown"     # low/medium/high/unknown
    regime: str = "unknown"
    entry_window: str = ""          # When signal fired (e.g. "09:30-10:00")

    def condition_key(self) -> str:
        return f"gap_{self.gap_bucket}_vix_{self.vix_bucket}_{self.day_type}"

    @staticmethod
    def gap_to_bucket(gap_pct: float) -> str:
        ag = abs(gap_pct)
        if ag < 0.2: return "flat"
        elif ag < 0.5: return "small"
        elif ag < 1.0: return "medium"
        return "large"

    @staticmethod
    def vix_to_bucket(vix: float) -> str:
        if vix <= 0: return "unknown"
        elif vix < 13: return "low"
        elif vix < 18: return "medium"
        return "high"


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
    condition: Optional[DayCondition] = None


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

    Uses index candles from PostgreSQL (populated by IndexCandleCollector).
    Falls back to AngelOne API if DB data is insufficient.
    Classifies each day's conditions and stores per-condition performance.
    """

    # Composite score weights
    W_WIN_RATE = 0.30
    W_PROFIT_FACTOR = 0.25
    W_SHARPE = 0.20
    W_AVG_PNL = 0.15
    W_CONSISTENCY = 0.10  # penalty for infrequent signals

    def __init__(self, lookback_days: int = 90) -> None:
        self.lookback_days = lookback_days
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def evaluate(
        self,
        instruments: list[InstrumentConfig],
        strategies: Optional[list[tuple[str, BaseStrategy]]] = None,
    ) -> EvaluationReport:
        """Run full evaluation across instruments and strategies."""
        import asyncio

        if strategies is None:
            strategies = _STRATEGY_REGISTRY

        report = await asyncio.to_thread(
            self._evaluate_sync, instruments, strategies
        )

        # Store condition performance asynchronously
        await self._store_condition_performance(report)

        return report

    def _evaluate_sync(
        self,
        instruments: list[InstrumentConfig],
        strategies: list[tuple[str, BaseStrategy]],
    ) -> EvaluationReport:
        """Synchronous evaluation — runs in a thread pool."""
        start_time = datetime.now()
        eval_date = datetime.now(_IST).strftime("%Y-%m-%d")
        report = EvaluationReport(
            eval_date=eval_date,
            instruments_evaluated=[i.symbol for i in instruments],
            strategies_evaluated=[name for name, _ in strategies],
            lookback_days=self.lookback_days,
        )

        all_recommendations: list[StrategyRecommendation] = []
        all_trades: list[EvalTrade] = []

        for instrument in instruments:
            try:
                logger.info("Evaluating %s...", instrument.symbol)

                # Fetch historical data from DB or AngelOne
                daily_data = self._fetch_historical_data(instrument)
                if not daily_data:
                    logger.warning("No historical data for %s — skipping", instrument.symbol)
                    continue

                # Classify conditions for each day
                day_conditions = self._classify_days(daily_data)

                # Detect current regime from most recent day
                current_regime = self._detect_current_regime(daily_data)

                for strat_name, strategy in strategies:
                    try:
                        rec, trades = self._evaluate_strategy(
                            instrument, strat_name, strategy, daily_data,
                            current_regime, day_conditions,
                        )
                        if rec.total_trades > 0:
                            all_recommendations.append(rec)
                            all_trades.extend(trades)
                    except Exception:
                        logger.exception(
                            "Error evaluating %s on %s", strat_name, instrument.symbol
                        )
            except Exception:
                logger.exception("Error processing instrument %s", instrument.symbol)

        # Rank by composite score
        all_recommendations.sort(key=lambda r: r.composite_score, reverse=True)
        for i, rec in enumerate(all_recommendations, 1):
            rec.rank = i

        report.recommendations = all_recommendations
        report.all_trades = all_trades
        report.run_time_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Evaluation complete: %d recommendations | %.1f seconds",
            len(all_recommendations), report.run_time_seconds,
        )

        return report

    def _fetch_historical_data(
        self, instrument: InstrumentConfig
    ) -> dict[str, pd.DataFrame]:
        """Fetch last N days of 1-min candle data from DB.

        Falls back to AngelOne API if DB has insufficient data.
        Returns dict of {date_string: DataFrame} with OHLCV columns.
        """
        import asyncio

        try:
            # Try DB first
            daily_data = asyncio.get_event_loop().run_until_complete(
                self._fetch_from_db(instrument)
            )
        except RuntimeError:
            # No event loop — create one
            daily_data = asyncio.run(self._fetch_from_db(instrument))

        if len(daily_data) >= max(5, self.lookback_days // 2):
            logger.info(
                "DB data for %s: %d days, %d total candles",
                instrument.symbol, len(daily_data),
                sum(len(v) for v in daily_data.values()),
            )
            return daily_data

        # Fallback: fetch from AngelOne API
        logger.info(
            "DB has only %d days for %s — fetching from AngelOne API",
            len(daily_data), instrument.symbol,
        )
        return self._fetch_from_angelone(instrument)

    async def _fetch_from_db(self, instrument: InstrumentConfig) -> dict[str, pd.DataFrame]:
        """Fetch candles from index_candles table."""
        end_date = datetime.now(_IST).strftime("%Y-%m-%d")
        start_date = (datetime.now(_IST) - timedelta(days=int(self.lookback_days * 1.6))).strftime("%Y-%m-%d")

        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT date, timestamp, open, high, low, close, volume "
                    "FROM index_candles "
                    "WHERE instrument = :inst AND date >= :start AND date <= :end "
                    "ORDER BY timestamp"
                ),
                {"inst": instrument.symbol, "start": start_date, "end": end_date},
            )
            rows = result.fetchall()

        if not rows:
            return {}

        df = pd.DataFrame(rows, columns=["date", "timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index = df.index.tz_localize("Asia/Kolkata") if df.index.tz is None else df.index

        daily_data: dict[str, pd.DataFrame] = {}
        for date_str, group in df.groupby("date"):
            group = group.drop(columns=["date"])
            if len(group) >= 30:  # At least 30 1-min candles
                daily_data[date_str] = group

        # Keep only last lookback_days
        dates = sorted(daily_data.keys())
        if len(dates) > self.lookback_days:
            dates = dates[-self.lookback_days:]
            daily_data = {d: daily_data[d] for d in dates}

        return daily_data

    def _fetch_from_angelone(self, instrument: InstrumentConfig) -> dict[str, pd.DataFrame]:
        """Fetch from AngelOne API as fallback when DB has insufficient data."""
        import time as time_mod
        from app.data.angelone_client import AngelOneClient

        angel = AngelOneClient()
        angel.authenticate()

        daily_data: dict[str, pd.DataFrame] = {}
        end = datetime.now(_IST).date()
        start = end - timedelta(days=int(self.lookback_days * 1.6))
        current = start

        while current <= end:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            date_str = current.strftime("%Y-%m-%d")
            try:
                candles = angel.get_candle_data(
                    instrument.token, instrument.exchange.value, "ONE_MINUTE",
                    f"{date_str} 09:15", f"{date_str} 15:30",
                )
                time_mod.sleep(0.3)

                if candles and len(candles) >= 30:
                    df = angel.candles_to_dataframe(candles)
                    if len(df) >= 30:
                        daily_data[date_str] = df
            except Exception as e:
                logger.warning("Failed to fetch %s candles for %s: %s", instrument.symbol, date_str, e)

            current += timedelta(days=1)

        # Keep only last lookback_days
        dates = sorted(daily_data.keys())
        if len(dates) > self.lookback_days:
            dates = dates[-self.lookback_days:]
            daily_data = {d: daily_data[d] for d in dates}

        logger.info(
            "AngelOne data for %s: %d trading days",
            instrument.symbol, len(daily_data),
        )
        return daily_data

    def _classify_days(self, daily_data: dict[str, pd.DataFrame]) -> dict[str, DayCondition]:
        """Classify market conditions for each day in the dataset."""
        from app.engine.day_classifier import DayClassifier

        classifier = DayClassifier()
        conditions: dict[str, DayCondition] = {}
        dates = sorted(daily_data.keys())

        prev_close = None
        for date_str in dates:
            df = daily_data[date_str].copy()
            df = self.feature_engine.compute_indicators(df)

            cond = DayCondition(date=date_str)

            # Gap % from previous day close
            if prev_close and prev_close > 0:
                day_open = float(df.iloc[0]["open"])
                cond.gap_pct = (day_open - prev_close) / prev_close * 100
            cond.gap_bucket = DayCondition.gap_to_bucket(cond.gap_pct)

            # VIX — not available in historical candles, use ATR as proxy
            atr = df.iloc[-1].get("atr", None)
            price = float(df.iloc[-1]["close"])
            if atr and price > 0:
                # Convert ATR to annualized vol proxy (rough VIX equivalent)
                implied_vix = (atr / price) * 100 * np.sqrt(252)
                cond.vix = implied_vix
            cond.vix_bucket = DayCondition.vix_to_bucket(cond.vix)

            # Day type classification
            snap = MarketSnapshot(
                instrument="NIFTY",
                price=price,
                nifty_price=price,
                indicators=self._extract_indicators(df),
            )
            if prev_close:
                snap.prev_day_close = prev_close
                snap.day_open = float(df.iloc[0]["open"])

            day_type = classifier.classify(df, snap, cond.vix if cond.vix > 0 else None)
            cond.day_type = day_type.value

            # Regime
            regime = self.regime_detector.detect(df)
            cond.regime = regime.value

            conditions[date_str] = cond
            prev_close = float(df.iloc[-1]["close"])

        return conditions

    def _extract_indicators(self, df: pd.DataFrame):
        """Extract TechnicalIndicators from the last row of a computed DataFrame."""
        from app.core.models import TechnicalIndicators
        last = df.iloc[-1]

        def _get(col):
            v = last.get(col, None)
            if v is not None and not pd.isna(v):
                return float(v)
            return None

        return TechnicalIndicators(
            ema9=_get("ema9"), ema20=_get("ema20"),
            ema50=_get("ema50"), ema200=_get("ema200"),
            vwap=_get("vwap"), rsi=_get("rsi"),
            atr=_get("atr"), adx=_get("adx"),
            bollinger_upper=_get("bollinger_upper"),
            bollinger_middle=_get("bollinger_middle"),
            bollinger_lower=_get("bollinger_lower"),
        )

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
        day_conditions: dict[str, DayCondition],
    ) -> tuple[StrategyRecommendation, list[EvalTrade]]:
        """Evaluate one strategy on one instrument over all days.

        Uses multi-window blending:
          Final Score = 50% × 90-day + 30% × 30-day + 20% × 7-day
        This prevents overfitting while keeping adaptability.
        """
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
                # Attach day condition to trade
                trade.condition = day_conditions.get(date_str)
                trades.append(trade)
                last_signal_date = date_str

        # Compute metrics from ALL trades (full window)
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
            return rec, []

        # Full-window stats for the recommendation record
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

        # Multi-window blended composite score
        # Score = 50% × 90d + 30% × 30d + 20% × 7d
        rec.composite_score = self._compute_blended_composite(trades, dates)

        return rec, trades

    def _compute_blended_composite(
        self, trades: list[EvalTrade], all_dates: list[str]
    ) -> float:
        """Compute blended composite from multiple time windows.

        Each window gets its own composite score, then they're blended:
          50% × 90-day + 30% × 30-day + 20% × 7-day
        If a window has no trades, its weight is redistributed proportionally.
        """
        if not trades or not all_dates:
            return 0.0

        today_str = datetime.now(_IST).strftime("%Y-%m-%d")
        today_dt = datetime.strptime(today_str, "%Y-%m-%d")

        blended = 0.0
        total_weight_used = 0.0

        for window_days, weight in _MULTI_WINDOW_WEIGHTS:
            cutoff = (today_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
            window_trades = [t for t in trades if t.date >= cutoff]
            window_dates = [d for d in all_dates if d >= cutoff]

            if not window_trades:
                continue

            # Build a temporary rec to score this window
            score = self._compute_window_score(window_trades, len(window_dates))
            blended += score * weight
            total_weight_used += weight

        # Redistribute unused weight (if some windows had no trades)
        if total_weight_used > 0 and total_weight_used < 1.0:
            blended = blended / total_weight_used

        return round(blended, 1)

    def _compute_window_score(
        self, trades: list[EvalTrade], total_days: int
    ) -> float:
        """Compute composite score for a specific time window of trades."""
        if not trades:
            return 0.0

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        n = len(trades)

        win_rate = (len(winners) / n * 100) if n else 0
        avg_pnl = sum(t.pnl for t in trades) / n if n else 0

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        pnls = [t.pnl for t in trades]
        sharpe = 0.0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls)) * np.sqrt(252)

        freq = n / total_days if total_days > 0 else 0

        # Same formula as _compute_composite
        wr_score = min(win_rate, 100)
        pf_score = min(profit_factor * 25, 100)
        sh_score = min(max(sharpe, 0) * 25, 100)
        ap_score = min(max(avg_pnl / 50 + 50, 0), 100)
        cs_score = min(freq * 100, 100)

        composite = (
            wr_score * self.W_WIN_RATE
            + pf_score * self.W_PROFIT_FACTOR
            + sh_score * self.W_SHARPE
            + ap_score * self.W_AVG_PNL
            + cs_score * self.W_CONSISTENCY
        )

        if n < 3:
            composite *= 0.5
        elif n < 5:
            composite *= 0.75

        return composite

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

        Exit simulation priority:
          1. If DB has option candles for the signal's strike → use SmartExitEngine
             (production-faithful, same as live trading).
          2. Otherwise → fall back to spot-based SL/T1/T2 proxy.
        """
        n = len(df)
        scan_start = 30   # warmup bars
        scan_end = max(scan_start + 1, n - 6)  # 6-bar buffer before close

        for i in range(scan_start, scan_end):
            partial_df = df.iloc[: i + 1].copy()
            spot = partial_df.iloc[-1]["close"]

            signal = strategy.evaluate(partial_df, options_metrics, spot)
            if signal is None:
                continue

            # ── Signal fired ──────────────────────────────────────────
            entry_bar = df.iloc[i]
            spot_price = float(entry_bar["close"])
            atr = entry_bar.get("atr", 0)
            if atr is None or atr <= 0:
                atr = spot_price * 0.005
            option_atr = atr * 0.5

            opt_type = signal.option_type.value  # "CE" or "PE"

            # Compute ATM strike
            strike = round(spot_price / instrument.strike_interval) * instrument.strike_interval

            # Try option candle exit (production-faithful)
            opt_df = self._load_option_candles_sync(instrument, date_str, strike, opt_type)
            if opt_df is not None and len(opt_df) >= 10:
                result = self._simulate_exit_with_smart_engine(
                    instrument, opt_df, df, entry_bar, strike, opt_type,
                    strat_name, date_str, option_atr,
                )
                if result is not None:
                    return result

            # ── Fallback: spot-based proxy simulation ─────────────────
            # Match production orchestrator: SL = 2.0× ATR, floor 25% max loss
            # T1 = 2.5× ATR, T2 = 4.0× ATR
            sl = round(max(spot_price - 2.0 * option_atr, spot_price * 0.75), 2)
            t1 = round(spot_price + 2.5 * option_atr, 2)
            t2 = round(spot_price + 4.0 * option_atr, 2)

            exit_price = spot_price
            exit_reason = "eod_close"

            for j in range(i + 1, n):
                bar = df.iloc[j]
                low, high = bar["low"], bar["high"]

                if opt_type == "CE":
                    if low <= spot_price - (spot_price - sl):
                        exit_price, exit_reason = sl, "stoploss"; break
                    if high >= spot_price + (t2 - spot_price):
                        exit_price, exit_reason = t2, "target2"; break
                    if high >= spot_price + (t1 - spot_price):
                        exit_price, exit_reason = t1, "target1"; break
                else:
                    if high >= spot_price + (spot_price - sl):
                        exit_price, exit_reason = sl, "stoploss"; break
                    if low <= spot_price - (t2 - spot_price):
                        exit_price, exit_reason = t2, "target2"; break
                    if low <= spot_price - (t1 - spot_price):
                        exit_price, exit_reason = t1, "target1"; break

            if exit_reason == "eod_close":
                last_close = df.iloc[-1]["close"]
                if opt_type == "CE":
                    movement = (last_close - spot_price)
                    exit_price = spot_price + movement * 0.5
                else:
                    movement = (spot_price - last_close)
                    exit_price = spot_price + movement * 0.5

            pnl = (exit_price - spot_price) * instrument.lot_size
            pnl_pct = ((exit_price - spot_price) / spot_price * 100) if spot_price > 0 else 0

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
                option_type=opt_type,
                entry_time=entry_time,
                entry_price=round(spot_price, 2),
                exit_price=round(exit_price, 2),
                stoploss=sl,
                target1=t1,
                target2=t2,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                exit_reason=exit_reason,
            )

        return None  # No signal this day

    # ── Option candle helpers ─────────────────────────────────────────────

    def _load_option_candles_sync(
        self,
        instrument: InstrumentConfig,
        date_str: str,
        strike: float,
        opt_type: str,
    ) -> Optional[pd.DataFrame]:
        """Load option candles from DB for a specific strike/date. Synchronous wrapper."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — use thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run, self._load_option_candles(instrument, date_str, strike, opt_type)
                    ).result(timeout=10)
            return loop.run_until_complete(
                self._load_option_candles(instrument, date_str, strike, opt_type)
            )
        except RuntimeError:
            return asyncio.run(self._load_option_candles(instrument, date_str, strike, opt_type))
        except Exception:
            return None

    async def _load_option_candles(
        self,
        instrument: InstrumentConfig,
        date_str: str,
        strike: float,
        opt_type: str,
    ) -> Optional[pd.DataFrame]:
        """Load 1-min option candles from the option_candles DB table."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT timestamp, open, high, low, close, volume "
                    "FROM option_candles "
                    "WHERE instrument = :inst AND date = :dt "
                    "  AND strike = :strike AND option_type = :ot "
                    "ORDER BY timestamp"
                ),
                {"inst": instrument.symbol, "dt": date_str, "strike": strike, "ot": opt_type},
            )
            rows = result.fetchall()

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        return df

    def _simulate_exit_with_smart_engine(
        self,
        instrument: InstrumentConfig,
        opt_df: pd.DataFrame,
        spot_df: pd.DataFrame,
        entry_bar,
        strike: float,
        opt_type: str,
        strat_name: str,
        date_str: str,
        option_atr: float,
    ) -> Optional[EvalTrade]:
        """Simulate trade exit using SmartExitEngine.evaluate() on real option candles.

        This is production-faithful: same engine, same exit hierarchy.
        datetime.now() is monkey-patched per bar so grace period / time exits work.
        """
        from datetime import time as dtime

        smart_exit = SmartExitEngine()

        # Build entry from the option candle at signal time
        entry_ts = entry_bar.name
        if hasattr(entry_ts, 'tz_localize') and entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize("Asia/Kolkata")

        # Find option candle at or after entry time
        entry_opt_idx = None
        for idx in range(len(opt_df)):
            if opt_df.index[idx] >= entry_ts:
                entry_opt_idx = idx
                break
        if entry_opt_idx is None:
            return None

        entry_opt_price = float(opt_df.iloc[entry_opt_idx]["close"])
        if entry_opt_price <= 0:
            return None

        # ATR-based SL/targets — match production orchestrator
        # SL = 2.0× ATR, floor 25% max loss; T1 = 2.5× ATR, T2 = 4.0× ATR
        sl = round(max(entry_opt_price - 2.0 * option_atr, entry_opt_price * 0.75), 2)
        t1 = round(entry_opt_price + 2.5 * option_atr, 2)
        t2 = round(entry_opt_price + 4.0 * option_atr, 2)

        # Build Trade object (same schema as production)
        entry_dt = opt_df.index[entry_opt_idx].to_pydatetime()
        if entry_dt.tzinfo is None:
            entry_dt = _IST.localize(entry_dt)

        trade = Trade(
            trade_id=f"EVAL-{date_str}-{strat_name[:4]}",
            instrument=instrument.symbol,
            engine="v2",
            date=date_str,
            time=entry_dt.strftime("%H:%M:%S"),
            symbol=f"{instrument.symbol}{int(strike)}{opt_type}",
            strike=strike,
            option_type=OptionType.CALL if opt_type == "CE" else OptionType.PUT,
            strategy=StrategyName(strat_name),
            entry_price=entry_opt_price,
            stoploss=sl,
            target1=t1,
            target2=t2,
            status=TradeStatus.OPEN,
            lot_size=instrument.lot_size,
            entry_datetime=entry_dt,
            max_hold_minutes=settings.v2_max_hold_minutes,
        )

        # Day type from spot data
        day_type = DayTypeEnum.UNCLEAR

        original_dt_se = _se_mod.datetime
        original_dt_rm = _rm_mod.datetime

        try:
            for idx in range(entry_opt_idx + 1, len(opt_df)):
                bar = opt_df.iloc[idx]
                bar_time = opt_df.index[idx]
                ltp = float(bar["close"])
                bar_high = float(bar["high"])

                # Get spot price at this timestamp
                spot_price = 0.0
                rsi_val = 50.0
                candidates = spot_df[spot_df.index <= bar_time]
                if len(candidates) > 0:
                    spot_row = candidates.iloc[-1]
                    spot_price = float(spot_row["close"])
                    rsi_raw = spot_row.get("rsi")
                    if rsi_raw is not None and not pd.isna(rsi_raw):
                        rsi_val = float(rsi_raw)

                snap = MarketSnapshot(
                    instrument=instrument.symbol,
                    price=spot_price,
                    indicators=TechnicalIndicators(rsi=rsi_val),
                )

                # Monkey-patch datetime.now() for this bar
                bar_dt = bar_time.to_pydatetime()
                if bar_dt.tzinfo is None:
                    bar_dt = _IST.localize(bar_dt)

                class _FakeDatetime(type(original_dt_se)):
                    @classmethod
                    def now(cls, tz=None):
                        return bar_dt

                _se_mod.datetime = _FakeDatetime
                _rm_mod.datetime = _FakeDatetime

                result = smart_exit.evaluate(
                    trade=trade,
                    current_ltp=ltp,
                    snap=snap,
                    day_type=day_type,
                    spot_price=spot_price,
                    candle_closed=True,
                    option_atr=option_atr,
                )

                _se_mod.datetime = original_dt_se
                _rm_mod.datetime = original_dt_rm

                # Target check on bar high (production checks on tick)
                if not result.should_exit:
                    if bar_high >= trade.target2:
                        result = ExitResult(
                            should_exit=True, exit_type="target2",
                            exit_price=trade.target2,
                            reason=f"T2 hit: high={bar_high:.2f}",
                        )
                    elif bar_high >= trade.target1:
                        result = ExitResult(
                            should_exit=True, exit_type="target1",
                            exit_price=trade.target1,
                            reason=f"T1 hit: high={bar_high:.2f}",
                        )

                # Update trailing SL
                if result.new_stoploss is not None and not result.should_exit:
                    trade.stoploss = result.new_stoploss

                if result.should_exit:
                    exit_price = result.exit_price
                    pnl = (exit_price - entry_opt_price) * instrument.lot_size
                    pnl_pct = ((exit_price - entry_opt_price) / entry_opt_price * 100) if entry_opt_price > 0 else 0
                    return EvalTrade(
                        date=date_str,
                        instrument=instrument.symbol,
                        strategy=strat_name,
                        option_type=opt_type,
                        entry_time=str(entry_dt),
                        entry_price=round(entry_opt_price, 2),
                        exit_price=round(exit_price, 2),
                        stoploss=sl,
                        target1=t1,
                        target2=t2,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        exit_reason=result.exit_type,
                    )

                # EOD forced close
                bt = bar_time.time() if hasattr(bar_time, 'time') else bar_time.to_pydatetime().time()
                if bt >= dtime(15, 10):
                    exit_price = ltp
                    pnl = (exit_price - entry_opt_price) * instrument.lot_size
                    pnl_pct = ((exit_price - entry_opt_price) / entry_opt_price * 100) if entry_opt_price > 0 else 0
                    return EvalTrade(
                        date=date_str,
                        instrument=instrument.symbol,
                        strategy=strat_name,
                        option_type=opt_type,
                        entry_time=str(entry_dt),
                        entry_price=round(entry_opt_price, 2),
                        exit_price=round(exit_price, 2),
                        stoploss=sl,
                        target1=t1,
                        target2=t2,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        exit_reason="eod_close",
                    )

        finally:
            _se_mod.datetime = original_dt_se
            _rm_mod.datetime = original_dt_rm

        # If we ran out of option candles, use last close
        last_ltp = float(opt_df.iloc[-1]["close"])
        pnl = (last_ltp - entry_opt_price) * instrument.lot_size
        pnl_pct = ((last_ltp - entry_opt_price) / entry_opt_price * 100) if entry_opt_price > 0 else 0
        return EvalTrade(
            date=date_str,
            instrument=instrument.symbol,
            strategy=strat_name,
            option_type=opt_type,
            entry_time=str(entry_dt),
            entry_price=round(entry_opt_price, 2),
            exit_price=round(last_ltp, 2),
            stoploss=sl,
            target1=t1,
            target2=t2,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason="eod_close",
        )

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

    async def _store_condition_performance(self, report: EvaluationReport) -> None:
        """Compute per-condition performance stats and store to DB.

        Groups all eval trades by (strategy, instrument, condition_key) and
        computes win_rate, profit_factor, avg_pnl, etc. for each group.
        Also generates partial/aggregate keys for cascading lookups.
        """
        from app.db.models import StrategyConditionPerformance

        # Collect all trades with conditions from all recommendations
        all_trades: list[EvalTrade] = report.all_trades

        # Group trades by (instrument, strategy, condition_key)
        from collections import defaultdict
        groups: dict[tuple[str, str, str], list[EvalTrade]] = defaultdict(list)

        for trade in all_trades:
            if trade.condition is None:
                continue
            key = (trade.instrument, trade.strategy, trade.condition.condition_key())
            groups[key].append(trade)

            # Also add to partial/aggregate keys for cascading fallback
            cond = trade.condition
            partial_keys = [
                f"gap_{cond.gap_bucket}_vix_{cond.vix_bucket}_any",
                f"gap_{cond.gap_bucket}_any_{cond.day_type}",
                f"any_vix_{cond.vix_bucket}_{cond.day_type}",
                f"gap_{cond.gap_bucket}_any_any",
                f"any_vix_{cond.vix_bucket}_any",
                f"any_any_{cond.day_type}",
                "any_any_any",
            ]
            for pk in partial_keys:
                groups[(trade.instrument, trade.strategy, pk)].append(trade)

        if not groups:
            logger.info("No condition-tagged trades to store")
            return

        eval_date = report.eval_date

        try:
            async with self._session_factory() as session:
                # Clear old records for today's eval
                await session.execute(
                    text("DELETE FROM strategy_condition_performance WHERE eval_date = :dt"),
                    {"dt": eval_date},
                )

                for (instrument, strategy, cond_key), trades in groups.items():
                    if len(trades) < 1:
                        continue

                    winners = [t for t in trades if t.pnl > 0]
                    losers = [t for t in trades if t.pnl <= 0]
                    total = len(trades)
                    win_rate = len(winners) / total * 100 if total else 0
                    avg_pnl = sum(t.pnl for t in trades) / total if total else 0

                    gross_profit = sum(t.pnl for t in winners) if winners else 0
                    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

                    pnls = [t.pnl for t in trades]
                    sharpe = 0.0
                    if len(pnls) > 1 and np.std(pnls) > 0:
                        sharpe = float(np.mean(pnls) / np.std(pnls)) * np.sqrt(252)

                    # Max drawdown
                    equity, peak, max_dd = 0.0, 0.0, 0.0
                    for t in trades:
                        equity += t.pnl
                        if equity > peak: peak = equity
                        dd = peak - equity
                        if dd > max_dd: max_dd = dd

                    # Best entry window
                    entry_times = [t.entry_time for t in trades if t.entry_time and t.pnl > 0]
                    best_window = self._compute_best_window(entry_times)

                    # Avg risk-reward ratio
                    rr_ratios = []
                    for t in trades:
                        risk = abs(t.entry_price - t.stoploss) if t.stoploss else 0
                        reward = abs(t.exit_price - t.entry_price)
                        if risk > 0:
                            rr_ratios.append(reward / risk)
                    avg_rr = np.mean(rr_ratios) if rr_ratios else 0.0

                    # Composite score (reuse same formula)
                    wr_score = min(win_rate, 100)
                    pf_score = min(profit_factor * 25, 100)
                    sh_score = min(max(sharpe, 0) * 25, 100)
                    ap_score = min(max(avg_pnl / 50 + 50, 0), 100)
                    freq = total / max(self.lookback_days, 1)
                    cs_score = min(freq * 100, 100)

                    composite = (
                        wr_score * self.W_WIN_RATE
                        + pf_score * self.W_PROFIT_FACTOR
                        + sh_score * self.W_SHARPE
                        + ap_score * self.W_AVG_PNL
                        + cs_score * self.W_CONSISTENCY
                    )
                    if total < 3:
                        composite *= 0.5
                    elif total < 5:
                        composite *= 0.75

                    # Extract buckets from condition key
                    parts = cond_key.split("_")
                    gap_bucket = parts[1] if len(parts) > 1 else None
                    vix_bucket = parts[3] if len(parts) > 3 else None
                    day_type = parts[4] if len(parts) > 4 else None

                    record = StrategyConditionPerformance(
                        eval_date=eval_date,
                        instrument=instrument,
                        strategy=strategy,
                        condition_key=cond_key,
                        day_type=day_type,
                        gap_bucket=gap_bucket,
                        vix_bucket=vix_bucket,
                        total_trades=total,
                        win_rate=round(win_rate, 1),
                        avg_pnl=round(avg_pnl, 2),
                        profit_factor=round(profit_factor, 2),
                        avg_rr=round(avg_rr, 2),
                        max_drawdown=round(max_dd, 2),
                        sharpe_ratio=round(sharpe, 2),
                        best_entry_window=best_window,
                        composite_score=round(composite, 1),
                        probability=round(win_rate, 1),  # Probability = win_rate for this condition
                        lookback_days=self.lookback_days,
                    )
                    session.add(record)

                await session.commit()
                logger.info("Stored %d condition-performance records", len(groups))

        except Exception:
            logger.exception("Error storing condition performance")

    def _compute_best_window(self, entry_times: list[str]) -> str:
        """Determine the best entry time window from winning trades."""
        if not entry_times:
            return ""

        # Parse entry times and bucket into 30-min windows
        from collections import Counter
        windows = Counter()
        for et in entry_times:
            try:
                # entry_time may be ISO format or just HH:MM
                if "T" in et:
                    dt = datetime.fromisoformat(et.replace("+05:30", ""))
                    h, m = dt.hour, dt.minute
                else:
                    parts = et.split(":")
                    h, m = int(parts[0]), int(parts[1])
                bucket_m = (m // 30) * 30
                windows[f"{h:02d}:{bucket_m:02d}-{h:02d}:{bucket_m+30:02d}"] += 1
            except Exception:
                continue

        if not windows:
            return ""
        return windows.most_common(1)[0][0]
