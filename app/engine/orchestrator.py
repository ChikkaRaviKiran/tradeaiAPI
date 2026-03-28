"""Main orchestrator — runs the continuous trading loop during market hours.

Multi-instrument architecture (SRS rebuild):
    - Loops over all active instruments from config
    - Each instrument gets independent analysis, regime detection, signal scoring
    - Shared risk management across all instruments
    - Supports indices (via AngelOne) and equities (via Yahoo Finance fallback)

Schedule:
    08:45  Load data, authenticate
    09:00  Fetch global market context + institutional/breadth data
    09:15  Start monitoring
    09:15–15:30  Continuous analysis (1-min loop per instrument)
    15:20  Close open trades
    15:30–16:00  Daily report
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, time as dtime, timedelta
from typing import Optional

import pandas as pd
import pytz

from app.alerts.alert_manager import AlertManager
from app.core.config import settings
from app.core.holidays import is_market_holiday, next_trading_date, previous_trading_date
from app.core.instruments import (
    InstrumentConfig,
    get_instrument,
    get_enabled_instruments,
)
from app.core.models import (
    GlobalBias,
    MarketRegime,
    MarketSnapshot,
    OptionsMetrics,
    StrategySignal,
    TechnicalIndicators,
    DayType,
)
from app.data.angelone_client import AngelOneClient
from app.data.global_markets import compute_global_bias, fetch_global_indices
from app.data.validator import DataValidator
from app.engine.ai_decision import AIDecisionEngine
from app.engine.feature_engine import FeatureEngine
from app.engine.regime_detector import RegimeDetector
from app.engine.signal_scorer import SignalScorer, MIN_SCORE, compute_adaptive_min_score, set_insight_manager, lock_session_threshold
from app.strategies.base import BaseStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.strategies.ema_breakout import EMABreakoutStrategy
from app.strategies.orb import ORBStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.strategies.breakout_20d import Breakout20DStrategy
from app.strategies.gex_bounce import GEXBounceStrategy
from app.strategies.rsi_extreme import RSIExtremeStrategy
from app.strategies.vwap_pullback import VWAPPullbackStrategy
from app.trading.history_logger import HistoryLogger
from app.trading.paper_trader import PaperTradingEngine
from app.trading.risk_manager import RiskManager
from app.trading.smart_exit import SmartExitEngine
from app.trading.trade_logger import TradeLogger
from app.backtest.scheduler import EvaluationScheduler
from app.ai.pre_market_analyst import PreMarketAnalyst
from app.ai.insight_manager import InsightManager
from app.engine.ai_decision import set_ai_insight_manager
from app.engine.day_classifier import DayClassifier
from app.engine.gex_calculator import GEXCalculator
from app.execution.angelone_broker import AngelOneBroker
from app.execution.broker_base import OrderRequest, OrderSide, OrderType, ProductType

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
PRE_CLOSE = dtime(15, 20)
REPORT_TIME = dtime(15, 30)
GLOBAL_FETCH_TIME = dtime(9, 0)
LOAD_TIME = dtime(8, 45)
NO_NEW_ENTRY_AFTER = dtime(14, 30)  # No new trades in last hour — high chop zone

LOOP_INTERVAL_SECONDS = 60  # 1-minute analysis cycle

# Regime-strategy compatibility map
# Strategies not listed here are allowed in all regimes
_REGIME_STRATEGY_COMPAT: dict[str, set[MarketRegime]] = {
    "ORBStrategy": {MarketRegime.TRENDING, MarketRegime.HIGH_VOLATILITY},
    "MomentumBreakoutStrategy": {MarketRegime.TRENDING, MarketRegime.HIGH_VOLATILITY},
    "EMABreakoutStrategy": {MarketRegime.TRENDING, MarketRegime.HIGH_VOLATILITY},
    "Breakout20DStrategy": {MarketRegime.TRENDING, MarketRegime.HIGH_VOLATILITY},
    "RangeBreakoutStrategy": {MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY, MarketRegime.TRENDING},
    "VWAPReclaimStrategy": {MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY, MarketRegime.TRENDING},
    "TrendPullbackStrategy": {MarketRegime.TRENDING, MarketRegime.LOW_VOLATILITY},
    "LiquiditySweepStrategy": {MarketRegime.RANGE_BOUND, MarketRegime.HIGH_VOLATILITY},
}


def _strategy_compatible_with_regime(strategy: BaseStrategy, regime: MarketRegime) -> bool:
    """Check if a strategy should run in the current market regime."""
    if regime == MarketRegime.INSUFFICIENT_DATA:
        # Allow strategies with their own time/structure filters (ORB, VWAP, LiquiditySweep)
        # They don't depend on ADX-based regime to function correctly
        class_name = type(strategy).__name__
        if class_name in ("ORBStrategy", "VWAPReclaimStrategy", "LiquiditySweepStrategy"):
            return True
        return False
    class_name = type(strategy).__name__
    allowed_regimes = _REGIME_STRATEGY_COMPAT.get(class_name)
    if allowed_regimes is None:
        return True  # Not in map = allowed everywhere
    return regime in allowed_regimes


# Breakout level keys per strategy — used for failed breakout detection
from app.core.models import OptionType, StrategyName as _SN

_BREAKOUT_LEVEL_KEYS: dict[_SN, dict[str, str]] = {
    _SN.ORB: {"CE": "orh", "PE": "orl"},
    _SN.RANGE_BREAKOUT: {"CE": "range_high", "PE": "range_low"},
    _SN.MOMENTUM_BREAKOUT: {"CE": "lookback_high", "PE": "lookback_low"},
    _SN.BREAKOUT_20D: {"CE": "high_20d", "PE": "low_20d"},
    _SN.EMA_BREAKOUT: {"CE": "ema50", "PE": "ema50"},
}


def _extract_breakout_level(signal: StrategySignal) -> Optional[float]:
    """Extract the spot price breakout level from signal details."""
    keys = _BREAKOUT_LEVEL_KEYS.get(signal.strategy)
    if keys is None:
        return None
    key = keys.get(signal.option_type.value)
    if key is None:
        return None
    level = signal.details.get(key)
    return float(level) if level is not None else None


class Orchestrator:
    """Central trading system controller — multi-instrument aware."""

    def __init__(self) -> None:
        # Components
        self.client = AngelOneClient()
        self.validator = DataValidator()
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()
        self.signal_scorer = SignalScorer()
        self.ai_engine = AIDecisionEngine()
        self.risk_manager = RiskManager()
        self.paper_trader = PaperTradingEngine()
        self.broker = AngelOneBroker()
        self.trade_logger = TradeLogger()
        self.history_logger = HistoryLogger()
        self.alert_manager = AlertManager()
        self.eval_scheduler = EvaluationScheduler(lookback_days=20)
        self.pre_market_analyst = PreMarketAnalyst()
        self.insight_manager = InsightManager(self.pre_market_analyst)

        # Wire insight manager into signal scorer and AI decision engine
        set_insight_manager(self.insight_manager)
        set_ai_insight_manager(self.insight_manager)

        # Strategies (applied to every instrument) — V1 Engine
        self.strategies: list[BaseStrategy] = [
            ORBStrategy(),
            VWAPReclaimStrategy(),
            TrendPullbackStrategy(),
            LiquiditySweepStrategy(),
            RangeBreakoutStrategy(),
            MomentumBreakoutStrategy(),
            EMABreakoutStrategy(),
            Breakout20DStrategy(),
        ]

        # ── V2 Engine components ─────────────────────────────────────────
        self.v2_paper_trader = PaperTradingEngine()
        self.v2_risk_manager = RiskManager(
            max_trades=settings.v2_max_trades_per_day,
            max_concurrent=settings.v2_max_concurrent_positions,
            risk_pct=settings.v2_risk_per_trade_pct,
            consecutive_limit=settings.v2_consecutive_loss_limit,
        )
        self.v2_signal_scorer = SignalScorer()
        self.v2_smart_exit = SmartExitEngine()
        self.v2_day_classifier = DayClassifier()
        self.v2_gex_calculator = GEXCalculator()
        self.v2_gex_bounce = GEXBounceStrategy()
        self.v2_strategies: list[BaseStrategy] = [
            VWAPPullbackStrategy(),
            RSIExtremeStrategy(),
            self.v2_gex_bounce,
        ]
        self.v2_day_type: DayType = DayType.PENDING
        self.v2_day_classified = False

        # Active instruments — resolved from evaluation or config
        self._active_instruments: list[InstrumentConfig] = []
        # Per-instrument strategy whitelist from evaluation
        # {symbol: [strategy_instance, ...]}  — empty means run all
        self._instrument_strategies: dict[str, list[BaseStrategy]] = {}
        # Eval score boost: {(symbol, strategy_name): composite_score}
        self._eval_scores: dict[tuple[str, str], float] = {}

        # State — per instrument
        self.running = False
        self._cycle_count = 0
        self._global_last_fetched: Optional[datetime] = None
        self.global_bias = GlobalBias.UNAVAILABLE
        self.global_indices: list = []
        # Per-instrument snapshots  {symbol: MarketSnapshot}
        self.snapshots: dict[str, MarketSnapshot] = {}
        # Per-instrument today DataFrames  {symbol: pd.DataFrame} — shared V1→V2
        self._df_today_cache: dict[str, pd.DataFrame] = {}
        # Per-instrument options metrics  {symbol: OptionsMetrics}
        self._options_metrics: dict[str, OptionsMetrics] = {}
        # Per-instrument expiry  {symbol: str}
        self._expiries: dict[str, str] = {}
        # Per-instrument expiry dates (parsed from SmartAPI)  {symbol: date}
        self._expiry_dates: dict[str, object] = {}
        self._last_option_chain: dict[str, list] = {}
        # Per-instrument higher-timeframe trend bias
        self._htf_biases: dict[str, str] = {}
        # Per-instrument 20-day Donchian levels from daily candles
        # {symbol: {"high_20d": float, "low_20d": float}}
        self._daily_levels: dict[str, dict] = {}
        # Backward compat
        # Backward compat
        self.snapshot: Optional[MarketSnapshot] = None

        # Margin safety — tracks whether live trading was auto-paused
        self._live_paused_insufficient_margin = False

        # Heartbeat tracking
        self._last_heartbeat_cycle = 0
        self._consecutive_no_signal = 0

        # RSS news polling (every 30 min during market hours)
        self._last_rss_fetch: Optional[datetime] = None

        # ── Pipeline Activity Log ────────────────────────────────────
        # Ring buffer of pipeline events for dashboard visibility
        self.activity_log: deque[dict] = deque(maxlen=500)
        # Data source health — quick status for dashboard
        self.data_sources: dict = {
            "angelone_auth": {"status": "pending", "updated": None},
            "candles": {"status": "pending", "updated": None, "detail": ""},
            "options_chain": {"status": "pending", "updated": None, "detail": ""},
            "fii_dii": {"status": "pending", "updated": None, "detail": ""},
            "breadth": {"status": "pending", "updated": None, "detail": ""},
            "news": {"status": "pending", "updated": None, "detail": ""},
            "global_indices": {"status": "pending", "updated": None, "detail": ""},
            "ai_insight": {"status": "pending", "updated": None, "detail": ""},
        }

    async def start(self) -> None:
        """Main entry point — runs forever, restarting each trading day."""
        # Load previous evaluation from DB on cold start
        try:
            await self.eval_scheduler.load_latest_from_db()
        except Exception:
            logger.warning("Could not load previous evaluation from DB")

        # Start auto-evaluation scheduler (runs daily if not already evaluated)
        self.eval_scheduler.start_auto_schedule(get_enabled_instruments)

        # Resolve instruments (auto-select or config)
        self._active_instruments = self._resolve_instruments()
        inst_names = [i.symbol for i in self._active_instruments]

        logger.info("=" * 60)
        logger.info("TradeAI Orchestrator started")
        logger.info("Mode: %s", "AUTO-SELECT" if settings.auto_select_instruments else "MANUAL")
        logger.info("Active instruments: %s", ", ".join(inst_names))
        logger.info("Paper trading: %s", settings.paper_trading)
        logger.info("Capital: ₹%s", f"{settings.initial_capital:,.0f}")
        logger.info("=" * 60)

        while True:
            try:
                await self._run_trading_day()
            except Exception:
                logger.exception("Unhandled error in trading day loop")
                self._log_event("error", "Trading day crashed with unhandled error")
                # If still within market hours, retry after a short delay
                # instead of sleeping until tomorrow
                now_ist = datetime.now(IST)
                if now_ist.time() < MARKET_CLOSE and not is_market_holiday(now_ist.date()):
                    retry_delay = 120  # 2 minutes
                    logger.info(
                        "Still within market hours — retrying in %d seconds...",
                        retry_delay,
                    )
                    self._log_event("system", f"Will retry in {retry_delay}s (still market hours)")
                    self.running = False
                    await asyncio.sleep(retry_delay)
                    continue

            # After each day, reset state and wait until next pre-market
            self._reset_for_new_day()
            await self._sleep_until_premarket()

    def _reset_for_new_day(self) -> None:
        """Reset daily state for a fresh trading day."""
        self._cycle_count = 0
        self._global_last_fetched = None
        self.global_bias = GlobalBias.UNAVAILABLE
        self.global_indices = []
        self.snapshots = {}
        self._df_today_cache = {}
        self._options_metrics = {}
        self._expiries = {}
        self._expiry_dates = {}
        self._last_option_chain = {}
        self._htf_biases = {}
        self.snapshot = None
        self._last_heartbeat_cycle = 0
        self._consecutive_no_signal = 0
        self._last_rss_fetch = None
        self.paper_trader = PaperTradingEngine()  # Fresh daily paper trader
        # V2 reset
        self.v2_paper_trader = PaperTradingEngine()
        self.v2_day_type = DayType.PENDING
        self.v2_day_classified = False
        self.running = False
        logger.info("Daily state reset complete")

    def _log_event(self, event_type: str, message: str, *, cycle: int = 0, instrument: str = "", data: dict | None = None) -> None:
        """Append a pipeline event to the activity log ring buffer."""
        self.activity_log.append({
            "ts": datetime.now(IST).strftime("%H:%M:%S"),
            "cycle": cycle,
            "instrument": instrument,
            "type": event_type,
            "msg": message,
            "data": data or {},
        })

    def _set_source(self, source: str, status: str, detail: str = "") -> None:
        """Update a data source health entry."""
        self.data_sources[source] = {
            "status": status,
            "updated": datetime.now(IST).strftime("%H:%M:%S"),
            "detail": detail,
        }

    async def _save_signal_record(
        self, signal: StrategySignal, score: float,
        ai_decision_str: str, ai_confidence: float, reason: str,
    ) -> None:
        """Persist signal to DB for audit trail."""
        try:
            from app.db.models import SignalRecord
            from app.trading.history_logger import _get_session_factory
            now = datetime.now(IST)
            SessionLocal = _get_session_factory()
            async with SessionLocal() as session:
                async with session.begin():
                    session.add(SignalRecord(
                        date=now.strftime("%Y-%m-%d"),
                        time=now.strftime("%H:%M:%S"),
                        instrument=getattr(signal, "instrument", "NIFTY"),
                        strategy=signal.strategy.value,
                        option_type=signal.option_type.value,
                        score=score,
                        confidence=score,
                        strike_price=signal.strike_price,
                        entry_price=signal.entry_price,
                        ai_decision=ai_decision_str,
                        ai_confidence=ai_confidence,
                        reason=reason[:500] if reason else None,
                    ))
        except Exception:
            logger.debug("Failed to save signal record (non-critical)")

    async def _sleep_until_premarket(self) -> None:
        """Sleep until next trading day's pre-market (08:45 IST)."""
        now = datetime.now(IST)
        next_day = next_trading_date(now.date())
        next_start = IST.localize(datetime.combine(next_day, LOAD_TIME))

        wait_seconds = (next_start - now).total_seconds()
        if wait_seconds > 0:
            logger.info(
                "Sleeping until next pre-market: %s (%.1f hours)",
                next_start.strftime("%Y-%m-%d %H:%M IST"),
                wait_seconds / 3600,
            )
            await asyncio.sleep(wait_seconds)

    async def _run_trading_day(self) -> None:
        """Execute a single trading day lifecycle."""
        today = datetime.now(IST).date()

        # Skip holidays
        if is_market_holiday(today):
            logger.info("Today (%s) is a market holiday — skipping", today)
            return

        self.running = True

        # ── Pre-market evaluation: auto-select instruments ───────────────
        if settings.auto_select_instruments:
            await self._run_premarket_evaluation()
            # Re-resolve instruments based on fresh evaluation
            self._active_instruments = self._resolve_instruments()
        else:
            self._active_instruments = self._resolve_instruments()

        # Authenticate with AngelOne (retry up to 3 times)
        auth_ok = False
        for auth_attempt in range(3):
            if self.client.authenticate():
                auth_ok = True
                break
            logger.warning("Auth attempt %d/3 failed — retrying in 10s...", auth_attempt + 1)
            self._log_event("auth", f"AngelOne auth attempt {auth_attempt + 1}/3 FAILED")
            await asyncio.sleep(10)

        if not auth_ok:
            logger.error("Failed to authenticate after 3 attempts. Aborting today.")
            self._set_source("angelone_auth", "error", "Authentication failed (3 attempts)")
            self._log_event("auth", "AngelOne authentication FAILED after 3 attempts")
            await self.alert_manager.send_info("AUTH FAILED", "AngelOne authentication failed after 3 attempts. Check credentials.")
            return
        self._set_source("angelone_auth", "ok", "Authenticated")
        self._log_event("auth", "AngelOne authenticated successfully")

        # Sync lot sizes from SmartAPI instrument master (after auth loads the master)
        try:
            self._log_event("setup", "Loading instrument master & syncing lot sizes...")
            from app.core.instruments import sync_lot_sizes_from_broker
            sync_lot_sizes_from_broker(self.client)
            self._log_event("setup", "Lot sizes synced from SmartAPI")
        except Exception:
            logger.exception("Failed to sync lot sizes — using defaults")
            self._log_event("setup", "Lot size sync FAILED — using defaults")

        # Determine weekly/monthly expiry for each active instrument (from SmartAPI)
        try:
            self._log_event("setup", "Discovering option expiries...")
            for inst in self._active_instruments:
                inst_name = inst.option_symbol_prefix or inst.symbol
                expiry = self.client.get_nearest_weekly_expiry(instrument_name=inst_name)
                if expiry:
                    self._expiries[inst.symbol] = expiry
                    try:
                        parsed = datetime.strptime(expiry, "%d%b%y").date()
                        self._expiry_dates[inst.symbol] = parsed
                        logger.info("Weekly expiry for %s: %s (%s)", inst.symbol, expiry, parsed)
                    except ValueError:
                        logger.warning("Could not parse expiry date '%s' for %s", expiry, inst.symbol)
                        logger.info("Weekly expiry for %s: %s", inst.symbol, expiry)
                else:
                    logger.warning("No expiry found for %s — options data unavailable", inst.symbol)
            self._log_event("setup", f"Expiries resolved: {dict(self._expiries)}")
        except Exception:
            logger.exception("Failed to determine expiries — options data may be unavailable")
            self._log_event("setup", "Expiry discovery FAILED — continuing without options")

        # ── Start WebSocket for real-time streaming ──────────────────────
        try:
            self._log_event("setup", "Starting WebSocket streaming...")
            if self.client.start_websocket():
                await self._bootstrap_websocket()
                self._log_event("setup", "WebSocket streaming active — API polling disabled for candles")
            else:
                self._log_event("setup", "WebSocket unavailable — using API polling fallback")
        except Exception:
            logger.exception("WebSocket startup failed (non-critical)")
            self._log_event("setup", "WebSocket startup FAILED — using API polling fallback")

        # Update shared state for API
        from app.api.routes import get_state
        state = get_state()
        state["eval_scheduler"] = self.eval_scheduler

        inst_names = [i.symbol for i in self._active_instruments]
        await self.alert_manager.send_info(
            "SYSTEM STARTED",
            f"Mode: {'AUTO-SELECT' if settings.auto_select_instruments else 'MANUAL'}\n"
            f"Instruments: {', '.join(inst_names)}\n"
            f"Paper trading: {settings.paper_trading}\nCapital: ₹{settings.initial_capital:,.0f}",
        )
        self._log_event("setup", f"System started: instruments={', '.join(inst_names)}")

        # ── Crash recovery: reload open trades from DB ───────────────────
        await self._recover_open_trades()

        # Authenticate broker for live trading
        if not settings.paper_trading:
            if self.broker.authenticate():
                logger.info("AngelOneBroker authenticated for live trading")
            else:
                logger.error("AngelOneBroker auth failed — falling back to paper trading for safety")
                await self.alert_manager.send_info(
                    "BROKER AUTH FAILED",
                    "Live broker authentication failed. System will NOT place real orders today.\n"
                    "Paper trading mode active as safety fallback.",
                )

        # ── Pre-market intelligence: AI analysis of all data sources ─────
        try:
            insight = await self.pre_market_analyst.run_analysis()
            if insight:
                # Update shared state for API
                from app.api.routes import get_state
                state = get_state()
                state["intelligence"] = insight
                state["pre_market_analyst"] = self.pre_market_analyst

                modifier = self.insight_manager.get_score_modifier()
                bias = self.insight_manager.get_market_bias()
                logger.info(
                    "Pre-market intelligence: bias=%s, score_modifier=%+d, risk=%s",
                    bias, modifier, self.insight_manager.get_risk_advice(),
                )
                self._set_source("ai_insight", "ok", f"Bias: {bias}, modifier: {modifier:+d}")
                self._log_event("intelligence", f"Pre-market AI: bias={bias}, score_modifier={modifier:+d}", data={
                    "bias": bias, "modifier": modifier, "risk": self.insight_manager.get_risk_advice(),
                    "summary": insight.get("summary", "")[:200],
                })

                # Log data source results
                analyst = self.pre_market_analyst
                if analyst.institutional_flow:
                    self._set_source("fii_dii", "ok", f"FII net: {analyst.institutional_flow.fii_net:+.0f} Cr")
                    self._log_event("data", f"FII/DII: FII net={analyst.institutional_flow.fii_net:+.0f} Cr, signal={analyst.institutional_flow.signal}")
                else:
                    self._set_source("fii_dii", "warn", "Not available")
                    self._log_event("data", "FII/DII data: NOT available")

                if analyst.market_breadth:
                    self._set_source("breadth", "ok", f"A/D ratio: {analyst.market_breadth.advance_decline_ratio:.2f}")
                    self._log_event("data", f"Market breadth: A/D={analyst.market_breadth.advance_decline_ratio:.2f}, signal={analyst.market_breadth.breadth_signal}")
                else:
                    self._set_source("breadth", "warn", "Not available")
                    self._log_event("data", "Market breadth: NOT available")

                news_count = len(analyst._news_items) if hasattr(analyst, '_news_items') else 0
                db_news_count = insight.get("news_count", 0)
                total_news = max(news_count, db_news_count)
                if total_news > 0:
                    self._set_source("news", "ok", f"{total_news} news items")
                    self._log_event("data", f"News: {total_news} items ({news_count} scraped, {db_news_count} in DB)")
                else:
                    self._set_source("news", "warn", "No news collected")
                    self._log_event("data", "News: none collected")

                # Send Telegram summary
                summary = insight.get("summary", "Analysis complete")
                plan = insight.get("trading_plan", "")
                await self.alert_manager.send_info(
                    f"PRE-MARKET ANALYSIS — {bias.upper()}",
                    f"{summary}\n\nPlan: {plan}\nScore modifier: {modifier:+d}",
                )
            else:
                self._set_source("ai_insight", "warn", "No insight returned")
                self._log_event("intelligence", "Pre-market AI returned no insight")
        except Exception:
            logger.exception("Pre-market intelligence failed (non-critical)")
            self._set_source("ai_insight", "error", "Analysis failed")
            self._log_event("intelligence", "Pre-market AI analysis FAILED (exception)")

        # ── Fetch daily candles for 20-day Donchian levels ───────────────
        try:
            self._log_event("setup", "Fetching daily candles for Donchian levels...")
            await self._fetch_daily_levels()
        except Exception:
            logger.exception("Failed to fetch daily Donchian levels (non-critical)")
            self._log_event("setup", "Daily candle fetch FAILED — Breakout20D will be inactive")

        # Lock the adaptive threshold for the entire session
        # Based on what data sources are actually available right now
        analyst = self.pre_market_analyst
        session_threshold = lock_session_threshold(
            options_available=bool(self._expiries),
            global_available=(self.global_bias != GlobalBias.UNAVAILABLE),
            fii_dii_available=(analyst.institutional_flow is not None),
            breadth_available=(analyst.market_breadth is not None),
            news_available=(self.insight_manager.has_insight and bool(analyst._news_items if hasattr(analyst, '_news_items') else False)),
        )
        logger.info("Session threshold locked at: %d", session_threshold)

        while self.running:
            now_ist = datetime.now(IST)
            current_time = now_ist.time()

            # Pre-market: Fetch global data
            if current_time >= GLOBAL_FETCH_TIME and current_time < MARKET_OPEN:
                await self._fetch_global_data()
                await asyncio.sleep(300)  # Re-check every 5 min pre-market
                continue

            # Market hours: continuous analysis
            if MARKET_OPEN <= current_time < PRE_CLOSE:
                # Fetch global data if never fetched (late start) or stale (>15 min)
                if self._global_last_fetched is None:
                    logger.info("Late start detected — fetching global data now")
                    try:
                        await self._fetch_global_data()
                    except Exception:
                        logger.exception("Failed to fetch global data on late start")
                elif (datetime.now(IST) - self._global_last_fetched).total_seconds() > 900:
                    try:
                        await self._fetch_global_data()
                    except Exception:
                        logger.exception("Failed to refresh global data")

                # Periodic RSS news fetch (every 30 min)
                await self._maybe_fetch_rss_news()

                await self._run_analysis_cycle()
                await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                continue

            # Pre-close: close open trades
            if PRE_CLOSE <= current_time < MARKET_CLOSE:
                await self._close_all_trades()
                await asyncio.sleep(60)
                continue

            # Post-market: generate daily report + run strategy evaluation
            if REPORT_TIME <= current_time <= dtime(16, 0):
                await self._generate_daily_report()
                await self._run_post_market_evaluation()
                self.running = False
                logger.info("Trading day complete. Will restart next trading day.")
                return

            # Outside market hours — wait
            logger.debug("Outside market hours (%s). Waiting...", current_time)
            await asyncio.sleep(60)

    # ── Core Analysis Cycle ──────────────────────────────────────────────

    async def _run_analysis_cycle(self) -> None:
        """Single iteration of the 1-minute analysis loop — all instruments."""
        self._cycle_count += 1
        cycle = self._cycle_count
        now = datetime.now(IST)
        logger.info("── Cycle %d ── %s ──", cycle, now.strftime("%H:%M:%S"))
        self._log_event("cycle", f"Cycle #{cycle} started", cycle=cycle)

        try:
            # Analyze each active instrument (with small delay between to avoid API rate limits)
            for idx, instrument in enumerate(self._active_instruments):
                if idx > 0:
                    await asyncio.sleep(2)  # 2s gap to avoid AngelOne rate limiting
                if settings.v1_enabled:
                    await self._analyze_instrument(instrument, cycle, now)

                # V2 engine: runs on the same data, independent trade book
                if settings.v2_enabled:
                    await self._v2_analyze_instrument(instrument, cycle, now)

            # Fetch basic price data for main indices not in active set
            await self._update_index_snapshots(now)

            # Update shared state for API (use NIFTY as primary if available)
            from app.api.routes import get_state
            state = get_state()
            nifty_snap = self.snapshots.get("NIFTY")
            if nifty_snap:
                state["snapshot"] = nifty_snap
                self.snapshot = nifty_snap
            elif self.snapshots:
                first = next(iter(self.snapshots.values()))
                state["snapshot"] = first
                self.snapshot = first
            state["open_trades"] = self.paper_trader.open_trades
            state["snapshots"] = self.snapshots  # All instrument snapshots
            state["intelligence"] = self.pre_market_analyst.latest_insight
            state["pre_market_analyst"] = self.pre_market_analyst
            # V2 state
            state["v2_open_trades"] = self.v2_paper_trader.open_trades
            state["v2_day_type"] = self.v2_day_type.value if self.v2_day_type else "pending"

            # Heartbeat alert every 15 cycles
            if self._consecutive_no_signal > 0 and cycle - self._last_heartbeat_cycle >= 15:
                self._last_heartbeat_cycle = cycle
                snap = self.snapshot
                if snap:
                    await self.alert_manager.send_info(
                        f"SYSTEM HEARTBEAT — Cycle #{cycle}",
                        f"Instruments: {', '.join(i.symbol for i in self._active_instruments)}\n"
                        f"No signals for {self._consecutive_no_signal} consecutive cycles.\n"
                        f"Strategies are monitoring — waiting for conditions to align.",
                    )

        except Exception:
            logger.exception("Error in analysis cycle")

    async def _maybe_fetch_rss_news(self) -> None:
        """Fetch RSS + Telegram news if 15+ minutes since last fetch."""
        now = datetime.now(IST)
        if self._last_rss_fetch and (now - self._last_rss_fetch).total_seconds() < 900:
            return  # Not time yet

        try:
            from app.data.rss_news import fetch_and_analyze
            from app.data.telegram_news import collect_telegram_news, save_news_to_db

            total_saved = 0

            # Fetch RSS feeds
            articles = await fetch_and_analyze()
            if articles:
                saved = await save_news_to_db(articles)
                total_saved += saved
                self._log_event("data", f"RSS news: {saved} new articles saved")

            # Fetch Telegram channel news
            try:
                tg_items = await collect_telegram_news()
                if tg_items:
                    tg_saved = await save_news_to_db(tg_items)
                    total_saved += tg_saved
                    self._log_event("data", f"Telegram news: {tg_saved} new items saved")
            except Exception:
                logger.warning("Telegram news fetch failed (non-critical)")

            if total_saved > 0:
                self._set_source("news", "ok", f"{total_saved} news items")
                await self._update_live_news_sentiment()
            else:
                self._log_event("data", "News: no new articles from RSS or Telegram")

            self._last_rss_fetch = now
        except Exception:
            logger.exception("News fetch failed (non-critical)")
            self._last_rss_fetch = now  # Don't retry immediately on failure

    async def _update_live_news_sentiment(self) -> None:
        """Recalculate news sentiment from recent DB entries and update insight manager."""
        try:
            from app.data.telegram_news import get_recent_news
            recent = await get_recent_news(days=1)

            if not recent:
                return

            # Weighted average: more recent items get higher weight
            # Only consider last 2 hours of news for live sentiment
            now = datetime.now(IST)
            cutoff = now - timedelta(hours=2)
            recent_scores = []
            for item in recent:
                score = item.get("sentiment_score", 0)
                if score and isinstance(score, (int, float)):
                    recent_scores.append(score)

            if recent_scores:
                avg_sentiment = sum(recent_scores) / len(recent_scores)
                # Update the insight with live news sentiment
                if self.pre_market_analyst.latest_insight is not None:
                    self.pre_market_analyst.latest_insight["news_sentiment"] = avg_sentiment
                    logger.info(
                        "Live news sentiment updated: %.2f (from %d items)",
                        avg_sentiment, len(recent_scores),
                    )
        except Exception:
            logger.warning("Failed to update live news sentiment")

    async def _update_index_snapshots(self, now: datetime) -> None:
        """Fetch basic price data for NIFTY/BANKNIFTY/FINNIFTY if not already tracked.

        Ensures the dashboard always shows main index prices even when
        those indices aren't in the active trading instrument set.
        """
        from app.core.instruments import NIFTY, BANKNIFTY, FINNIFTY

        index_configs = [NIFTY, BANKNIFTY, FINNIFTY]
        active_symbols = {i.symbol for i in self._active_instruments}

        for idx_conf in index_configs:
            if idx_conf.symbol in active_symbols:
                continue  # Already fully analyzed
            if idx_conf.symbol in self.snapshots:
                # Already have a snapshot from this session — just refresh price
                pass

            try:
                from_date = now.strftime("%Y-%m-%d 09:15")
                to_date = now.strftime("%Y-%m-%d %H:%M")
                candles = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.get_candle_data,
                        idx_conf.token, idx_conf.exchange.value,
                        "FIVE_MINUTE", from_date, to_date,
                    ),
                    timeout=15,
                )
                df = self.client.candles_to_dataframe(candles)
                if df.empty:
                    continue

                spot_price = float(df.iloc[-1]["close"])
                prev_close = float(df.iloc[0]["open"]) if len(df) > 1 else None
                day_high = float(df["high"].max())
                day_low = float(df["low"].min())

                # Compute basic indicators for display
                indicators = self.feature_engine.get_latest_indicators(
                    self.feature_engine.compute_indicators(df, today_date=now.strftime("%Y-%m-%d"))
                )

                snap = MarketSnapshot(
                    instrument=idx_conf.symbol,
                    price=spot_price,
                    nifty_price=spot_price if idx_conf.symbol == "NIFTY" else (
                        self.snapshots["NIFTY"].nifty_price if "NIFTY" in self.snapshots else spot_price
                    ),
                    vwap=indicators.vwap,
                    regime=self.regime_detector.detect(df),
                    global_bias=self.global_bias,
                    indicators=indicators,
                    options_metrics=self._options_metrics.get(idx_conf.symbol, OptionsMetrics()),
                    timestamp=now,
                    prev_day_high=day_high,
                    prev_day_low=day_low,
                    prev_day_close=prev_close,
                    is_expiry_day=False,
                    htf_trend=self._htf_biases.get(idx_conf.symbol),
                )
                self.snapshots[idx_conf.symbol] = snap
                logger.debug("[%s] Index snapshot updated: %.2f", idx_conf.symbol, spot_price)
            except Exception as e:
                logger.debug("[%s] Index snapshot fetch failed: %s", idx_conf.symbol, e)

    async def _fetch_daily_levels(self) -> None:
        """Fetch daily candles for each active instrument and compute 20-day Donchian levels.

        Called once per trading day during setup. The levels are cached in
        self._daily_levels and passed to strategies that need them (e.g. Breakout20D).

        Book reference: Donchian (1960s), Curtis Faith *Way of the Turtle* —
        20-day high/low breakout channel.
        """
        for instrument in self._active_instruments:
            symbol = instrument.symbol
            try:
                daily_candles = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.get_daily_candles,
                        instrument.token, instrument.exchange.value,
                        days=30,  # ~20 trading days
                    ),
                    timeout=30,
                )
                if not daily_candles or len(daily_candles) < 15:
                    logger.warning(
                        "[%s] Insufficient daily candles (%d) for Donchian — need ≥15 trading days",
                        symbol, len(daily_candles) if daily_candles else 0,
                    )
                    continue

                df_daily = self.client.candles_to_dataframe(daily_candles)
                # Exclude today's (partial) candle — use only completed days
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                df_daily = df_daily[df_daily.index.strftime("%Y-%m-%d") != today_str]

                if len(df_daily) < 15:
                    logger.warning("[%s] Only %d completed daily candles — skipping Donchian", symbol, len(df_daily))
                    continue

                # Use last 20 completed days (or all if fewer)
                lookback = df_daily.tail(20)
                high_20d = float(lookback["high"].max())
                low_20d = float(lookback["low"].min())

                self._daily_levels[symbol] = {
                    "high_20d": high_20d,
                    "low_20d": low_20d,
                    "days_used": len(lookback),
                }
                logger.info(
                    "[%s] Donchian 20-day levels: high=%.2f, low=%.2f (%d days)",
                    symbol, high_20d, low_20d, len(lookback),
                )
                self._log_event(
                    "data",
                    f"Donchian daily levels: high={high_20d:.2f}, low={low_20d:.2f} ({len(lookback)} days)",
                    instrument=symbol,
                )
            except asyncio.TimeoutError:
                logger.warning("[%s] Daily candle fetch timed out", symbol)
            except Exception:
                logger.exception("[%s] Failed to fetch daily candles for Donchian", symbol)

    async def _bootstrap_websocket(self) -> None:
        """Subscribe instruments to WebSocket and seed CandleBuilders with history.

        Called once after WebSocket connects.  Fetches previous-day +
        today's candles via API (one-time cost), seeds the builder, then
        subscribes the token so subsequent ticks are aggregated in
        real-time — eliminating repeated getCandleData polling.
        """
        now = datetime.now(IST)
        prev_day = previous_trading_date(now.date())
        from_date = prev_day.strftime("%Y-%m-%d 09:15")
        to_date = now.strftime("%Y-%m-%d %H:%M")

        for inst in self._active_instruments:
            token = inst.token
            exchange = inst.exchange.value
            futures_token = None

            # Bootstrap spot candles
            try:
                await asyncio.to_thread(
                    self.client.bootstrap_ws_candles,
                    token, exchange, from_date, to_date,
                )
            except Exception:
                logger.warning("[%s] WS bootstrap (spot) failed", inst.symbol)

            # Bootstrap futures candles (for volume)
            if inst.is_index:
                futures_token = self.client._get_index_fut_token(inst.symbol)
                if futures_token:
                    try:
                        await asyncio.to_thread(
                            self.client.bootstrap_ws_candles,
                            futures_token, self.client.nfo_exchange,
                            from_date, to_date,
                        )
                    except Exception:
                        logger.warning("[%s] WS bootstrap (futures) failed", inst.symbol)

            # Subscribe to spot + futures tokens
            self.client.subscribe_instrument(token, exchange, futures_token)

        logger.info(
            "WebSocket bootstrapped for %d instruments",
            len(self._active_instruments),
        )

    async def _analyze_instrument(
        self, instrument: InstrumentConfig, cycle: int, now: datetime
    ) -> None:
        """Run full analysis for a single instrument."""
        symbol = instrument.symbol
        try:
            # 1. Fetch candle data (include previous trading day for indicator warmup)
            prev_day = previous_trading_date(now.date())
            from_date = prev_day.strftime("%Y-%m-%d 09:15")
            to_date = now.strftime("%Y-%m-%d %H:%M")
            today_str = now.strftime("%Y-%m-%d")

            # Try WebSocket first — zero API calls when connected
            df = self.client.get_live_candles(instrument.token, instrument.exchange.value)
            if not df.empty:
                # If WS data has gaps (e.g. after reconnection), re-bootstrap
                # to heal the CandleBuilder — otherwise has_data_gap would
                # reject every cycle for the rest of the day.
                if self.validator.has_data_gap(df):
                    logger.warning("[%s] WS candle gap detected — re-bootstrapping from API", symbol)
                    cache_key = f"{instrument.exchange.value}:{instrument.token}"
                    self.client._ws_bootstrapped.discard(cache_key)
                    try:
                        await asyncio.to_thread(
                            self.client.bootstrap_ws_candles,
                            instrument.token, instrument.exchange.value,
                            from_date, to_date,
                        )
                        df = self.client.get_live_candles(instrument.token, instrument.exchange.value)
                    except Exception:
                        logger.warning("[%s] WS re-bootstrap failed — using API fallback", symbol)
                        df = pd.DataFrame()
                if not df.empty:
                    self._log_event("candle", f"Candles from WebSocket: {len(df)} rows", cycle=cycle, instrument=symbol)
            if df.empty:
                # Fallback: API polling (used on first cycle or when WS is down)
                try:
                    candles = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.get_candle_data,
                            instrument.token, instrument.exchange.value,
                            "ONE_MINUTE", from_date, to_date,
                        ),
                        timeout=45,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[%s][Cycle %d] Candle data fetch timed out", symbol, cycle)
                    self._log_event("candle", "Candle fetch TIMED OUT", cycle=cycle, instrument=symbol)
                    return
                df = self.client.candles_to_dataframe(candles)

            if df.empty:
                logger.warning("[%s][Cycle %d] No candle data received", symbol, cycle)
                self._log_event("candle", "No candle data received", cycle=cycle, instrument=symbol)
                return

            # 2. Validate data
            df = self.validator.validate_candles(df, is_index=instrument.is_index)
            if not self.validator.is_valid_for_trading(df):
                logger.warning("[%s][Cycle %d] Data validation failed", symbol, cycle)
                self._log_event("candle", "Data validation FAILED", cycle=cycle, instrument=symbol)
                return

            self._set_source("candles", "ok", f"{len(df)} candles for {symbol}")
            self._log_event("candle", f"Candles OK: {len(df)} rows", cycle=cycle, instrument=symbol)

            # 2b. For indices, merge futures volume
            if instrument.is_index:
                # Try WebSocket-fed futures candles first
                fut_df = self.client.get_live_futures_candles(instrument.symbol)
                if not fut_df.empty:
                    df = self.feature_engine.merge_futures_volume(df, fut_df)
                    self._log_event("data", f"Futures volume (WS) merged: {fut_df['volume'].sum():,}", cycle=cycle, instrument=symbol)
                else:
                    # Fallback: API call for futures candles
                    try:
                        fut_candles = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.client.get_index_futures_candles,
                                instrument.symbol,  # NIFTY, BANKNIFTY, etc.
                                "ONE_MINUTE", from_date, to_date,
                            ),
                            timeout=45,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[%s] Futures candle fetch timed out", symbol)
                        fut_candles = []
                    except Exception:
                        logger.warning("[%s] Futures candle fetch failed", symbol, exc_info=True)
                        fut_candles = []
                    if fut_candles:
                        fut_df = self.client.candles_to_dataframe(fut_candles)
                        df = self.feature_engine.merge_futures_volume(df, fut_df)
                        self._log_event("data", f"Futures volume merged: {fut_df['volume'].sum():,}", cycle=cycle, instrument=symbol)
                    else:
                        logger.warning("[%s][Cycle %d] No futures volume — VWAP/volume scoring limited", symbol, cycle)
                        self._log_event("data", "No futures volume available — using ATR/option proxies", cycle=cycle, instrument=symbol)

            # 2c. Compute 5-min HTF bias (resample from 1-min data — no API call)
            if now.minute % 5 == 0 or symbol not in self._htf_biases:
                try:
                    df_5m = df.resample("5min").agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                    ).dropna(subset=["open"])
                    if not df_5m.empty:
                        htf_bias = self.feature_engine.compute_htf_bias(df_5m)
                    else:
                        htf_bias = "neutral"
                    self._htf_biases[symbol] = htf_bias
                    logger.debug("[%s] HTF (5-min) trend: %s", symbol, htf_bias)
                except Exception as e:
                    logger.warning("[%s] 5-min resample failed: %s", symbol, e)
                    self._htf_biases.setdefault(symbol, "neutral")

            # 3. Feature engineering (on full history for proper indicator warmup)
            df = self.feature_engine.compute_indicators(df, today_date=today_str)
            # Filter to today-only for strategies/snapshot (indicators are already computed)
            df_today = df[df.index.strftime("%Y-%m-%d") == today_str]
            if df_today.empty:
                logger.warning("[%s][Cycle %d] No today candles after filtering", symbol, cycle)
                return

            # Cache for V2 reuse
            self._df_today_cache[symbol] = df_today

            # 3b. Compute previous day high/low/close for support/resistance context
            df_prev = df[df.index.strftime("%Y-%m-%d") != today_str]
            prev_day_high = float(df_prev["high"].max()) if not df_prev.empty else None
            prev_day_low = float(df_prev["low"].min()) if not df_prev.empty else None
            prev_day_close = float(df_prev.iloc[-1]["close"]) if not df_prev.empty else None
            day_open = float(df_today.iloc[0]["open"]) if not df_today.empty else None

            if df_prev.empty and cycle <= 2:
                logger.warning(
                    "[%s] No previous day candle data — prev day levels will be unavailable. "
                    "API returned %d candles, all from today.",
                    symbol, len(df),
                )

            indicators = self.feature_engine.get_latest_indicators(df_today)
            spot_price = df_today.iloc[-1]["close"] if not df_today.empty else 0

            logger.info(
                "[%s][Cycle %d] Price=%.2f | RSI=%s | ADX=%s | EMA200=%s",
                symbol, cycle, spot_price,
                f"{indicators.rsi:.1f}" if indicators.rsi is not None else "N/A",
                f"{indicators.adx:.1f}" if indicators.adx is not None else "N/A",
                f"{indicators.ema200:.1f}" if indicators.ema200 is not None else "N/A",
            )
            rsi_str = f"{indicators.rsi:.1f}" if indicators.rsi is not None else "N/A"
            adx_str = f"{indicators.adx:.1f}" if indicators.adx is not None else "N/A"
            self._log_event("indicators", f"Price={spot_price:.2f} RSI={rsi_str} ADX={adx_str}", cycle=cycle, instrument=symbol, data={
                "price": round(spot_price, 2),
                "rsi": round(indicators.rsi, 1) if indicators.rsi else None,
                "adx": round(indicators.adx, 1) if indicators.adx else None,
                "vwap": round(indicators.vwap, 2) if indicators.vwap else None,
                "atr": round(indicators.atr, 2) if indicators.atr else None,
            })

            # 4. Options chain (for all instruments with expiry, every 3 min)
            options_metrics = self._options_metrics.get(symbol, OptionsMetrics())
            expiry = self._expiries.get(symbol)
            if expiry and (now.minute % 3 == 0 or options_metrics.pcr is None):
                options_metrics = await self._update_options_chain_for(instrument, spot_price, expiry)
                self._options_metrics[symbol] = options_metrics
                pcr_val = f"{options_metrics.pcr:.2f}" if options_metrics.pcr else "N/A"
                self._set_source("options_chain", "ok", f"{symbol} PCR={pcr_val}")
                self._log_event("options", f"Options chain updated: PCR={pcr_val}, MaxPain={options_metrics.max_pain}", cycle=cycle, instrument=symbol)

            # 5. Market regime detection
            regime = self.regime_detector.detect(df)
            self._log_event("regime", f"Regime: {regime.value}", cycle=cycle, instrument=symbol)

            # 6. Build market snapshot (use today's data)
            # Use NIFTY price from existing NIFTY snapshot for non-NIFTY instruments
            nifty_snap = self.snapshots.get("NIFTY")
            # Use NIFTY price if available; fall back to this instrument's price
            # so snapshots/day summaries aren't blank when NIFTY isn't active
            nifty_price = spot_price if symbol == "NIFTY" else (nifty_snap.nifty_price if nifty_snap else spot_price)
            # Check if today is expiry day for this instrument (from real SmartAPI data)
            expiry_date = self._expiry_dates.get(symbol)
            is_expiry = (expiry_date == now.date()) if expiry_date else False

            snap = MarketSnapshot(
                instrument=symbol,
                price=spot_price,
                nifty_price=nifty_price,
                vwap=indicators.vwap,
                regime=regime,
                global_bias=self.global_bias,
                indicators=indicators,
                options_metrics=options_metrics,
                timestamp=now,
                prev_day_high=prev_day_high,
                prev_day_low=prev_day_low,
                prev_day_close=prev_day_close,
                day_open=day_open,
                is_expiry_day=is_expiry,
                htf_trend=self._htf_biases.get(symbol),
            )
            self.snapshots[symbol] = snap
            await self.history_logger.save_snapshot(snap)

            # Update shared API state incrementally (so dashboard shows data ASAP)
            from app.api.routes import get_state
            state = get_state()
            if state.get("snapshot") is None:
                state["snapshot"] = snap
                self.snapshot = snap
            state["snapshots"] = self.snapshots

            # 7. Check exits on open trades for this instrument
            await self._check_trade_exits_for(instrument, spot_price)

            # 8. Time-of-day circuit breaker — no new entries after 14:30
            if now.time() >= NO_NEW_ENTRY_AFTER:
                logger.debug("[%s][Cycle %d] After %s — no new entries", symbol, cycle, NO_NEW_ENTRY_AFTER)
                self._log_event("gate", f"Past {NO_NEW_ENTRY_AFTER} — no new entries", cycle=cycle, instrument=symbol)
                return

            # 8b. Global risk check
            if not self.risk_manager.can_trade(
                self.paper_trader.all_today_trades,
                open_count=len(self.paper_trader.open_trades),
            ):
                logger.info("[%s][Cycle %d] Risk limits reached", symbol, cycle)
                self._log_event("gate", "Risk limits reached — skipping", cycle=cycle, instrument=symbol)
                return

            # 9. Run strategies — filtered by market regime
            strategies_to_run = self._instrument_strategies.get(symbol, self.strategies)
            daily_levels = self._daily_levels.get(symbol)
            signals: list[StrategySignal] = []
            for strategy in strategies_to_run:
                # Regime-strategy filtering: skip strategies that don't fit current regime
                if not _strategy_compatible_with_regime(strategy, regime):
                    continue
                signal = strategy.evaluate(df_today, options_metrics, spot_price, daily_levels=daily_levels)
                if signal:
                    signal.instrument = symbol
                    signals.append(signal)

            if not signals:
                self._consecutive_no_signal += 1
                self._log_event("signal", "No strategy signals", cycle=cycle, instrument=symbol)
                return
            else:
                self._consecutive_no_signal = 0
                self._log_event("signal", f"{len(signals)} signal(s): {', '.join(s.strategy.value for s in signals)}", cycle=cycle, instrument=symbol)

            # 9b. Multi-timeframe filter: skip signals against 5-min trend
            # Override: if 1-min RSI is extreme (≤30 or ≥70), momentum is clearly
            # directional — filter signals to match the RSI direction.
            # RSI ≤ 30 = oversold → expect bounce → CALL only
            # RSI ≥ 70 = overbought → expect pullback → PUT only
            htf_bias = self._htf_biases.get(symbol, "neutral")
            current_rsi = snap.indicators.rsi if snap and snap.indicators else None
            htf_rsi_override = current_rsi is not None and (current_rsi <= 30 or current_rsi >= 70)
            if htf_rsi_override and signals:
                # RSI extreme: filter signals to match expected bounce/pullback direction
                rsi_direction = OptionType.CALL if current_rsi <= 30 else OptionType.PUT
                pre_count = len(signals)
                signals = [s for s in signals if s.option_type == rsi_direction]
                filtered = pre_count - len(signals)
                logger.info(
                    "[%s][Cycle %d] RSI %.1f extreme — keeping only %s signals (%d filtered)",
                    symbol, cycle, current_rsi, rsi_direction.value, filtered,
                )
                self._log_event("filter", f"RSI {current_rsi:.1f} extreme — only {rsi_direction.value} allowed ({filtered} filtered)", cycle=cycle, instrument=symbol)
                if not signals:
                    self._log_event("filter", "All signals filtered by RSI extreme direction — skipping", cycle=cycle, instrument=symbol)
                    return
            elif htf_bias != "neutral" and signals:
                pre_count = len(signals)
                signals = [
                    s for s in signals
                    if not (htf_bias == "bullish" and s.option_type == OptionType.PUT)
                    and not (htf_bias == "bearish" and s.option_type == OptionType.CALL)
                ]
                filtered = pre_count - len(signals)
                if filtered > 0:
                    logger.info(
                        "[%s][Cycle %d] HTF filter removed %d signal(s) against %s trend",
                        symbol, cycle, filtered, htf_bias,
                    )
                    self._log_event("filter", f"HTF filter removed {filtered} signal(s) against {htf_bias} trend", cycle=cycle, instrument=symbol)
                if not signals:
                    self._log_event("filter", "All signals filtered by HTF — skipping", cycle=cycle, instrument=symbol)
                    await self.alert_manager.send_info(
                        f"HTF FILTER — {symbol}",
                        f"All {pre_count} signal(s) removed by 5-min {htf_bias} trend filter",
                    )
                    return

            # 10. Score and filter (with evaluation boost for top-ranked combos)
            adaptive_min = compute_adaptive_min_score(options_metrics, self.global_bias)

            # Apply insight-based score modifier to threshold
            insight_modifier = self.insight_manager.get_score_modifier()
            adaptive_min = max(35, adaptive_min + insight_modifier)

            # Expiry day awareness: use real expiry date from SmartAPI
            # Raise threshold by 5 to require stronger confirmation on expiry days
            expiry_dt = self._expiry_dates.get(symbol)
            if expiry_dt and expiry_dt == now.date():
                adaptive_min += 5
                logger.debug("[%s] Expiry day (%s) — threshold raised to %d", symbol, expiry_dt, adaptive_min)

            best_signal = None
            best_score = 0.0
            best_score_result = None
            all_scored: list[tuple] = []  # (signal, boosted_score) for near-miss alerts

            for signal in signals:
                score_result = self.signal_scorer.score(
                    signal, df, options_metrics, self.global_bias
                )
                raw_score = score_result.total

                # Evaluation boost: top-ranked strategy+instrument combos
                # get a bonus based on their historical composite score.
                # Max boost = 15 pts for combos with eval score >= 80.
                eval_key = (symbol, signal.strategy.value)
                eval_composite = self._eval_scores.get(eval_key, 0)
                eval_boost = 0
                if eval_composite >= 80:
                    eval_boost = 15
                elif eval_composite >= 70:
                    eval_boost = 10
                elif eval_composite >= 60:
                    eval_boost = 5

                boosted_score = min(raw_score + eval_boost, 100)
                all_scored.append((signal, boosted_score))

                if eval_boost > 0:
                    logger.info(
                        "[%s][Cycle %d] %s scored %.0f + %d eval boost = %.0f/100 (min=%d)",
                        symbol, cycle, signal.strategy.value,
                        raw_score, eval_boost, boosted_score, adaptive_min,
                    )
                    self._log_event("score", f"{signal.strategy.value}: {raw_score:.0f} + {eval_boost} boost = {boosted_score:.0f}/100 (min={adaptive_min})", cycle=cycle, instrument=symbol)
                else:
                    logger.info(
                        "[%s][Cycle %d] %s scored %.0f/100 (min=%d)",
                        symbol, cycle, signal.strategy.value, raw_score, adaptive_min,
                    )
                    self._log_event("score", f"{signal.strategy.value}: {raw_score:.0f}/100 (min={adaptive_min})", cycle=cycle, instrument=symbol)

                if boosted_score >= adaptive_min and boosted_score > best_score:
                    best_signal = signal
                    best_score = boosted_score
                    best_score_result = score_result
                    signal.score = boosted_score

            # If no signal passed the threshold, check if AI can override
            # a marginal signal (score 40+ but below threshold)
            if best_signal is None:
                # AI override path: let AI rescue marginal signals (score >= 40)
                ai_override_candidates = [
                    (s, sc) for s, sc in all_scored if sc >= 40
                ]
                if ai_override_candidates:
                    # Pick the highest-scoring marginal signal
                    ai_override_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_signal = ai_override_candidates[0][0]
                    best_score = ai_override_candidates[0][1]
                    best_score_result = self.signal_scorer.score(
                        best_signal, df, options_metrics, self.global_bias
                    )
                    best_signal.score = best_score
                    # Mark as AI-override path so we require high AI confidence
                    best_signal.details["ai_override_path"] = True
                    logger.info(
                        "[%s][Cycle %d] Score %d < threshold %d — sending to AI for override decision",
                        symbol, cycle, int(best_score), adaptive_min,
                    )
                    self._log_event("score", f"Below threshold ({best_score:.0f}/{adaptive_min}) — AI override path", cycle=cycle, instrument=symbol)
                else:
                    self._log_event("score", f"No signal passed threshold ({adaptive_min})", cycle=cycle, instrument=symbol)
                    # Alert on near-misses (within 5 pts of threshold)
                    near_misses = [(s, sc) for s, sc in all_scored if sc >= adaptive_min - 5]
                    if near_misses:
                        miss_info = "\n".join(
                            f"{s.strategy.value} {s.option_type.value}: {sc:.0f}/{adaptive_min}"
                            for s, sc in near_misses
                        )
                        await self.alert_manager.send_info(
                            f"NEAR MISS — {symbol}",
                            f"Signal(s) close to threshold but not passed:\n{miss_info}",
                        )
                    return

            # 11. Fetch option premium (with bid-ask spread)
            option_quote = await self._fetch_option_quote_for(instrument, best_signal, expiry)
            if option_quote is None or option_quote["ltp"] <= 0:
                logger.warning("[%s] Could not fetch option LTP — skipping trade", symbol)
                self._log_event("option_ltp", "Option LTP fetch FAILED — skipping", cycle=cycle, instrument=symbol)
                return

            option_ltp = option_quote["ltp"]
            best_bid = option_quote["best_bid"]
            best_ask = option_quote["best_ask"]
            spread_pct = option_quote["spread_pct"]

            # 11a. Liquidity gate — skip illiquid strikes with wide bid-ask spread
            if spread_pct > settings.max_spread_pct and best_bid > 0:
                logger.warning(
                    "[%s] Bid-ask spread %.1f%% > %.1f%% threshold — skipping (bid=%.2f ask=%.2f)",
                    symbol, spread_pct, settings.max_spread_pct, best_bid, best_ask,
                )
                self._log_event(
                    "gate",
                    f"LIQUIDITY GATE: spread {spread_pct:.1f}% > {settings.max_spread_pct:.1f}% "
                    f"(bid={best_bid:.2f} ask={best_ask:.2f})",
                    cycle=cycle, instrument=symbol,
                )
                await self.alert_manager.send_info(
                    f"SIGNAL BLOCKED — {symbol} {best_signal.strategy.value}",
                    f"Reason: Bid-ask spread too wide ({spread_pct:.1f}%)\n"
                    f"Bid: {best_bid:.2f} | Ask: {best_ask:.2f} | LTP: {option_ltp:.2f}\n"
                    f"Strike: {int(best_signal.strike_price)} {best_signal.option_type.value}\n"
                    f"Score: {best_score:.0f}",
                )
                return

            # Use ask price for entry (realistic fill) when available, else LTP
            entry_price = best_ask if best_ask > 0 else option_ltp

            logger.info(
                "[%s] Option quote: %s %.0f%s = ₹%.2f (bid=%.2f ask=%.2f spread=%.1f%%)",
                symbol, instrument.symbol, best_signal.strike_price,
                best_signal.option_type.value, option_ltp, best_bid, best_ask, spread_pct,
            )

            # Set ATR-based SL/targets
            atr = indicators.atr
            if atr is None or atr <= 0:
                logger.warning("[%s] ATR unavailable — cannot compute SL/targets, skipping", symbol)
                self._log_event("gate", "ATR unavailable — SL/targets cannot be computed", cycle=cycle, instrument=symbol)
                await self.alert_manager.send_info(
                    f"SIGNAL BLOCKED — {symbol} {best_signal.strategy.value}",
                    f"Reason: ATR not available for position sizing\n"
                    f"Strike: {int(best_signal.strike_price)} {best_signal.option_type.value}\n"
                    f"Score: {best_score:.0f}",
                )
                return
            # ATR-based SL/targets (Wilder: 2×ATR stop, Tharp: 2R+ reward)
            option_atr = atr * 0.5  # ATM delta ~0.5 scales spot ATR to option premium
            best_signal.entry_price = entry_price
            best_signal.stoploss = round(max(entry_price - (2.0 * option_atr), entry_price * 0.70), 2)
            best_signal.target1 = round(entry_price + (2.0 * option_atr), 2)
            best_signal.target2 = round(entry_price + (3.5 * option_atr), 2)

            # 12. AI validation — OR logic:
            #   - Score >= 65: HIGH conviction → skip AI entirely (saves API cost)
            #   - Score 48-64: NORMAL flow → ask AI with adaptive threshold
            #   - Score 40-47: AI OVERRIDE path → ask AI, need confidence >= 80
            is_ai_override = best_signal.details.get("ai_override_path", False)

            if best_score >= 80 and not is_ai_override:
                # HIGH CONVICTION — skip AI gate, system score is strong enough
                logger.info(
                    "[%s][Cycle %d] Score %.0f >= 80 — HIGH CONVICTION, skipping AI gate",
                    symbol, cycle, best_score,
                )
                self._log_event("ai", f"AI BYPASSED — score {best_score:.0f} >= 80 (high conviction)", cycle=cycle, instrument=symbol)
                # Create a synthetic decision for downstream compatibility
                from app.core.models import AIDecision
                decision = AIDecision(
                    trade_decision=True,
                    confidence_score=best_score,  # Use system score as proxy
                    entry_price=option_ltp,
                    stoploss=best_signal.stoploss,
                    target1=best_signal.target1,
                    target2=best_signal.target2,
                    reason=f"High conviction bypass (score={best_score:.0f}, >= 80)",
                )
                await self._save_signal_record(
                    best_signal, best_score, "accepted",
                    decision.confidence_score, decision.reason,
                )
            else:
                # Ask AI for validation
                decision = await self.ai_engine.evaluate(best_signal, snap, best_score, best_score_result)

                # Adaptive AI confidence threshold:
                # - AI override path (score < threshold): need high AI confidence (80)
                # - Normal path: adaptive based on score
                if is_ai_override:
                    ai_threshold = 80  # AI must be very confident to rescue a low score
                elif best_score >= 55:
                    ai_threshold = 60
                else:
                    ai_threshold = 65

                if not decision.trade_decision or decision.confidence_score < ai_threshold:
                    # Override: if AI gave confidence >= threshold but trade_decision=False,
                    # trust the confidence score (AI sometimes contradicts itself)
                    if decision.confidence_score >= ai_threshold and not decision.trade_decision:
                        logger.info(
                            "[%s] AI override: confidence %.0f%% >= %d but trade_decision=False — approving",
                            symbol, decision.confidence_score, ai_threshold,
                        )
                        decision.trade_decision = True
                        self._log_event("ai", f"AI OVERRIDE: confidence {decision.confidence_score:.0f}% >= {ai_threshold} — approving despite trade_decision=False", cycle=cycle, instrument=symbol)
                    else:
                        path_label = "AI-OVERRIDE" if is_ai_override else "NORMAL"
                        logger.info(
                            "[%s] AI rejected [%s path] (confidence=%.0f%%, threshold=%d): %s",
                            symbol, path_label, decision.confidence_score, ai_threshold, decision.reason,
                        )
                        self._log_event("ai", f"AI REJECTED [{path_label}] ({decision.confidence_score:.0f}%/{ai_threshold}): {decision.reason}", cycle=cycle, instrument=symbol, data={
                            "confidence": decision.confidence_score, "reason": decision.reason,
                            "strategy": best_signal.strategy.value, "score": best_score,
                        })
                        await self.alert_manager.send_info(
                            f"SIGNAL REJECTED — {symbol} {best_signal.strategy.value}",
                            f"{symbol} {int(best_signal.strike_price)} {best_signal.option_type.value}\n"
                            f"Premium: ₹{option_ltp:.2f} | Score: {best_score:.0f} | AI: {decision.confidence_score:.0f}%\n"
                            f"Reason: {decision.reason}",
                        )
                        await self._save_signal_record(
                            best_signal, best_score, "rejected",
                            decision.confidence_score, decision.reason,
                        )
                        return

                self._log_event("ai", f"AI APPROVED ({decision.confidence_score:.0f}%): {best_signal.strategy.value} {best_signal.option_type.value}", cycle=cycle, instrument=symbol, data={
                    "confidence": decision.confidence_score, "strategy": best_signal.strategy.value,
                    "strike": best_signal.strike_price, "option_type": best_signal.option_type.value,
                    "entry": decision.entry_price, "sl": decision.stoploss, "target1": decision.target1,
                })
                await self._save_signal_record(
                    best_signal, best_score, "accepted",
                    decision.confidence_score, decision.reason,
                )

            # 13. Build NFO trading symbol
            nfo_symbol = instrument.build_option_symbol(
                expiry or "", best_signal.strike_price, best_signal.option_type.value
            )

            # 14. Position size
            num_lots = self.risk_manager.compute_position_size(
                decision.entry_price, decision.stoploss
            )

            # 15. Enter trade (use instrument-specific lot size)
            trade = self.paper_trader.enter_trade(
                best_signal, decision, nfo_symbol,
                num_lots=num_lots, instrument_lot_size=instrument.lot_size,
            )
            trade.instrument = symbol

            # Store breakout level for failed breakout detection
            trade.breakout_level = _extract_breakout_level(best_signal)

            # 15b. Place real order if live trading
            if not settings.paper_trading:
                # Pre-trade margin check
                margin_ok = await self._check_margin_before_trade(
                    symbol, trade, nfo_symbol
                )
                if not margin_ok:
                    # Remove from paper tracker — order not placed
                    if trade in self.paper_trader.open_trades:
                        self.paper_trader.open_trades.remove(trade)
                    return

                order_resp = await self._place_live_order(
                    instrument, trade, nfo_symbol, OrderSide.BUY,
                )
                if not order_resp or order_resp.status.value == "REJECTED":
                    # Remove from paper tracker if live order failed
                    if trade in self.paper_trader.open_trades:
                        self.paper_trader.open_trades.remove(trade)
                    err_msg = order_resp.message if order_resp else "no response"
                    logger.error("[%s] Live BUY order REJECTED: %s", symbol, err_msg)
                    # Detect margin/fund related rejections and auto-pause
                    if err_msg and any(kw in err_msg.lower() for kw in (
                        "insufficient", "margin", "fund", "balance", "limit exceed",
                    )):
                        await self._auto_pause_live_trading(symbol, err_msg)
                    else:
                        await self.alert_manager.send_info(
                            f"ORDER REJECTED — {symbol}",
                            f"{nfo_symbol}\nError: {err_msg}",
                        )
                    return
                logger.info("[%s] Live BUY order placed: %s", symbol, order_resp.order_id)

            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_signal_alert(best_signal, decision)
            self._log_event("trade", f"TRADE ENTERED: {nfo_symbol} x{num_lots} lots @ ₹{decision.entry_price:.2f} (bid={best_bid:.2f} ask={best_ask:.2f} spread={spread_pct:.1f}%)", cycle=cycle, instrument=symbol, data={
                "symbol": nfo_symbol, "lots": num_lots, "entry": decision.entry_price,
                "sl": decision.stoploss, "target1": decision.target1,
                "live": not settings.paper_trading,
                "best_bid": best_bid, "best_ask": best_ask, "spread_pct": spread_pct,
            })

        except Exception:
            logger.exception("[%s] Error in instrument analysis", symbol)

    # ── Support Methods ──────────────────────────────────────────────────

    def _resolve_instruments(self) -> list[InstrumentConfig]:
        """Resolve which instruments to trade.

        If AUTO_SELECT_INSTRUMENTS is True (default):
          - Uses the latest evaluation recommendations to pick the best
            instrument+strategy combos.
          - Also builds per-instrument strategy whitelists so only the
            proven strategies run during market hours.
        If False or no evaluation data available:
          - Falls back to ACTIVE_INSTRUMENTS from config.
        """
        # Manual override: if auto-select is off, use config
        if not settings.auto_select_instruments:
            return self._resolve_from_config()

        # Auto-select from evaluation data
        recs = self.eval_scheduler.latest_recommendations
        if not recs:
            logger.info("No evaluation data yet — falling back to config / defaults")
            return self._resolve_from_config()

        return self._resolve_from_recommendations(recs)

    def _resolve_from_config(self) -> list[InstrumentConfig]:
        """Resolve instruments from ACTIVE_INSTRUMENTS config."""
        names = settings.get_active_instrument_list()
        instruments = []
        for name in names:
            inst = get_instrument(name)
            if inst:
                instruments.append(inst)
            else:
                logger.warning("Unknown instrument in config: %s — skipping", name)
        if not instruments:
            from app.core.instruments import NIFTY
            instruments = [NIFTY]
            logger.warning("No instruments configured — defaulting to NIFTY")
        # No strategy filter in manual mode — run all
        self._instrument_strategies = {}
        return instruments

    def _resolve_from_recommendations(
        self, recs: list,
    ) -> list[InstrumentConfig]:
        """Pick instruments and per-instrument strategies from evaluation."""
        min_score = settings.min_composite_score
        max_inst = settings.max_active_instruments

        # Group recommendations by instrument, track best strategies
        inst_best: dict[str, float] = {}  # symbol -> best composite score
        inst_strats: dict[str, list[str]] = {}  # symbol -> [strategy names]

        # Only these 3 indices are allowed for trading
        ALLOWED_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY"}

        for r in recs:
            if r.composite_score < min_score:
                continue
            sym = r.instrument
            if sym not in ALLOWED_INDICES:
                continue
            if sym not in inst_best:
                inst_best[sym] = r.composite_score
                inst_strats[sym] = []
            if r.strategy not in inst_strats[sym]:
                inst_strats[sym].append(r.strategy)

        if not inst_best:
            logger.warning(
                "No recommendations above min score (%.0f) — falling back to config",
                min_score,
            )
            return self._resolve_from_config()

        # Rank instruments by their best composite score, pick top N
        ranked = sorted(inst_best.items(), key=lambda x: x[1], reverse=True)
        selected_symbols = [sym for sym, _ in ranked[:max_inst]]

        instruments: list[InstrumentConfig] = []
        strategy_map: dict[str, BaseStrategy] = {
            type(s).__name__: s for s in self.strategies
        }
        # Build a name -> instance lookup using the same names as evaluator
        strat_name_map: dict[str, BaseStrategy] = {}
        from app.backtest.strategy_evaluator import _STRATEGY_REGISTRY
        for name, _ in _STRATEGY_REGISTRY:
            # Find matching instance in self.strategies by class
            for s in self.strategies:
                if type(s).__name__.upper().replace("STRATEGY", "").replace("_", "") == name.replace("_", ""):
                    strat_name_map[name] = s
                    break

        for sym in selected_symbols:
            inst = get_instrument(sym)
            if not inst:
                continue
            instruments.append(inst)

            # Build strategy whitelist for this instrument
            allowed_names = inst_strats.get(sym, [])
            allowed = [strat_name_map[n] for n in allowed_names if n in strat_name_map]
            if allowed:
                self._instrument_strategies[sym] = allowed

        # Build eval score lookup for signal boost
        self._eval_scores = {}
        for r in recs:
            key = (r.instrument, r.strategy)
            if key not in self._eval_scores:
                self._eval_scores[key] = r.composite_score

        logger.info(
            "AUTO-SELECT: Picked %d instruments from evaluation: %s",
            len(instruments),
            ", ".join(
                f"{i.symbol} ({len(self._instrument_strategies.get(i.symbol, self.strategies))} strats)"
                for i in instruments
            ),
        )
        for inst in instruments:
            strats = self._instrument_strategies.get(inst.symbol, self.strategies)
            strat_names = [type(s).__name__ for s in strats]
            # Show eval scores for each strategy
            boost_info = []
            for s in strats:
                for name, _ in _STRATEGY_REGISTRY:
                    if type(s).__name__.upper().replace("STRATEGY", "").replace("_", "") == name.replace("_", ""):
                        sc = self._eval_scores.get((inst.symbol, name), 0)
                        if sc > 0:
                            boost_info.append(f"{name}={sc:.0f}")
                        break
            logger.info("  %s → %s | eval: %s", inst.symbol, ", ".join(strat_names), ", ".join(boost_info) or "none")

        return instruments

    # ── Live Order Execution ─────────────────────────────────────────────

    async def _place_live_order(
        self,
        instrument: InstrumentConfig,
        trade: "Trade",
        nfo_symbol: str,
        side: OrderSide,
    ):
        """Place a real order via AngelOneBroker (live trading only)."""
        token_info = self.client._search_symbol(nfo_symbol)
        if not token_info:
            logger.error("Cannot find token for %s — order not placed", nfo_symbol)
            return None

        request = OrderRequest(
            instrument=instrument,
            trading_symbol=token_info.get("tradingsymbol", nfo_symbol),
            symbol_token=token_info.get("symboltoken", ""),
            exchange="NFO",
            side=side,
            order_type=OrderType.MARKET,
            product_type=ProductType.INTRADAY,
            quantity=trade.lot_size,
            price=0.0,
            trigger_price=0.0,
        )
        try:
            resp = await asyncio.to_thread(self.broker.place_order, request)
            return resp
        except Exception:
            logger.exception("Live order placement failed for %s", nfo_symbol)
            return None

    async def _check_margin_before_trade(
        self,
        symbol: str,
        trade: "Trade",
        nfo_symbol: str,
    ) -> bool:
        """Check broker margin before placing a live order.

        Returns True if margin is sufficient, False otherwise.
        On insufficient margin, auto-pauses live trading and alerts.
        """
        try:
            margin_data = await asyncio.to_thread(self.broker.get_margin)
            if not margin_data:
                logger.warning("[%s] Could not fetch margin data — proceeding with order", symbol)
                return True  # Don't block if margin API is unavailable

            # AngelOne rmsLimit returns: net, availablecash, availableintradaypayin, etc.
            available = 0.0
            for key in ("net", "availablecash", "available_cash"):
                val = margin_data.get(key)
                if val is not None:
                    try:
                        available = float(val)
                        break
                    except (ValueError, TypeError):
                        continue

            min_required = settings.min_margin_required
            if available < min_required:
                logger.warning(
                    "[%s] Insufficient margin: ₹%.2f available, ₹%.2f required",
                    symbol, available, min_required,
                )
                await self._auto_pause_live_trading(
                    symbol,
                    f"Available margin ₹{available:,.2f} is below minimum ₹{min_required:,.2f}",
                )
                return False

            # Reset pause flag if margin is healthy again
            self._live_paused_insufficient_margin = False
            return True

        except Exception:
            logger.exception("[%s] Margin check failed — proceeding with order", symbol)
            return True  # Don't block on margin API errors

    async def _auto_pause_live_trading(self, symbol: str, reason: str) -> None:
        """Auto-switch to paper trading mode due to insufficient funds."""
        if self._live_paused_insufficient_margin:
            # Already paused and alerted — don't spam
            logger.debug("[%s] Live trading already paused — skipping duplicate alert", symbol)
            return

        self._live_paused_insufficient_margin = True
        settings.paper_trading = True
        logger.warning(
            "AUTO-PAUSED live trading → paper mode. Reason: %s", reason,
        )
        await self.alert_manager.send_info(
            "⚠️ LIVE TRADING AUTO-PAUSED",
            f"Switched to PAPER mode automatically.\n\n"
            f"Reason: {reason}\n"
            f"Instrument: {symbol}\n\n"
            f"No real orders will be placed until you manually re-enable live mode.\n"
            f"Add funds to your AngelOne account and toggle back to LIVE from the dashboard.",
        )

    async def _place_live_exit(
        self,
        instrument: Optional[InstrumentConfig],
        trade: "Trade",
    ) -> None:
        """Place sell order for an exiting trade (live trading only)."""
        token_info = self.client._search_symbol(trade.symbol)
        if not token_info:
            logger.error("Cannot find token for exit: %s", trade.symbol)
            return
        # Resolve instrument config for the order request
        if instrument is None:
            inst_sym = getattr(trade, "instrument", "NIFTY")
            instrument = get_instrument(inst_sym)
            if instrument is None:
                from app.core.instruments import NIFTY
                instrument = NIFTY

        request = OrderRequest(
            instrument=instrument,
            trading_symbol=token_info.get("tradingsymbol", trade.symbol),
            symbol_token=token_info.get("symboltoken", ""),
            exchange="NFO",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            product_type=ProductType.INTRADAY,
            quantity=trade.lot_size,
            price=0.0,
            trigger_price=0.0,
        )
        try:
            resp = await asyncio.to_thread(self.broker.place_order, request)
            if resp and resp.status.value != "REJECTED":
                logger.info("Live SELL order placed for %s: %s", trade.symbol, resp.order_id)
            else:
                logger.error("Live SELL order REJECTED for %s: %s", trade.symbol, resp.message if resp else "no response")
                await self.alert_manager.send_info(
                    f"EXIT ORDER REJECTED — {trade.symbol}",
                    f"Failed to close position. Manual intervention needed.\n{resp.message if resp else 'no response'}",
                )
        except Exception:
            logger.exception("Live exit order failed for %s", trade.symbol)
            await self.alert_manager.send_info(
                f"EXIT ORDER ERROR — {trade.symbol}",
                "Exception during exit order placement. Check positions manually.",
            )

    async def _recover_open_trades(self) -> None:
        """Recover open trades from DB after a mid-day restart.

        Reloads trades with status='open' for today into the paper trader's
        open_trades list so exit monitoring continues seamlessly.
        """
        try:
            today_trades = await self.trade_logger.get_today_trades()
            open_trades = [t for t in today_trades if t.status.value == "open"]
            if open_trades:
                for trade in open_trades:
                    # Avoid duplicates
                    if not any(t.trade_id == trade.trade_id for t in self.paper_trader.open_trades):
                        self.paper_trader.open_trades.append(trade)
                logger.info("Recovered %d open trades from DB", len(open_trades))
                await self.alert_manager.send_info(
                    "CRASH RECOVERY",
                    f"Recovered {len(open_trades)} open trades from database.\n"
                    + "\n".join(f"  {t.symbol} @ {t.entry_price}" for t in open_trades),
                )
            else:
                logger.info("No open trades to recover from DB")
        except Exception:
            logger.exception("Failed to recover open trades from DB")

    async def _fetch_option_ltp_for(
        self,
        instrument: InstrumentConfig,
        signal: StrategySignal,
        expiry: Optional[str],
    ) -> Optional[float]:
        """Fetch the real LTP for a specific option contract from AngelOne."""
        quote = await self._fetch_option_quote_for(instrument, signal, expiry)
        return quote["ltp"] if quote else None

    async def _fetch_option_quote_for(
        self,
        instrument: InstrumentConfig,
        signal: StrategySignal,
        expiry: Optional[str],
    ) -> Optional[dict]:
        """Fetch full quote (ltp, bid, ask, spread) for a specific option contract.

        Returns:
            dict with keys: ltp, best_bid, best_ask, spread, spread_pct
            or None if fetch fails.
        """
        if not expiry:
            return None
        symbol = instrument.build_option_symbol(
            expiry, signal.strike_price, signal.option_type.value
        )
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            logger.warning("Token not found for %s", symbol)
            return None
        try:
            quote = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.get_option_quote,
                    "NFO",
                    token_info.get("tradingsymbol", ""),
                    token_info.get("symboltoken", ""),
                ),
                timeout=15,
            )
        except asyncio.TimeoutError:
            logger.warning("Quote fetch timed out for %s", symbol)
            return None
        return quote

    async def _update_options_chain_for(
        self,
        instrument: InstrumentConfig,
        spot_price: float,
        expiry: str,
    ) -> OptionsMetrics:
        """Fetch options chain and compute metrics for an instrument."""
        try:
            chain = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.get_option_chain,
                    expiry,
                    symbol_prefix=instrument.option_symbol_prefix or instrument.symbol,
                    spot_price=spot_price,
                    strike_interval=instrument.strike_interval,
                ),
                timeout=60,
            )
            # Cache chain for V2 GEX calculation
            self._last_option_chain[instrument.symbol] = chain
            metrics = self.feature_engine.compute_options_metrics(chain, spot_price)
            logger.info(
                "[%s] Options: PCR=%s MaxPain=%s",
                instrument.symbol,
                f"{metrics.pcr:.2f}" if metrics.pcr is not None else "N/A",
                f"{metrics.max_pain:.0f}" if metrics.max_pain is not None else "N/A",
            )
            return metrics
        except asyncio.TimeoutError:
            logger.warning("[%s] Options chain fetch timed out after 60s", instrument.symbol)
            return self._options_metrics.get(instrument.symbol, OptionsMetrics())
        except Exception:
            logger.exception("[%s] Error updating options chain", instrument.symbol)
            return self._options_metrics.get(instrument.symbol, OptionsMetrics())

    async def _check_trade_exits_for(
        self,
        instrument: InstrumentConfig,
        spot_price: float,
    ) -> None:
        """Check open trades for this instrument for exit conditions."""
        inst_trades = [
            t for t in self.paper_trader.open_trades
            if getattr(t, "instrument", "NIFTY") == instrument.symbol
        ]
        if not inst_trades:
            return

        expiry = self._expiries.get(instrument.symbol, "")
        current_prices: dict[str, float] = {}
        for trade in inst_trades:
            token_info = self.client._search_symbol(trade.symbol)
            if token_info:
                try:
                    quote = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.get_option_quote,
                            "NFO",
                            token_info.get("tradingsymbol", ""),
                            token_info.get("symboltoken", ""),
                        ),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Quote fetch timed out for %s", trade.symbol)
                    quote = None
                if quote and quote["ltp"] > 0:
                    # Use bid price for exit (realistic sell fill), fall back to LTP
                    exit_price = quote["best_bid"] if quote["best_bid"] > 0 else quote["ltp"]
                    current_prices[trade.symbol] = exit_price

        # Failed breakout detection: exit early if spot reverses through breakout level
        # Require 0.15% buffer past breakout level to avoid wick-trigger exits
        for trade in list(inst_trades):
            if trade.breakout_level is None:
                continue
            if trade.symbol not in current_prices:
                continue
            is_call = trade.option_type == OptionType.CALL
            buffer = trade.breakout_level * 0.0015  # 0.15% buffer
            failed = (is_call and spot_price < trade.breakout_level - buffer) or \
                     (not is_call and spot_price > trade.breakout_level + buffer)
            if failed:
                trade.exit_price = current_prices[trade.symbol]
                self.paper_trader._close_trade(trade, "failed_breakout")
                if not settings.paper_trading:
                    await self._place_live_exit(instrument, trade)
                await self.trade_logger.log_trade(trade)
                await self.alert_manager.send_exit_alert(trade)
                logger.info(
                    "[%s] FAILED BREAKOUT EXIT: %s spot=%.2f %s breakout=%.2f | PnL=%.2f",
                    instrument.symbol, trade.symbol, spot_price,
                    "below" if is_call else "above", trade.breakout_level,
                    trade.pnl or 0,
                )

        closed = self.paper_trader.check_exits(current_prices)
        for trade in closed:
            # Place live sell order if not paper trading
            if not settings.paper_trading:
                await self._place_live_exit(instrument, trade)
            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_exit_alert(trade)

    async def _fetch_global_data(self) -> None:
        """Fetch and analyze global market indices, FII/DII, and breadth."""
        try:
            self.global_indices = await fetch_global_indices()
            self.global_bias = compute_global_bias(self.global_indices)
            self._global_last_fetched = datetime.now(IST)
            logger.info("Global bias: %s (%d indices)", self.global_bias.value, len(self.global_indices))
            self._set_source("global_indices", "ok", f"{len(self.global_indices)} indices, bias={self.global_bias.value}")
            self._log_event("data", f"Global indices: {len(self.global_indices)} fetched, bias={self.global_bias.value}")

            # Update shared state so API can serve index details
            from app.api.routes import get_state
            state = get_state()
            state["global_indices"] = [
                {"symbol": i.symbol, "change_pct": i.change_pct, "last_price": i.last_price}
                for i in self.global_indices
            ]
        except Exception:
            logger.exception("Error fetching global data")
            self._set_source("global_indices", "error", "Fetch failed")
            self._log_event("data", "Global indices fetch FAILED")

        # Fetch FII/DII data (non-blocking — don't fail if unavailable)
        try:
            from app.data.institutional import fetch_fii_dii_data
            fii_data = await fetch_fii_dii_data()
            if fii_data:
                self.pre_market_analyst._institutional_flow = fii_data
                logger.info("FII/DII: FII=%+.0fcr DII=%+.0fcr signal=%s",
                            fii_data.fii_net, fii_data.dii_net, fii_data.signal)
                self._set_source("fii_dii", "ok", f"FII={fii_data.fii_net:+.0f}Cr, signal={fii_data.signal}")
                self._log_event("data", f"FII/DII refreshed: FII={fii_data.fii_net:+.0f}Cr DII={fii_data.dii_net:+.0f}Cr signal={fii_data.signal}")
            else:
                self._set_source("fii_dii", "warn", "No data returned")
        except Exception:
            logger.warning("FII/DII fetch failed (non-critical)")
            self._set_source("fii_dii", "error", "Fetch failed")

        # Fetch market breadth (non-blocking)
        try:
            from app.data.market_breadth import fetch_market_breadth
            breadth = await fetch_market_breadth()
            if breadth:
                self.pre_market_analyst._market_breadth = breadth
                logger.info("Breadth: A/D=%.2f signal=%s",
                            breadth.advance_decline_ratio, breadth.breadth_signal)
                self._set_source("breadth", "ok", f"A/D={breadth.advance_decline_ratio:.2f}, signal={breadth.breadth_signal}")
                self._log_event("data", f"Breadth refreshed: A/D={breadth.advance_decline_ratio:.2f} signal={breadth.breadth_signal}")
            else:
                self._set_source("breadth", "warn", "No data returned")
        except Exception:
            logger.warning("Market breadth fetch failed (non-critical)")
            self._set_source("breadth", "error", "Fetch failed")

    async def _update_options_chain(self, spot_price: float) -> None:
        """Legacy: update options chain for all index instruments."""
        for inst in self._active_instruments:
            if inst.is_index:
                expiry = self._expiries.get(inst.symbol)
                if expiry:
                    self._options_metrics[inst.symbol] = await self._update_options_chain_for(
                        inst, spot_price, expiry
                    )

    async def _check_trade_exits(self, spot_price: float) -> None:
        """Check all open trades for stoploss/target hits."""
        if not self.paper_trader.open_trades:
            return

        current_prices: dict[str, float] = {}
        for trade in self.paper_trader.open_trades:
            token_info = self.client._search_symbol(trade.symbol)
            if token_info:
                ltp = self.client.get_ltp(
                    "NFO",
                    token_info.get("tradingsymbol", ""),
                    token_info.get("symboltoken", ""),
                )
                if ltp:
                    current_prices[trade.symbol] = ltp

        closed = self.paper_trader.check_exits(current_prices)
        for trade in closed:
            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_exit_alert(trade)

    # ── V2 Engine ─────────────────────────────────────────────────────────

    async def _v2_analyze_instrument(
        self, instrument: InstrumentConfig, cycle: int, now: datetime
    ) -> None:
        """V2 engine analysis — day-classified strategy execution.

        Uses shared data (snapshots, df_today, options) computed by V1.
        Runs V2 strategies gated by day type, with independent paper trader.
        """
        symbol = instrument.symbol
        snap = self.snapshots.get(symbol)
        df_today = self._df_today_cache.get(symbol)
        if not snap or df_today is None or df_today.empty:
            return

        try:
            # Day classification at 10:00 (once per day)
            if not self.v2_day_classified and now.time() >= dtime(10, 0):
                vix_val = None
                for idx_data in self.global_indices:
                    if hasattr(idx_data, 'symbol') and 'VIX' in idx_data.symbol:
                        vix_val = idx_data.last_price
                        break
                self.v2_day_type = self.v2_day_classifier.classify(df_today, snap, vix_val)
                self.v2_day_classified = True
                self._log_event(
                    "v2_classify",
                    f"V2 Day classified: {self.v2_day_type.value}",
                    cycle=cycle, instrument=symbol,
                )
                logger.info("[V2][%s] Day classified as: %s", symbol, self.v2_day_type.value)

            # Skip if day is UNCLEAR or not yet classified
            if self.v2_day_type == DayType.PENDING:
                return
            if self.v2_day_type == DayType.UNCLEAR and settings.v2_skip_unclear_days:
                return

            # Check V2 exits on open trades
            await self._v2_check_exits(instrument, snap)

            # Time gate: no new entries after 14:00 (V2 is more conservative)
            if now.time() >= dtime(14, 0):
                return

            # Risk check (V2 independent limits)
            if not self.v2_risk_manager.can_trade(
                self.v2_paper_trader.all_today_trades,
                open_count=len(self.v2_paper_trader.open_trades),
            ):
                return

            # ── V2 Strategy execution ──────────────────────────────────────
            # Gate strategies by day type:
            #   TREND    → VWAP_PULLBACK
            #   RANGE    → GEX_BOUNCE
            #   VOLATILE → RSI_EXTREME
            options_metrics = self._options_metrics.get(symbol, OptionsMetrics())
            spot_price = snap.price or snap.nifty_price
            daily_levels = self._daily_levels.get(symbol)

            # Compute GEX for RANGE days (feed to GEX Bounce strategy)
            if self.v2_day_type == DayType.RANGE:
                chain = self._last_option_chain.get(symbol, [])
                if chain:
                    expiry_dt = self._expiry_dates.get(symbol)
                    gex_result = self.v2_gex_calculator.compute(chain, spot_price, expiry_dt)
                    self.v2_gex_bounce.set_gex_result(gex_result)
            else:
                self.v2_gex_bounce.set_gex_result(None)

            v2_signals: list[StrategySignal] = []
            for strategy in self.v2_strategies:
                # Day-type gating
                strat_name = type(strategy).__name__
                if self.v2_day_type == DayType.TREND and strat_name != "VWAPPullbackStrategy":
                    continue
                if self.v2_day_type == DayType.VOLATILE and strat_name != "RSIExtremeStrategy":
                    continue
                if self.v2_day_type == DayType.RANGE and strat_name != "GEXBounceStrategy":
                    continue

                signal = strategy.evaluate(df_today, options_metrics, spot_price, daily_levels=daily_levels)
                if signal:
                    signal.instrument = symbol
                    v2_signals.append(signal)

            if not v2_signals:
                if cycle % 15 == 1:
                    self._log_event(
                        "v2_status",
                        f"V2 active: day={self.v2_day_type.value}, no signals, "
                        f"trades={len(self.v2_paper_trader.all_today_trades)}",
                        cycle=cycle, instrument=symbol,
                    )
                return

            # Score V2 signals (reuse V1 scorer)
            best_signal = None
            best_score = 0.0
            best_score_result = None
            for signal in v2_signals:
                score_result = self.v2_signal_scorer.score(
                    signal, df_today, options_metrics, self.global_bias
                )
                if score_result.total > best_score:
                    best_score = score_result.total
                    best_signal = signal
                    best_score_result = score_result
                    signal.score = score_result.total

            if best_signal is None or best_score < 40:
                self._log_event(
                    "v2_score",
                    f"V2 signal below threshold: {best_score:.0f}/40",
                    cycle=cycle, instrument=symbol,
                )
                return

            self._log_event(
                "v2_signal",
                f"V2 {best_signal.strategy.value} {best_signal.option_type.value}: score={best_score:.0f}",
                cycle=cycle, instrument=symbol,
            )

            # Fetch option premium (with bid-ask spread)
            expiry = self._expiries.get(symbol, "")
            option_quote = await self._fetch_option_quote_for(instrument, best_signal, expiry)
            if option_quote is None or option_quote["ltp"] <= 0:
                self._log_event("v2_ltp", "V2 option LTP fetch failed", cycle=cycle, instrument=symbol)
                return

            option_ltp = option_quote["ltp"]
            best_bid = option_quote["best_bid"]
            best_ask = option_quote["best_ask"]
            spread_pct = option_quote["spread_pct"]

            # Liquidity gate — skip illiquid strikes
            if spread_pct > settings.max_spread_pct and best_bid > 0:
                self._log_event(
                    "v2_gate",
                    f"V2 LIQUIDITY GATE: spread {spread_pct:.1f}% > {settings.max_spread_pct:.1f}% "
                    f"(bid={best_bid:.2f} ask={best_ask:.2f})",
                    cycle=cycle, instrument=symbol,
                )
                return

            # Use ask price for entry (realistic fill) when available, else LTP
            entry_price = best_ask if best_ask > 0 else option_ltp

            # V2 SL/targets: tighter and more conservative
            atr = snap.indicators.atr
            if atr is None or atr <= 0:
                return
            best_signal.entry_price = entry_price
            best_signal.stoploss = round(max(
                entry_price - (settings.v2_stoploss_pct / 100 * entry_price),
                entry_price * 0.70,
            ), 2)
            best_signal.target1 = round(entry_price + (settings.v2_quick_target_pct / 100 * entry_price), 2)
            best_signal.target2 = round(entry_price + (settings.v2_quick_target_pct * 1.5 / 100 * entry_price), 2)

            # AI validation (V2 uses configured model)
            decision = await self.ai_engine.evaluate(best_signal, snap, best_score, best_score_result)
            if not decision.trade_decision or decision.confidence_score < 60:
                self._log_event(
                    "v2_ai",
                    f"V2 AI rejected: {decision.confidence_score:.0f}% — {decision.reason}",
                    cycle=cycle, instrument=symbol,
                )
                return

            # Enter V2 paper trade
            logger.info(
                "[V2][%s] ENTERING: %s %s %.0f%s | Entry=%.2f SL=%.2f T1=%.2f (bid=%.2f ask=%.2f spread=%.1f%%)",
                symbol, best_signal.strategy.value, symbol,
                best_signal.strike_price, best_signal.option_type.value,
                entry_price, best_signal.stoploss, best_signal.target1,
                best_bid, best_ask, spread_pct,
            )

            nfo_symbol = instrument.build_option_symbol(
                expiry or "", best_signal.strike_price, best_signal.option_type.value
            )
            trade = self.v2_paper_trader.enter_trade(
                best_signal, decision,
                nfo_symbol=nfo_symbol,
                instrument_lot_size=instrument.lot_size,
            )
            trade.breakout_level = _extract_breakout_level(best_signal)
            # Stamp V2 fields
            trade.engine = "v2"
            trade.instrument = symbol
            trade.entry_datetime = datetime.now(IST)
            trade.max_hold_minutes = settings.v2_max_hold_minutes
            trade.day_type = self.v2_day_type.value

            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_entry_alert(trade, best_score)
            self._log_event(
                "v2_trade",
                f"V2 TRADE: {trade.symbol} {best_signal.strategy.value} | "
                f"Entry={entry_price:.2f} SL={trade.stoploss:.2f} T1={trade.target1:.2f} "
                f"(bid={best_bid:.2f} ask={best_ask:.2f} spread={spread_pct:.1f}%)",
                cycle=cycle, instrument=symbol,
            )

        except Exception:
            logger.exception("[V2][%s] Error in V2 analysis", symbol)

    async def _v2_check_exits(
        self, instrument: InstrumentConfig, snap: MarketSnapshot
    ) -> None:
        """Check V2 open trades for exits using SmartExitEngine."""
        v2_trades = [
            t for t in self.v2_paper_trader.open_trades
            if getattr(t, "instrument", "NIFTY") == instrument.symbol
        ]
        if not v2_trades:
            return

        current_prices: dict[str, float] = {}
        for trade in v2_trades:
            token_info = self.client._search_symbol(trade.symbol)
            if token_info:
                try:
                    quote = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.get_option_quote,
                            "NFO",
                            token_info.get("tradingsymbol", ""),
                            token_info.get("symboltoken", ""),
                        ),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    quote = None
                if quote and quote["ltp"] > 0:
                    # Use bid price for exit (realistic sell fill), fall back to LTP
                    exit_price = quote["best_bid"] if quote["best_bid"] > 0 else quote["ltp"]
                    current_prices[trade.symbol] = exit_price

        spot_price = snap.price or snap.nifty_price

        for trade in list(v2_trades):
            ltp = current_prices.get(trade.symbol)
            if ltp is None:
                continue

            result = self.v2_smart_exit.evaluate(
                trade=trade,
                current_ltp=ltp,
                snap=snap,
                day_type=self.v2_day_type,
                spot_price=spot_price,
            )

            # Update trailing SL (even if not exiting)
            if result.new_stoploss is not None and not result.should_exit:
                if result.new_stoploss > trade.stoploss:
                    logger.debug(
                        "[V2] SL UPDATE: %s | Old=%.2f → New=%.2f | %s",
                        trade.symbol, trade.stoploss, result.new_stoploss, result.reason,
                    )
                    trade.stoploss = result.new_stoploss

            if result.should_exit:
                trade.exit_price = result.exit_price
                trade.exit_type = result.exit_type
                self.v2_paper_trader._close_trade(trade, result.exit_type)
                await self.trade_logger.log_trade(trade)
                await self.alert_manager.send_exit_alert(trade)
                logger.info(
                    "[V2] EXIT [%s]: %s | PnL=%.2f | %s",
                    result.exit_type, trade.symbol, trade.pnl or 0, result.reason,
                )
                self._log_event(
                    "v2_exit",
                    f"{result.exit_type}: {trade.symbol}, PnL={trade.pnl or 0:.2f} — {result.reason}",
                    instrument=instrument.symbol,
                )

    async def _close_all_trades(self) -> None:
        """Close all open trades at 15:20 (both V1 and V2)."""
        all_traders = [("v1", self.paper_trader), ("v2", self.v2_paper_trader)]
        for engine_label, trader in all_traders:
            if not trader.open_trades:
                continue

            current_prices: dict[str, float] = {}
            for trade in trader.open_trades:
                token_info = self.client._search_symbol(trade.symbol)
                if token_info:
                    quote = self.client.get_option_quote(
                        "NFO",
                        token_info.get("tradingsymbol", ""),
                        token_info.get("symboltoken", ""),
                    )
                    if quote and quote["ltp"] > 0:
                        # Use bid for EOD exit (realistic sell fill)
                        exit_price = quote["best_bid"] if quote["best_bid"] > 0 else quote["ltp"]
                        current_prices[trade.symbol] = exit_price

            closed = trader.close_all_open(current_prices)
            for trade in closed:
                trade.exit_type = "eod"
                if not settings.paper_trading and engine_label == "v1":
                    await self._place_live_exit(None, trade)
                await self.trade_logger.log_trade(trade)
                await self.alert_manager.send_exit_alert(trade)

            logger.info("[%s] All trades closed for EOD. Count: %d", engine_label.upper(), len(closed))

        # Disconnect WebSocket at end of day
        self.client.stop_websocket()

    async def _generate_daily_report(self) -> None:
        """Generate and send daily performance report."""
        try:
            today_trades = await self.trade_logger.get_today_trades()
            metrics = await self.trade_logger.compute_performance(today_trades)
            await self.trade_logger.save_daily_report(metrics)

            report = (
                "📊 DAILY REPORT\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"Date: {datetime.now(IST).strftime('%Y-%m-%d')}\n"
                f"Total Trades: {metrics.total_trades}\n"
                f"Winners: {metrics.winning_trades}\n"
                f"Losers: {metrics.losing_trades}\n"
                f"Win Rate: {metrics.win_rate:.1f}%\n"
                f"Total PnL: ₹{metrics.total_pnl:,.2f}\n"
                f"Profit Factor: {metrics.profit_factor:.2f}\n"
                f"Max Drawdown: ₹{metrics.max_drawdown:,.2f}\n"
                "━━━━━━━━━━━━━━━━━━"
            )

            await self.alert_manager.send_daily_report(report)
            logger.info("Daily report generated and sent")
            logger.info(report)

        except Exception:
            logger.exception("Error generating daily report")

    async def _run_premarket_evaluation(self) -> None:
        """Run strategy evaluation before market open to auto-select instruments.

        Evaluates ALL registered instruments so the system can pick the best ones.
        Skips if a recent evaluation (from today or yesterday post-market) already exists.
        """
        try:
            recs = self.eval_scheduler.latest_recommendations
            if recs:
                eval_date = recs[0].eval_date if recs else ""
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                yesterday = (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")
                if eval_date in (today_str, yesterday):
                    logger.info(
                        "Recent evaluation found (date=%s, %d recs) — skipping pre-market eval",
                        eval_date, len(recs),
                    )
                    return

            logger.info("Running pre-market evaluation on ALL registered instruments...")
            all_instruments = get_enabled_instruments()
            report = await self.eval_scheduler.run_evaluation(all_instruments)
            logger.info(
                "Pre-market evaluation done: %d recommendations from %d instruments",
                len(report.recommendations), len(all_instruments),
            )
        except Exception:
            logger.exception("Error in pre-market evaluation — will use existing data or config fallback")

    async def _run_post_market_evaluation(self) -> None:
        """Run strategy evaluation after daily report — ranks strategies for next day."""
        try:
            logger.info("Starting post-market strategy evaluation...")
            all_instruments = get_enabled_instruments()
            report = await self.eval_scheduler.run_evaluation(all_instruments)

            # Update shared state so the API can serve recommendations
            from app.api.routes import get_state
            state = get_state()
            state["eval_scheduler"] = self.eval_scheduler

            if report.recommendations:
                top = report.recommendations[0]
                logger.info(
                    "Evaluation done: %d recs | Top: %s on %s (score=%.1f)",
                    len(report.recommendations), top.strategy,
                    top.instrument, top.composite_score,
                )
        except Exception:
            logger.exception("Error in post-market strategy evaluation")

    # NOTE: No hardcoded expiry day fallback — expiry dates come exclusively from
    # SmartAPI instrument master via get_nearest_weekly_expiry(). If the instrument
    # master is unavailable, options data is simply marked as unavailable for that
    # instrument (no guessing).
