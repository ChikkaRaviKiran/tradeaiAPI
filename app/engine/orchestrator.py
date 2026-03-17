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
from datetime import datetime, time as dtime, timedelta
from typing import Optional

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
)
from app.data.angelone_client import AngelOneClient
from app.data.global_markets import compute_global_bias, fetch_global_indices
from app.data.validator import DataValidator
from app.engine.ai_decision import AIDecisionEngine
from app.engine.feature_engine import FeatureEngine
from app.engine.regime_detector import RegimeDetector
from app.engine.signal_scorer import SignalScorer, MIN_SCORE, compute_adaptive_min_score
from app.strategies.base import BaseStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.strategies.ema_breakout import EMABreakoutStrategy
from app.strategies.orb import ORBStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.strategies.breakout_20d import Breakout20DStrategy
from app.trading.history_logger import HistoryLogger
from app.trading.paper_trader import PaperTradingEngine
from app.trading.risk_manager import RiskManager
from app.trading.trade_logger import TradeLogger
from app.backtest.scheduler import EvaluationScheduler

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
PRE_CLOSE = dtime(15, 20)
REPORT_TIME = dtime(15, 30)
GLOBAL_FETCH_TIME = dtime(9, 0)
LOAD_TIME = dtime(8, 45)

LOOP_INTERVAL_SECONDS = 60  # 1-minute analysis cycle


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
        self.trade_logger = TradeLogger()
        self.history_logger = HistoryLogger()
        self.alert_manager = AlertManager()
        self.eval_scheduler = EvaluationScheduler(lookback_days=20)

        # Strategies (applied to every instrument)
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

        # Active instruments — resolved from evaluation or config
        self._active_instruments: list[InstrumentConfig] = []
        # Per-instrument strategy whitelist from evaluation
        # {symbol: [strategy_instance, ...]}  — empty means run all
        self._instrument_strategies: dict[str, list[BaseStrategy]] = {}

        # State — per instrument
        self.running = False
        self._cycle_count = 0
        self._global_last_fetched: Optional[datetime] = None
        self.global_bias = GlobalBias.UNAVAILABLE
        self.global_indices: list = []
        # Per-instrument snapshots  {symbol: MarketSnapshot}
        self.snapshots: dict[str, MarketSnapshot] = {}
        # Per-instrument options metrics  {symbol: OptionsMetrics}
        self._options_metrics: dict[str, OptionsMetrics] = {}
        # Per-instrument expiry  {symbol: str}
        self._expiries: dict[str, str] = {}
        # Backward compat
        # Backward compat
        self.snapshot: Optional[MarketSnapshot] = None

        # Heartbeat tracking
        self._last_heartbeat_cycle = 0
        self._consecutive_no_signal = 0

    async def start(self) -> None:
        """Main entry point — runs forever, restarting each trading day."""
        # Load previous evaluation from DB on cold start
        try:
            await self.eval_scheduler.load_latest_from_db()
        except Exception:
            logger.warning("Could not load previous evaluation from DB")

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
        self._options_metrics = {}
        self._expiries = {}
        self.snapshot = None
        self._last_heartbeat_cycle = 0
        self._consecutive_no_signal = 0
        self.paper_trader = PaperTradingEngine()  # Fresh daily paper trader
        self.running = False
        logger.info("Daily state reset complete")

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

        # Authenticate with AngelOne
        if not self.client.authenticate():
            logger.error("Failed to authenticate. Aborting today.")
            await self.alert_manager.send_info("AUTH FAILED", "AngelOne authentication failed. Check credentials.")
            return

        # Determine weekly expiry for each index instrument
        for inst in self._active_instruments:
            if inst.is_index:
                expiry = self.client.get_nearest_weekly_expiry()
                if not expiry:
                    expiry = self._get_weekly_expiry_fallback()
                self._expiries[inst.symbol] = expiry
                logger.info("Weekly expiry for %s: %s", inst.symbol, expiry)

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

        try:
            # Analyze each active instrument
            for instrument in self._active_instruments:
                await self._analyze_instrument(instrument, cycle, now)

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
                return
            df = self.client.candles_to_dataframe(candles)

            if df.empty:
                logger.warning("[%s][Cycle %d] No candle data received", symbol, cycle)
                return

            # 2. Validate data
            df = self.validator.validate_candles(df, is_index=instrument.is_index)
            if not self.validator.is_valid_for_trading(df):
                logger.warning("[%s][Cycle %d] Data validation failed", symbol, cycle)
                return

            # 2b. For indices, merge futures volume
            if instrument.is_index:
                try:
                    fut_candles = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.get_nifty_futures_candles,
                            interval="ONE_MINUTE", from_date=from_date, to_date=to_date,
                        ),
                        timeout=45,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[%s] Futures candle fetch timed out", symbol)
                    fut_candles = []
                if fut_candles:
                    fut_df = self.client.candles_to_dataframe(fut_candles)
                    df = self.feature_engine.merge_futures_volume(df, fut_df)

            # 3. Feature engineering (on full history for proper indicator warmup)
            df = self.feature_engine.compute_indicators(df, today_date=today_str)
            # Filter to today-only for strategies/snapshot (indicators are already computed)
            df_today = df[df.index.strftime("%Y-%m-%d") == today_str]
            if df_today.empty:
                logger.warning("[%s][Cycle %d] No today candles after filtering", symbol, cycle)
                return
            indicators = self.feature_engine.get_latest_indicators(df_today)
            spot_price = df_today.iloc[-1]["close"] if not df_today.empty else 0

            logger.info(
                "[%s][Cycle %d] Price=%.2f | RSI=%s | ADX=%s | EMA200=%s",
                symbol, cycle, spot_price,
                f"{indicators.rsi:.1f}" if indicators.rsi is not None else "N/A",
                f"{indicators.adx:.1f}" if indicators.adx is not None else "N/A",
                f"{indicators.ema200:.1f}" if indicators.ema200 is not None else "N/A",
            )

            # 4. Options chain (only for index instruments, every 5 min)
            options_metrics = self._options_metrics.get(symbol, OptionsMetrics())
            expiry = self._expiries.get(symbol)
            if instrument.is_index and expiry and (now.minute % 5 == 0 or options_metrics.pcr is None):
                options_metrics = await self._update_options_chain_for(instrument, spot_price, expiry)
                self._options_metrics[symbol] = options_metrics

            # 5. Market regime detection
            regime = self.regime_detector.detect(df)

            # 6. Build market snapshot (use today's data)
            snap = MarketSnapshot(
                instrument=symbol,
                price=spot_price,
                nifty_price=spot_price if symbol == "NIFTY" else 0,
                vwap=indicators.vwap,
                regime=regime,
                global_bias=self.global_bias,
                indicators=indicators,
                options_metrics=options_metrics,
                timestamp=now,
            )
            self.snapshots[symbol] = snap
            await self.history_logger.save_snapshot(snap)

            # 7. Check exits on open trades for this instrument
            await self._check_trade_exits_for(instrument, spot_price)

            # 8. Global risk check
            if not self.risk_manager.can_trade(
                self.paper_trader.all_today_trades,
                open_count=len(self.paper_trader.open_trades),
            ):
                logger.info("[%s][Cycle %d] Risk limits reached", symbol, cycle)
                return

            # 9. Run strategies (use today-only data for strategy evaluation)
            strategies_to_run = self._instrument_strategies.get(symbol, self.strategies)
            signals: list[StrategySignal] = []
            for strategy in strategies_to_run:
                signal = strategy.evaluate(df_today, options_metrics, spot_price)
                if signal:
                    signal.instrument = symbol
                    signals.append(signal)

            if not signals:
                self._consecutive_no_signal += 1
                return
            else:
                self._consecutive_no_signal = 0

            # 10. Score and filter
            adaptive_min = compute_adaptive_min_score(options_metrics, self.global_bias)
            best_signal = None
            best_score = 0.0

            for signal in signals:
                score_result = self.signal_scorer.score(
                    signal, df, options_metrics, self.global_bias
                )
                logger.info(
                    "[%s][Cycle %d] %s scored %.0f/100 (min=%d)",
                    symbol, cycle, signal.strategy.value, score_result.total, adaptive_min,
                )
                if score_result.total >= adaptive_min and score_result.total > best_score:
                    best_signal = signal
                    best_score = score_result.total
                    signal.score = score_result.total

            if best_signal is None:
                return

            # 11. Fetch option premium
            option_ltp = await self._fetch_option_ltp_for(instrument, best_signal, expiry)
            if option_ltp is None or option_ltp <= 0:
                logger.warning("[%s] Could not fetch option LTP — skipping trade", symbol)
                return

            # Set ATR-based SL/targets
            atr = indicators.atr
            if atr is None or atr <= 0:
                return
            option_atr = atr * 0.5
            best_signal.entry_price = option_ltp
            best_signal.stoploss = round(max(option_ltp - (1.5 * option_atr), option_ltp * 0.70), 2)
            best_signal.target1 = round(option_ltp + (2.0 * option_atr), 2)
            best_signal.target2 = round(option_ltp + (3.5 * option_atr), 2)

            # 12. AI validation
            decision = await self.ai_engine.evaluate(best_signal, snap, best_score)

            if not decision.trade_decision or decision.confidence_score < 65:
                logger.info(
                    "[%s] AI rejected (confidence=%.0f%%): %s",
                    symbol, decision.confidence_score, decision.reason,
                )
                await self.alert_manager.send_info(
                    f"SIGNAL REJECTED — {symbol} {best_signal.strategy.value}",
                    f"{symbol} {int(best_signal.strike_price)} {best_signal.option_type.value}\n"
                    f"Premium: ₹{option_ltp:.2f} | Score: {best_score:.0f} | AI: {decision.confidence_score:.0f}%\n"
                    f"Reason: {decision.reason}",
                )
                return

            # 13. Build NFO trading symbol
            nfo_symbol = instrument.build_option_symbol(
                expiry or "", best_signal.strike_price, best_signal.option_type.value
            )

            # 14. Position size
            num_lots = self.risk_manager.compute_position_size(
                decision.entry_price, decision.stoploss
            )

            # 15. Enter trade
            trade = self.paper_trader.enter_trade(best_signal, decision, nfo_symbol, num_lots=num_lots)
            trade.instrument = symbol
            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_signal_alert(best_signal, decision)

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

        for r in recs:
            if r.composite_score < min_score:
                continue
            sym = r.instrument
            if sym not in inst_best:
                inst_best[sym] = r.composite_score
                inst_strats[sym] = []
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
            logger.info("  %s → %s", inst.symbol, ", ".join(strat_names))

        return instruments

    async def _fetch_option_ltp_for(
        self,
        instrument: InstrumentConfig,
        signal: StrategySignal,
        expiry: Optional[str],
    ) -> Optional[float]:
        """Fetch the real LTP for a specific option contract from AngelOne."""
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
            ltp = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.get_ltp,
                    "NFO",
                    token_info.get("tradingsymbol", ""),
                    token_info.get("symboltoken", ""),
                ),
                timeout=15,
            )
        except asyncio.TimeoutError:
            logger.warning("LTP fetch timed out for %s", symbol)
            return None
        return ltp

    async def _update_options_chain_for(
        self,
        instrument: InstrumentConfig,
        spot_price: float,
        expiry: str,
    ) -> OptionsMetrics:
        """Fetch options chain and compute metrics for an instrument."""
        try:
            chain = await asyncio.wait_for(
                asyncio.to_thread(self.client.get_option_chain, expiry),
                timeout=60,
            )
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
                    ltp = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.get_ltp,
                            "NFO",
                            token_info.get("tradingsymbol", ""),
                            token_info.get("symboltoken", ""),
                        ),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    logger.warning("LTP fetch timed out for %s", trade.symbol)
                    ltp = None
                if ltp:
                    current_prices[trade.symbol] = ltp

        closed = self.paper_trader.check_exits(current_prices)
        for trade in closed:
            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_exit_alert(trade)

    async def _fetch_global_data(self) -> None:
        """Fetch and analyze global market indices."""
        try:
            self.global_indices = await fetch_global_indices()
            self.global_bias = compute_global_bias(self.global_indices)
            self._global_last_fetched = datetime.now(IST)
            logger.info("Global bias: %s (%d indices)", self.global_bias.value, len(self.global_indices))

            # Update shared state so API can serve index details
            from app.api.routes import get_state
            state = get_state()
            state["global_indices"] = [
                {"symbol": i.symbol, "change_pct": i.change_pct, "last_price": i.last_price}
                for i in self.global_indices
            ]
        except Exception:
            logger.exception("Error fetching global data")

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

    async def _close_all_trades(self) -> None:
        """Close all open trades at 15:20."""
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

        closed = self.paper_trader.close_all_open(current_prices)
        for trade in closed:
            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_exit_alert(trade)

        logger.info("All trades closed for EOD. Count: %d", len(closed))

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

    @staticmethod
    def _get_weekly_expiry_fallback() -> str:
        """Fallback: estimate weekly expiry if instrument master is unavailable.

        Tries Wednesday first (current NIFTY weekly expiry day), then Thursday.
        """
        today = datetime.now(IST)
        # NIFTY weekly expiry moved to Wednesday
        days_until_wed = (2 - today.weekday()) % 7
        if days_until_wed == 0 and today.hour >= 16:
            days_until_wed = 7
        expiry_date = today + timedelta(days=days_until_wed)
        return expiry_date.strftime("%d%b%Y").upper()
