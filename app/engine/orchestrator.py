"""Main orchestrator — runs the continuous trading loop during market hours.

Schedule:
    08:45  Load data, authenticate
    09:00  Fetch global market context
    09:15  Start monitoring
    09:15–15:30  Continuous analysis (1-min loop)
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
from app.core.holidays import is_market_holiday, next_trading_date
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
from app.engine.signal_scorer import SignalScorer, MIN_SCORE
from app.strategies.base import BaseStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.orb import ORBStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.trading.history_logger import HistoryLogger
from app.trading.paper_trader import PaperTradingEngine
from app.trading.risk_manager import RiskManager
from app.trading.trade_logger import TradeLogger

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
    """Central trading system controller."""

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

        # Strategies
        self.strategies: list[BaseStrategy] = [
            ORBStrategy(),
            VWAPReclaimStrategy(),
            TrendPullbackStrategy(),
            LiquiditySweepStrategy(),
            RangeBreakoutStrategy(),
        ]

        # State
        self.running = False
        self._cycle_count = 0
        self._global_last_fetched: Optional[datetime] = None
        self.global_bias = GlobalBias.UNAVAILABLE
        self.global_indices: list = []  # Store individual index data
        self.snapshot: Optional[MarketSnapshot] = None
        self.options_metrics = OptionsMetrics()

        # Heartbeat tracking
        self._last_heartbeat_cycle = 0
        self._consecutive_no_signal = 0

        # Expiry format for current week (needs to be set dynamically)
        self._expiry: Optional[str] = None

    async def start(self) -> None:
        """Main entry point — runs forever, restarting each trading day."""
        logger.info("=" * 60)
        logger.info("TradeAI Orchestrator started (continuous mode)")
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
        self.snapshot = None
        self.options_metrics = OptionsMetrics()
        self._expiry = None
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

        # Authenticate with AngelOne
        if not self.client.authenticate():
            logger.error("Failed to authenticate. Aborting today.")
            await self.alert_manager.send_info("AUTH FAILED", "AngelOne authentication failed. Check credentials.")
            return

        # Determine current weekly expiry from live instrument data
        self._expiry = self.client.get_nearest_weekly_expiry()
        if not self._expiry:
            self._expiry = self._get_weekly_expiry_fallback()
        logger.info("Weekly expiry: %s", self._expiry)

        await self.alert_manager.send_info(
            "SYSTEM STARTED",
            f"Paper trading: {settings.paper_trading}\nCapital: ₹{settings.initial_capital:,.0f}\nExpiry: {self._expiry}",
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
                    await self._fetch_global_data()
                elif (datetime.now(IST) - self._global_last_fetched).total_seconds() > 900:
                    await self._fetch_global_data()

                await self._run_analysis_cycle()
                await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                continue

            # Pre-close: close open trades
            if PRE_CLOSE <= current_time < MARKET_CLOSE:
                await self._close_all_trades()
                await asyncio.sleep(60)
                continue

            # Post-market: generate daily report
            if REPORT_TIME <= current_time <= dtime(16, 0):
                await self._generate_daily_report()
                self.running = False
                logger.info("Trading day complete. Will restart next trading day.")
                return

            # Outside market hours — wait
            logger.debug("Outside market hours (%s). Waiting...", current_time)
            await asyncio.sleep(60)

    # ── Core Analysis Cycle ──────────────────────────────────────────────

    async def _run_analysis_cycle(self) -> None:
        """Single iteration of the 1-minute analysis loop."""
        self._cycle_count += 1
        cycle = self._cycle_count
        try:
            # 1. Fetch NIFTY candle data
            now = datetime.now(IST)
            from_date = now.strftime("%Y-%m-%d 09:15")
            to_date = now.strftime("%Y-%m-%d %H:%M")

            logger.info("── Cycle %d ── %s ──", cycle, now.strftime("%H:%M:%S"))

            candles = self.client.get_nifty_candles(
                interval="ONE_MINUTE", from_date=from_date, to_date=to_date
            )
            df = self.client.candles_to_dataframe(candles)

            if df.empty:
                logger.warning("[Cycle %d] No candle data received from AngelOne", cycle)
                return

            # 2. Validate data (NIFTY is an index — volume is always 0)
            df = self.validator.validate_candles(df, is_index=True)
            if not self.validator.is_valid_for_trading(df):
                logger.warning("[Cycle %d] Data validation failed — skipping cycle", cycle)
                return

            # 2b. Fetch NIFTY Futures candles and merge volume into spot data
            #     This gives us real volume for VWAP and volume analysis
            fut_candles = self.client.get_nifty_futures_candles(
                interval="ONE_MINUTE", from_date=from_date, to_date=to_date
            )
            if fut_candles:
                fut_df = self.client.candles_to_dataframe(fut_candles)
                df = self.feature_engine.merge_futures_volume(df, fut_df)
            else:
                logger.warning("[Cycle %d] No NIFTY Futures data — no volume/VWAP", cycle)

            # 3. Feature engineering
            df = self.feature_engine.compute_indicators(df)
            indicators = self.feature_engine.get_latest_indicators(df)
            spot_price = df.iloc[-1]["close"] if not df.empty else 0

            logger.info(
                "[Cycle %d] NIFTY=%.2f | RSI=%s | ADX=%s | MACD=%s | EMA9=%s | EMA20=%s | Vol=%d",
                cycle, spot_price,
                f"{indicators.rsi:.1f}" if indicators.rsi is not None else "N/A",
                f"{indicators.adx:.1f}" if indicators.adx is not None else "N/A",
                f"{indicators.macd:.2f}" if indicators.macd is not None else "N/A",
                f"{indicators.ema9:.1f}" if indicators.ema9 is not None else "N/A",
                f"{indicators.ema20:.1f}" if indicators.ema20 is not None else "N/A",
                df.iloc[-1].get("volume", 0) if not df.empty else 0,
            )

            # 4. Fetch options chain (less frequently — every 5 minutes)
            if now.minute % 5 == 0:
                await self._update_options_chain(spot_price)

            # 5. Market regime detection
            regime = self.regime_detector.detect(df)

            # 6. Build market snapshot
            self.snapshot = MarketSnapshot(
                nifty_price=spot_price,
                vwap=indicators.vwap,
                regime=regime,
                global_bias=self.global_bias,
                indicators=indicators,
                options_metrics=self.options_metrics,
                timestamp=now,
            )

            # Update shared state for API
            from app.api.routes import get_state
            state = get_state()
            state["snapshot"] = self.snapshot
            state["open_trades"] = self.paper_trader.open_trades

            # Persist snapshot to database for historical review
            await self.history_logger.save_snapshot(self.snapshot)

            # 7. Check exits on open trades
            await self._check_trade_exits(spot_price)

            # 8. Check risk limits
            if not self.risk_manager.can_trade(self.paper_trader.all_today_trades):
                logger.info("[Cycle %d] Risk limits reached — no new trades allowed", cycle)
                return

            # 9. Run strategies
            signals: list[StrategySignal] = []
            for strategy in self.strategies:
                signal = strategy.evaluate(df, self.options_metrics, spot_price)
                if signal:
                    signals.append(signal)
                    logger.info(
                        "[Cycle %d] Signal from %s: %s %s @ %.0f",
                        cycle, signal.strategy.value, signal.option_type.value,
                        signal.direction.value if hasattr(signal, 'direction') else 'N/A',
                        signal.strike_price,
                    )

            if not signals:
                self._consecutive_no_signal += 1
                logger.info("[Cycle %d] No strategy signals this cycle (regime=%s)", cycle, regime.value)

                # Send heartbeat alert every 15 cycles (~15 min) so user knows system is active
                if cycle - self._last_heartbeat_cycle >= 15:
                    self._last_heartbeat_cycle = cycle
                    await self.alert_manager.send_info(
                        f"SYSTEM HEARTBEAT — Cycle #{cycle}",
                        f"NIFTY: {spot_price:,.2f} | Regime: {regime.value}\n"
                        f"RSI: {indicators.rsi:.1f if indicators.rsi else 'N/A'} | "
                        f"ADX: {indicators.adx:.1f if indicators.adx else 'N/A'}\n"
                        f"VWAP: {indicators.vwap:,.2f if indicators.vwap else 'N/A'} | "
                        f"Global: {self.global_bias.value}\n"
                        f"No signals for {self._consecutive_no_signal} consecutive cycles.\n"
                        f"Strategies are monitoring — waiting for conditions to align.",
                    )
                return
            else:
                self._consecutive_no_signal = 0

            # 10. Score and filter signals
            best_signal = None
            best_score = 0.0

            for signal in signals:
                score_result = self.signal_scorer.score(
                    signal, df, self.options_metrics, self.global_bias
                )
                logger.info(
                    "[Cycle %d] %s scored %.0f/100 (min=%d)",
                    cycle, signal.strategy.value, score_result.total, MIN_SCORE,
                )
                if score_result.total >= MIN_SCORE and score_result.total > best_score:
                    best_signal = signal
                    best_score = score_result.total
                    signal.score = score_result.total

            if best_signal is None:
                logger.info("[Cycle %d] All signals below minimum score threshold (%d)", cycle, MIN_SCORE)
                await self.alert_manager.send_info(
                    "LOW SCORE SIGNALS",
                    f"{len(signals)} signal(s) detected but none scored >= {MIN_SCORE}\n"
                    f"Strategies: {', '.join(s.strategy.value for s in signals)}",
                )
                return

            logger.info(
                "Best signal: %s %s (score=%.0f)",
                best_signal.strategy.value,
                best_signal.option_type.value,
                best_score,
            )

            # 11. Fetch real option premium from AngelOne
            option_ltp = self._fetch_option_ltp(best_signal)
            if option_ltp is None or option_ltp <= 0:
                logger.warning(
                    "Could not fetch option LTP for %s %s — skipping trade",
                    int(best_signal.strike_price),
                    best_signal.option_type.value,
                )
                return

            # Set real option premium on the signal
            # SL and targets are ATR-based, not fixed percentages
            atr = indicators.atr
            if atr is None or atr <= 0:
                logger.warning(
                    "ATR unavailable — cannot compute market-based SL/targets. Skipping trade."
                )
                await self.alert_manager.send_info(
                    f"TRADE SKIPPED — {best_signal.strategy.value}",
                    f"ATR data unavailable — cannot set SL/targets from real market data.\n"
                    f"Signal: NIFTY {int(best_signal.strike_price)} {best_signal.option_type.value}\n"
                    f"Score: {best_score:.0f} | Premium: ₹{option_ltp:.2f}",
                )
                return

            # ATR represents the expected NIFTY move per candle
            # Option premium moves ~50-70% of NIFTY's ATR for ATM options (delta ~0.5)
            option_atr = atr * 0.5
            best_signal.entry_price = option_ltp
            best_signal.stoploss = round(max(option_ltp - (1.5 * option_atr), option_ltp * 0.70), 2)
            best_signal.target1 = round(option_ltp + (2.0 * option_atr), 2)
            best_signal.target2 = round(option_ltp + (3.5 * option_atr), 2)
            logger.info(
                "Option premium: %.2f | SL=%.2f | T1=%.2f | T2=%.2f",
                option_ltp, best_signal.stoploss, best_signal.target1, best_signal.target2,
            )

            # 12. AI validation
            decision = await self.ai_engine.evaluate(
                best_signal, self.snapshot, best_score
            )

            if not decision.trade_decision or decision.confidence_score < 70:
                logger.info(
                    "AI rejected signal (confidence=%.0f%%): %s",
                    decision.confidence_score,
                    decision.reason,
                )
                await self.alert_manager.send_info(
                    f"SIGNAL REJECTED — {best_signal.strategy.value}",
                    f"NIFTY {int(best_signal.strike_price)} {best_signal.option_type.value}\n"
                    f"Premium: ₹{option_ltp:.2f} | Score: {best_score:.0f} | AI Confidence: {decision.confidence_score:.0f}%\n"
                    f"Reason: {decision.reason}",
                )
                return

            # 13. Build NFO trading symbol for consistent price tracking
            nfo_symbol = self._build_nfo_symbol(best_signal)

            # 14. Enter trade
            trade = self.paper_trader.enter_trade(best_signal, decision, nfo_symbol)
            await self.trade_logger.log_trade(trade)
            await self.alert_manager.send_signal_alert(best_signal, decision)

        except Exception:
            logger.exception("Error in analysis cycle")

    # ── Support Methods ──────────────────────────────────────────────────

    def _fetch_option_ltp(self, signal: StrategySignal) -> Optional[float]:
        """Fetch the real LTP for a specific option contract from AngelOne."""
        token_info = self.client.get_nifty_option_tokens(
            self._expiry or "",
            signal.strike_price,
            signal.option_type.value,
        )
        if not token_info:
            logger.warning(
                "Token not found for NIFTY %s %s",
                int(signal.strike_price), signal.option_type.value,
            )
            return None

        ltp = self.client.get_ltp(
            "NFO",
            token_info.get("tradingsymbol", ""),
            token_info.get("symboltoken", ""),
        )
        return ltp

    def _build_nfo_symbol(self, signal: StrategySignal) -> str:
        """Build the NFO trading symbol for an option contract.

        e.g. NIFTY17MAR202622500CE
        """
        return f"NIFTY{self._expiry or ''}{int(signal.strike_price)}{signal.option_type.value}"

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
        """Fetch options chain and compute metrics."""
        try:
            if self._expiry:
                chain = self.client.get_option_chain(self._expiry)
                self.options_metrics = self.feature_engine.compute_options_metrics(
                    chain, spot_price
                )
                logger.info(
                    "Options: PCR=%s MaxPain=%s ATM_Vol=%d",
                    f"{self.options_metrics.pcr:.2f}" if self.options_metrics.pcr is not None else "N/A",
                    f"{self.options_metrics.max_pain:.0f}" if self.options_metrics.max_pain is not None else "N/A",
                    self.options_metrics.atm_option_volume,
                )
        except Exception:
            logger.exception("Error updating options chain")

    async def _check_trade_exits(self, spot_price: float) -> None:
        """Check open trades for stoploss/target hits."""
        if not self.paper_trader.open_trades:
            return

        # Build current price map
        current_prices: dict[str, float] = {}
        for trade in self.paper_trader.open_trades:
            # Fetch current LTP for each option
            token_info = self.client.get_nifty_option_tokens(
                self._expiry or "",
                trade.strike,
                trade.option_type.value,
            )
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
            token_info = self.client.get_nifty_option_tokens(
                self._expiry or "",
                trade.strike,
                trade.option_type.value,
            )
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
