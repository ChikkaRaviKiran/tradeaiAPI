"""Smart Exit Engine — V2 context-aware exit management.

Replaces the fixed SL/target exit logic with dynamic exits that adapt to:
- Day type (trend/range/volatile)
- Market regime and momentum
- Time elapsed since entry
- Thesis validation (breakout level integrity)

Exit hierarchy (evaluated in order):
1. Hard stoploss (never violated)
2. Thesis-break exit (spot reverses through breakout level)
3. Time-based exit (max_hold_minutes exceeded)
4. Breakeven + trailing stoploss
5. Partial / target hits (T1, T2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pytz

from app.core.config import settings
from app.core.models import (
    DayType,
    MarketSnapshot,
    OptionType,
    Trade,
    TradeStatus,
)

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")


@dataclass
class ExitResult:
    """Result of evaluating a trade for exit."""

    should_exit: bool = False
    exit_type: str = ""           # stoploss / thesis_break / time_exit / trailing / target1 / target2 / eod
    exit_price: float = 0.0
    new_stoploss: Optional[float] = None  # Updated SL (if trailing, but not exiting)
    reason: str = ""


@dataclass
class ExitConfig:
    """Per-trade exit parameters — populated at entry time."""

    stoploss_pct: float = 20.0          # Max loss %
    quick_target_pct: float = 25.0      # T1 %
    breakeven_trigger_pct: float = 12.0 # Move SL to entry after this % profit (was 8% — too early)
    trail_activation_pct: float = 18.0  # Start trailing only after this % profit (new)
    trail_factor: float = 0.40          # Trail SL at this fraction of profit (was 0.50 — too tight)
    max_hold_minutes: int = 120         # LOCKED v1.0: Time-based exit limit
    thesis_break_buffer: float = 0.0015 # 0.15% buffer for breakout fail
    use_close_based_sl: bool = True     # Close-based SL — only exit on candle close (was False)
    catastrophic_atr_mult: float = 3.0  # Immediate SL if drop > this × ATR
    grace_period_minutes: int = 3       # No trailing SL adjustment in first N minutes (new)
    sl_confirm_candles: int = 3         # FIX 4: Require 3 consecutive candle closes below SL (was 2)


class SmartExitEngine:
    """Evaluate V2 trades for exit using context-aware rules.

    This engine does NOT close trades directly — it returns ExitResult
    objects that the orchestrator acts on. This keeps concerns separated.
    """

    # Track consecutive candle closes below SL per trade
    _sl_breach_counts: dict[str, int] = {}

    def evaluate(
        self,
        trade: Trade,
        current_ltp: float,
        snap: MarketSnapshot,
        day_type: DayType,
        spot_price: Optional[float] = None,
        candle_closed: bool = False,
        option_atr: float = 0.0,
    ) -> ExitResult:
        """Evaluate a single trade for exit conditions.

        Args:
            trade: The open V2 trade to check.
            current_ltp: Current option LTP.
            snap: Latest market snapshot with indicators.
            day_type: Today's classification.
            spot_price: Current underlying spot price (for thesis-break check).

        Returns:
            ExitResult with exit decision and updated SL if applicable.
        """
        cfg = self._build_config(day_type, trade)

        # 1. Hard stoploss — non-negotiable
        #    Close-based SL: only trigger if candle_closed (1-min close below SL)
        #    Exception: catastrophic drop (> 3× ATR) exits immediately
        result = self._check_hard_stoploss(trade, current_ltp, cfg, candle_closed, option_atr)
        if result.should_exit:
            return result

        # 2. Thesis-break — spot reversed through breakout level
        if spot_price is not None:
            result = self._check_thesis_break(trade, spot_price, cfg)
            if result.should_exit:
                result.exit_price = current_ltp
                return result

        # 3. Time-based exit
        result = self._check_time_exit(trade, current_ltp, cfg)
        if result.should_exit:
            return result

        # 4. Momentum fade — RSI divergence from trade direction
        result = self._check_momentum_fade(trade, current_ltp, snap, cfg)
        if result.should_exit:
            return result

        # 5. Trailing stoploss + breakeven management
        result = self._manage_trailing(trade, current_ltp, cfg)
        if result.should_exit:
            return result
        # Even if not exiting, propagate SL update
        if result.new_stoploss is not None:
            return result

        # 6. Target hits
        result = self._check_targets(trade, current_ltp)
        if result.should_exit:
            return result

        return ExitResult()  # No exit

    # ── Exit checks ──────────────────────────────────────────────────────

    def _check_hard_stoploss(self, trade: Trade, ltp: float, cfg: ExitConfig, candle_closed: bool = False, option_atr: float = 0.0) -> ExitResult:
        # ── Grace period: block ALL SL exits except catastrophic ────────
        # During the first N minutes, ORB/TREND trades face violent washouts
        # that recover. Only exit if drop is truly catastrophic (>3×ATR).
        # FIX 2: SL Delay Activation — grace ends when EITHER:
        #   a) grace_period_minutes elapsed, OR
        #   b) price moved +5% in your favor (SL activates to protect gains)
        in_grace = False
        if cfg.grace_period_minutes > 0 and trade.entry_datetime is not None:
            now = datetime.now(_IST)
            elapsed_min = (now - trade.entry_datetime).total_seconds() / 60
            time_grace_over = elapsed_min >= cfg.grace_period_minutes
            # Profit-based grace exit: if +5% in favor, SL should activate to protect
            profit_pct = (ltp - trade.entry_price) / trade.entry_price * 100 if trade.entry_price > 0 else 0
            profit_grace_over = profit_pct >= 5.0
            # Grace is active only if NEITHER condition is met
            in_grace = not time_grace_over and not profit_grace_over

        # Catastrophic exit: immediate if drop > 3× ATR (regardless of grace/candle close)
        if option_atr > 0:
            catastrophic_level = trade.entry_price - (cfg.catastrophic_atr_mult * option_atr)
            if ltp <= catastrophic_level:
                return ExitResult(
                    should_exit=True,
                    exit_type="stoploss",
                    exit_price=ltp,
                    reason=f"CATASTROPHIC SL: {ltp:.2f} <= {catastrophic_level:.2f} (>{cfg.catastrophic_atr_mult:.0f}×ATR below entry)",
                )

        # During grace period, skip all normal SL checks — let trade breathe
        if in_grace:
            return ExitResult()

        # Issue 4: Big opposite candle override — exit immediately if a single
        # candle drops > 1.5× ATR (real reversal, not noise). Skips multi-candle wait.
        if candle_closed and option_atr > 0 and trade.entry_price > 0:
            candle_drop = trade.entry_price - ltp  # positive = loss
            if ltp <= trade.stoploss and candle_drop > 1.5 * option_atr:
                SmartExitEngine._sl_breach_counts.pop(trade.trade_id, None)
                return ExitResult(
                    should_exit=True,
                    exit_type="stoploss",
                    exit_price=ltp,
                    reason=f"BIG CANDLE SL: drop {candle_drop:.2f} > 1.5×ATR ({1.5*option_atr:.2f}) — immediate exit",
                )

        # Close-based SL with 3-candle confirmation:
        # First candle close below SL = warning. Third consecutive = exit.
        # This survives single-candle washouts/stop-hunts.
        if cfg.use_close_based_sl:
            tid = trade.trade_id
            if candle_closed and ltp <= trade.stoploss:
                breach_count = SmartExitEngine._sl_breach_counts.get(tid, 0) + 1
                SmartExitEngine._sl_breach_counts[tid] = breach_count
                if breach_count >= cfg.sl_confirm_candles:
                    SmartExitEngine._sl_breach_counts.pop(tid, None)
                    return ExitResult(
                        should_exit=True,
                        exit_type="stoploss",
                        exit_price=ltp,
                        reason=f"Close-based SL confirmed ({breach_count} candles below SL={trade.stoploss:.2f})",
                    )
                else:
                    logger.info(
                        "SL BREACH %d/%d: %s LTP=%.2f < SL=%.2f — waiting for confirmation",
                        breach_count, cfg.sl_confirm_candles, trade.symbol, ltp, trade.stoploss,
                    )
                    return ExitResult()
            elif candle_closed and ltp > trade.stoploss:
                # Price recovered above SL — reset breach counter
                if tid in SmartExitEngine._sl_breach_counts:
                    logger.info(
                        "SL BREACH RESET: %s recovered above SL (LTP=%.2f > SL=%.2f)",
                        trade.symbol, ltp, trade.stoploss,
                    )
                    SmartExitEngine._sl_breach_counts.pop(tid, None)
            return ExitResult()

        # Legacy tick-based SL
        if ltp <= trade.stoploss:
            return ExitResult(
                should_exit=True,
                exit_type="stoploss",
                exit_price=ltp,
                reason=f"Hard SL hit at {ltp:.2f} (SL={trade.stoploss:.2f})",
            )
        return ExitResult()

    def _check_thesis_break(
        self, trade: Trade, spot: float, cfg: ExitConfig
    ) -> ExitResult:
        if trade.breakout_level is None:
            return ExitResult()

        buffer = trade.breakout_level * cfg.thesis_break_buffer
        is_call = trade.option_type == OptionType.CALL

        failed = (
            (is_call and spot < trade.breakout_level - buffer)
            or (not is_call and spot > trade.breakout_level + buffer)
        )
        if failed:
            return ExitResult(
                should_exit=True,
                exit_type="thesis_break",
                reason=f"Breakout failed: spot={spot:.2f}, level={trade.breakout_level:.2f}",
            )
        return ExitResult()

    def _check_time_exit(
        self, trade: Trade, ltp: float, cfg: ExitConfig
    ) -> ExitResult:
        max_mins = trade.max_hold_minutes or cfg.max_hold_minutes
        if max_mins <= 0 or trade.entry_datetime is None:
            return ExitResult()

        now = datetime.now(_IST)
        elapsed = (now - trade.entry_datetime).total_seconds() / 60

        if elapsed >= max_mins:
            return ExitResult(
                should_exit=True,
                exit_type="time_exit",
                exit_price=ltp,
                reason=f"Max hold {int(elapsed)}m >= {max_mins}m",
            )
        return ExitResult()

    def _check_momentum_fade(
        self,
        trade: Trade,
        ltp: float,
        snap: MarketSnapshot,
        cfg: ExitConfig,
    ) -> ExitResult:
        """Exit if RSI diverges strongly against trade direction while in profit.

        Only triggers when the trade is already in profit (avoids premature exits
        on losing trades) and RSI shows exhaustion.
        """
        rsi = snap.indicators.rsi
        if rsi is None:
            return ExitResult()

        profit_pct = ((ltp - trade.entry_price) / trade.entry_price * 100
                      if trade.entry_price > 0 else 0)

        # Only consider momentum fade when the trade has some profit
        if profit_pct < cfg.breakeven_trigger_pct * 0.5:
            return ExitResult()

        is_call = trade.option_type == OptionType.CALL

        # CALL trade but RSI excessively overbought (>78) → momentum may fade
        # PUT trade but RSI excessively oversold (<22) → momentum may fade
        fade = (is_call and rsi > 78) or (not is_call and rsi < 22)
        if fade:
            return ExitResult(
                should_exit=True,
                exit_type="momentum_fade",
                exit_price=ltp,
                reason=f"RSI={rsi:.1f} extreme while in profit ({profit_pct:.1f}%)",
            )
        return ExitResult()

    def _manage_trailing(
        self, trade: Trade, ltp: float, cfg: ExitConfig
    ) -> ExitResult:
        """Breakeven move + trailing stoploss with grace period.

        Phases:
        1. Grace period (first N minutes): No SL movement at all — let trade breathe
        2. Breakeven move: Once profit >= breakeven_trigger_pct, move SL to entry
        3. Trailing: Once profit >= trail_activation_pct, trail at configured factor
        """
        if trade.entry_price <= 0:
            return ExitResult()

        profit_pct = (ltp - trade.entry_price) / trade.entry_price * 100

        # Grace period — no trailing in first N minutes after entry
        if cfg.grace_period_minutes > 0 and trade.entry_datetime is not None:
            now = datetime.now(_IST)
            elapsed_min = (now - trade.entry_datetime).total_seconds() / 60
            if elapsed_min < cfg.grace_period_minutes:
                return ExitResult()

        # Phase 1: Move SL to breakeven at trigger percentage
        if profit_pct >= cfg.breakeven_trigger_pct:
            be_sl = max(trade.entry_price, trade.stoploss)
            if be_sl > trade.stoploss:
                return ExitResult(
                    new_stoploss=round(be_sl, 2),
                    reason=f"Breakeven: profit {profit_pct:.1f}% >= {cfg.breakeven_trigger_pct}%",
                )

        # Phase 2: Trail at configured factor only after trail_activation_pct
        if (
            trade.stoploss >= trade.entry_price
            and profit_pct >= cfg.trail_activation_pct
        ):
            trail_sl = trade.entry_price + (ltp - trade.entry_price) * cfg.trail_factor
            if trail_sl > trade.stoploss:
                return ExitResult(
                    new_stoploss=round(trail_sl, 2),
                    reason=f"Trail SL updated to {trail_sl:.2f} ({cfg.trail_factor*100:.0f}% of profit, activated at {cfg.trail_activation_pct}%)",
                )

        return ExitResult()

    def _check_targets(self, trade: Trade, ltp: float) -> ExitResult:
        if ltp >= trade.target2:
            return ExitResult(
                should_exit=True,
                exit_type="target2",
                exit_price=ltp,
                reason=f"T2 hit: {ltp:.2f} >= {trade.target2:.2f}",
            )
        # FIX 5: Partial exit at T1 — exit 50%, keep riding the rest
        if ltp >= trade.target1 and not trade.partial_exit_done:
            return ExitResult(
                should_exit=True,
                exit_type="partial_target1",
                exit_price=ltp,
                reason=f"PARTIAL T1: {ltp:.2f} >= {trade.target1:.2f} — exit 50%, trail rest",
            )
        # Issue 2: Time-based partial exit
        # If held 20+ min and profit > 10% but hasn't reached T1 → partial exit
        if not trade.partial_exit_done and trade.entry_datetime is not None and trade.entry_price > 0:
            now = datetime.now(_IST)
            elapsed_min = (now - trade.entry_datetime).total_seconds() / 60
            profit_pct = (ltp - trade.entry_price) / trade.entry_price * 100
            if elapsed_min >= 20 and profit_pct >= 10.0:
                return ExitResult(
                    should_exit=True,
                    exit_type="partial_target1",
                    exit_price=ltp,
                    reason=f"TIME PARTIAL: {elapsed_min:.0f}min held, profit {profit_pct:.1f}% >= 10% — exit 50%, trail rest",
                )
        return ExitResult()

    # ── Config builder ───────────────────────────────────────────────────

    def _build_config(self, day_type: DayType, trade: Optional[Trade] = None) -> ExitConfig:
        """Build exit config for RB-only mode with uniform 20% SL framework.

        Aligned with backtest: 20% SL (1R), T1 at +1R → SL→breakeven, T2 at +2R → lock 1R.
        Day-type adjustments removed — uniform exits proven across all regimes.
        """
        base = ExitConfig(
            stoploss_pct=20.0,                # Uniform 20% SL = 1R
            quick_target_pct=20.0,            # T1 at +1R (same as SL distance)
            breakeven_trigger_pct=20.0,       # Move SL to breakeven when +1R hit
            trail_activation_pct=30.0,        # Start trailing after +1.5R
            trail_factor=0.50,                # Lock 50% of unrealized profit
            max_hold_minutes=settings.v2_max_hold_minutes,
            use_close_based_sl=True,
            grace_period_minutes=5,
            sl_confirm_candles=3,
        )

        return base
