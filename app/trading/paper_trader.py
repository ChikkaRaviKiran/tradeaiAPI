"""Paper trading engine — simulates trade execution and management."""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Optional

import pytz

from app.core.config import settings
from app.core.models import (
    AIDecision,
    OptionType,
    StrategySignal,
    Trade,
    TradeStatus,
)

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")


class PaperTradingEngine:
    """Simulate trade entry, exit, and PnL tracking."""

    def __init__(self) -> None:
        self.lot_size = settings.nifty_lot_size
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []

    @property
    def all_today_trades(self) -> list[Trade]:
        today = datetime.now(_IST).date().isoformat()
        return [
            t
            for t in self.open_trades + self.closed_trades
            if t.date == today
        ]

    def enter_trade(
        self,
        signal: StrategySignal,
        decision: AIDecision,
        nfo_symbol: str = "",
        num_lots: int = 1,
        instrument_lot_size: int = 0,
    ) -> Trade:
        """Open a new paper trade.

        Args:
            signal: The strategy signal that triggered the trade.
            decision: AI validation result with entry/SL/targets.
            nfo_symbol: NFO trading symbol (e.g. NIFTY17MAR202622500CE)
                        used for consistent price lookups during exit monitoring.
            num_lots: Number of lots to trade (from risk-based position sizing).
            instrument_lot_size: Lot size from instrument config (0 = use default).
        """
        # Prevent duplicate trades on the same option symbol
        symbol_to_use = nfo_symbol or f"NIFTY {int(signal.strike_price)} {signal.option_type.value}"
        for t in self.open_trades:
            if t.symbol == symbol_to_use:
                logger.warning(
                    "DUPLICATE BLOCKED: %s already has an open trade — skipping",
                    symbol_to_use,
                )
                return t  # Return existing trade instead of creating a new one

        now = datetime.now(_IST)
        # Use instrument lot size if provided, otherwise fall back to default
        lot_size = instrument_lot_size if instrument_lot_size > 0 else self.lot_size
        display_name = f"NIFTY {int(signal.strike_price)} {signal.option_type.value}"
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            symbol=symbol_to_use,
            strike=signal.strike_price,
            option_type=signal.option_type,
            strategy=signal.strategy,
            entry_price=decision.entry_price,
            stoploss=decision.stoploss,
            target1=decision.target1,
            target2=decision.target2,
            confidence=decision.confidence_score,
            status=TradeStatus.OPEN,
            lot_size=lot_size * num_lots,
            reason=decision.reason,
        )
        self.open_trades.append(trade)
        logger.info(
            "TRADE ENTERED: %s | Entry=%.2f @%s | SL=%.2f | T1=%.2f | T2=%.2f | LotSize=%d",
            trade.symbol,
            trade.entry_price,
            trade.time,
            trade.stoploss,
            trade.target1,
            trade.target2,
            trade.lot_size,
        )
        return trade

    def check_exits(self, current_prices: dict[str, float]) -> list[Trade]:
        """Check all open trades against current prices for exit conditions.

        Trailing SL phases:
        1. Grace period (first 3 min): no SL movement
        2. Once price moves 50% toward T1: SL moves to breakeven
        3. Once price moves 75% toward T1: SL trails at 35% of profit

        Args:
            current_prices: Symbol → current LTP mapping.

        Returns:
            List of trades that were just closed.
        """
        closed_now: list[Trade] = []
        now = datetime.now(_IST)

        for trade in list(self.open_trades):
            symbol_key = trade.symbol
            current_ltp = current_prices.get(symbol_key)
            if current_ltp is None:
                continue

            # Grace period — no trailing in first 3 minutes
            if trade.entry_datetime is not None:
                elapsed_min = (now - trade.entry_datetime).total_seconds() / 60
                if elapsed_min < 3:
                    # Only check hard SL and targets during grace period
                    exit_reason = None
                    if current_ltp <= trade.stoploss:
                        exit_reason = "stoploss"
                        trade.exit_price = current_ltp
                    elif current_ltp >= trade.target2:
                        exit_reason = "target2"
                        trade.exit_price = current_ltp
                    elif current_ltp >= trade.target1:
                        exit_reason = "target1"
                        trade.exit_price = current_ltp
                    if exit_reason:
                        self._close_trade(trade, exit_reason)
                        closed_now.append(trade)
                    continue

            # Trailing stoploss logic
            move_to_t1 = trade.target1 - trade.entry_price
            current_profit = current_ltp - trade.entry_price

            if move_to_t1 > 0 and current_profit > 0:
                # Phase 1: Once price has moved 50% toward T1, move SL to breakeven
                if current_profit >= move_to_t1 * 0.5:
                    new_sl = trade.entry_price  # Breakeven
                    # Phase 2: Once 75% toward T1, trail at 35% of profit (was 40%)
                    if current_profit >= move_to_t1 * 0.75:
                        new_sl = trade.entry_price + (current_profit * 0.35)
                    new_sl = max(new_sl, trade.stoploss)  # Never lower the SL
                    if new_sl > trade.stoploss:
                        logger.debug(
                            "TRAILING SL: %s | Old=%.2f | New=%.2f | LTP=%.2f",
                            trade.symbol, trade.stoploss, new_sl, current_ltp,
                        )
                        trade.stoploss = round(new_sl, 2)

            exit_reason = None

            # Check stoploss — distinguish trailing SL from raw SL
            if current_ltp <= trade.stoploss:
                if trade.stoploss > trade.entry_price:
                    exit_reason = "trailing_stoploss"
                else:
                    exit_reason = "stoploss"
                trade.exit_price = current_ltp

            # Check target 2 first (full exit)
            elif current_ltp >= trade.target2:
                exit_reason = "target2"
                trade.exit_price = current_ltp

            # Check target 1 (partial — for simplicity, full exit at T1)
            elif current_ltp >= trade.target1:
                exit_reason = "target1"
                trade.exit_price = current_ltp

            if exit_reason:
                self._close_trade(trade, exit_reason)
                closed_now.append(trade)

        return closed_now

    def close_all_open(self, current_prices: dict[str, float]) -> list[Trade]:
        """Close all open trades (end-of-day at 15:20)."""
        closed: list[Trade] = []
        for trade in list(self.open_trades):
            price = current_prices.get(trade.symbol, trade.entry_price)
            trade.exit_price = price
            self._close_trade(trade, "eod_close")
            closed.append(trade)
        return closed

    def _close_trade(self, trade: Trade, reason: str) -> None:
        """Finalize a trade closure."""
        if trade.exit_price is None:
            trade.exit_price = trade.entry_price

        trade.exit_time = datetime.now(_IST).strftime("%H:%M:%S")
        trade.pnl = round(
            (trade.exit_price - trade.entry_price) * trade.lot_size, 2
        )
        trade.status = TradeStatus.CLOSED
        trade.reason = reason  # Store exit reason (overrides AI entry reason)

        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        logger.info(
            "TRADE CLOSED [%s]: %s | Entry=%.2f @%s | Exit=%.2f @%s | PnL=%.2f | Lots=%d",
            reason,
            trade.symbol,
            trade.entry_price,
            trade.time,
            trade.exit_price,
            trade.exit_time,
            trade.pnl,
            trade.lot_size,
        )

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Calculate total unrealized PnL for open trades."""
        total = 0.0
        for trade in self.open_trades:
            price = current_prices.get(trade.symbol, trade.entry_price)
            total += (price - trade.entry_price) * trade.lot_size
        return round(total, 2)
