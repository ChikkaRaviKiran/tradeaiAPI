"""Paper trading engine — simulates trade execution and management."""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Optional

from app.core.config import settings
from app.core.models import (
    AIDecision,
    OptionType,
    StrategySignal,
    Trade,
    TradeStatus,
)

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Simulate trade entry, exit, and PnL tracking."""

    def __init__(self) -> None:
        self.lot_size = settings.nifty_lot_size
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []

    @property
    def all_today_trades(self) -> list[Trade]:
        today = date.today().isoformat()
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
    ) -> Trade:
        """Open a new paper trade.

        Args:
            signal: The strategy signal that triggered the trade.
            decision: AI validation result with entry/SL/targets.
            nfo_symbol: NFO trading symbol (e.g. NIFTY17MAR202622500CE)
                        used for consistent price lookups during exit monitoring.
        """
        now = datetime.now()
        # Use NFO symbol for price tracking; display-friendly name in reason
        display_name = f"NIFTY {int(signal.strike_price)} {signal.option_type.value}"
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            symbol=nfo_symbol or display_name,
            strike=signal.strike_price,
            option_type=signal.option_type,
            strategy=signal.strategy,
            entry_price=decision.entry_price,
            stoploss=decision.stoploss,
            target1=decision.target1,
            target2=decision.target2,
            confidence=decision.confidence_score,
            status=TradeStatus.OPEN,
            lot_size=self.lot_size,
            reason=decision.reason,
        )
        self.open_trades.append(trade)
        logger.info(
            "TRADE ENTERED: %s | Entry=%.2f | SL=%.2f | T1=%.2f | T2=%.2f",
            trade.symbol,
            trade.entry_price,
            trade.stoploss,
            trade.target1,
            trade.target2,
        )
        return trade

    def check_exits(self, current_prices: dict[str, float]) -> list[Trade]:
        """Check all open trades against current prices for exit conditions.

        Args:
            current_prices: Symbol → current LTP mapping.

        Returns:
            List of trades that were just closed.
        """
        closed_now: list[Trade] = []

        for trade in list(self.open_trades):
            symbol_key = trade.symbol
            current_ltp = current_prices.get(symbol_key)
            if current_ltp is None:
                continue

            exit_reason = None

            # Check stoploss
            if current_ltp <= trade.stoploss:
                exit_reason = "stoploss"
                trade.exit_price = trade.stoploss

            # Check target 2 first (full exit)
            elif current_ltp >= trade.target2:
                exit_reason = "target2"
                trade.exit_price = trade.target2

            # Check target 1 (partial — for simplicity, full exit at T1)
            elif current_ltp >= trade.target1:
                exit_reason = "target1"
                trade.exit_price = trade.target1

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

        trade.pnl = round(
            (trade.exit_price - trade.entry_price) * trade.lot_size, 2
        )
        trade.status = TradeStatus.CLOSED

        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        logger.info(
            "TRADE CLOSED [%s]: %s | Entry=%.2f | Exit=%.2f | PnL=%.2f",
            reason,
            trade.symbol,
            trade.entry_price,
            trade.exit_price,
            trade.pnl,
        )

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Calculate total unrealized PnL for open trades."""
        total = 0.0
        for trade in self.open_trades:
            price = current_prices.get(trade.symbol, trade.entry_price)
            total += (price - trade.entry_price) * trade.lot_size
        return round(total, 2)
