"""Risk manager — enforces position limits and loss controls.

SRS Rules:
    Max risk per trade: 1% of capital
    Max daily loss: 3% of capital
    Max positions: 2 concurrent trades (1 per instrument)
    Max trades per day: 3
    Consecutive loss limit: 3
    Position sizing: Automatic based on risk per trade
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pytz

from app.core.config import settings
from app.core.models import Trade, TradeStatus

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")


class RiskManager:
    """Enforce risk management rules per SRS specification.

    Rules:
        Max risk per trade: 1% of capital
        Max daily loss: 3% of capital
        Max trades per day: 3
        Max concurrent positions: 2 (1 per instrument)
        Consecutive loss limit: 3
    """

    def __init__(
        self,
        max_trades: int = 0,
        max_concurrent: int = 0,
        risk_pct: float = 0,
        consecutive_limit: int = 0,
    ) -> None:
        self.capital = settings.initial_capital
        self.max_trades = max_trades or settings.max_trades_per_day
        self.max_daily_loss_pct = settings.max_daily_loss_pct
        self.risk_per_trade_pct = risk_pct or settings.risk_per_trade_pct
        self.max_concurrent = max_concurrent or settings.max_concurrent_positions
        self.consecutive_loss_limit = consecutive_limit or settings.consecutive_loss_limit

    def can_trade(self, today_trades: list[Trade], open_count: int = 0, instrument: str = "", open_trades: list[Trade] | None = None) -> bool:
        """Check if a new trade is allowed based on risk rules.

        Args:
            today_trades: All trades taken today (open + closed).
            open_count: Total open positions across all instruments.
            instrument: If provided, also enforce per-instrument concurrent limit.
            open_trades: If provided, list of currently open trades (for per-instrument check).
        """
        today = datetime.now(_IST).date().isoformat()
        todays = [t for t in today_trades if t.date == today]

        # Max trades per day
        if len(todays) >= self.max_trades:
            logger.warning("Max daily trades reached (%d)", self.max_trades)
            return False

        # Max concurrent positions (global)
        if open_count >= self.max_concurrent:
            logger.warning("Max concurrent positions reached (%d)", self.max_concurrent)
            return False

        # Per-instrument concurrent limit
        if instrument and open_trades is not None:
            max_per_inst = settings.max_concurrent_per_instrument
            inst_open = sum(1 for t in open_trades if t.instrument == instrument)
            if inst_open >= max_per_inst:
                logger.warning("Max concurrent per-instrument reached for %s (%d)", instrument, max_per_inst)
                return False

        # Daily loss limit
        daily_pnl = sum(t.pnl or 0 for t in todays if t.status == TradeStatus.CLOSED)
        max_loss = self.capital * (self.max_daily_loss_pct / 100)
        if daily_pnl < 0 and abs(daily_pnl) >= max_loss:
            logger.warning("Daily loss limit reached: %.2f (max: %.2f)", daily_pnl, -max_loss)
            return False

        # Consecutive loss limit
        closed = [t for t in todays if t.status == TradeStatus.CLOSED]
        if len(closed) >= self.consecutive_loss_limit:
            last_n = closed[-self.consecutive_loss_limit:]
            if all((t.pnl or 0) < 0 for t in last_n):
                logger.warning("Consecutive loss limit reached (%d)", self.consecutive_loss_limit)
                return False

        return True

    def compute_position_size(self, entry_price: float, stoploss: float, lot_size: int = 0, allocated_capital: float = 0) -> int:
        """Compute position size based on risk per trade.

        If allocated_capital is provided, uses that instead of total capital.
        Uses instrument-specific lot_size if provided.

        Returns the number of lots (minimum 1).
        """
        capital = allocated_capital if allocated_capital > 0 else self.capital
        risk_amount = capital * (self.risk_per_trade_pct / 100)
        per_unit_risk = abs(entry_price - stoploss)
        if per_unit_risk <= 0:
            return 1

        ls = lot_size if lot_size > 0 else settings.nifty_lot_size
        max_lots = int(risk_amount / (per_unit_risk * ls))
        return max(1, max_lots)

    def compute_stoploss(self, entry_price: float, sl_pct: float = 0.27) -> float:
        """Compute stoploss at given percentage of premium."""
        return round(entry_price * (1 - sl_pct), 2)

    def compute_targets(
        self, entry_price: float
    ) -> tuple[float, float]:
        """Compute target prices: T1 = 50% profit, T2 = 100% profit."""
        t1 = round(entry_price * 1.5, 2)
        t2 = round(entry_price * 2.0, 2)
        return t1, t2

    def get_daily_pnl(self, today_trades: list[Trade]) -> float:
        """Sum PnL for today's closed trades."""
        today = datetime.now(_IST).date().isoformat()
        return sum(
            t.pnl or 0
            for t in today_trades
            if t.date == today and t.status == TradeStatus.CLOSED
        )

    def get_risk_status(self, today_trades: list[Trade], open_count: int = 0) -> dict:
        """Return current risk metrics for dashboard display."""
        today = datetime.now(_IST).date().isoformat()
        todays = [t for t in today_trades if t.date == today]
        daily_pnl = self.get_daily_pnl(today_trades)
        max_loss = self.capital * (self.max_daily_loss_pct / 100)

        return {
            "capital": self.capital,
            "daily_pnl": daily_pnl,
            "daily_loss_limit": -max_loss,
            "daily_loss_used_pct": round(abs(daily_pnl) / max_loss * 100, 1) if daily_pnl < 0 else 0,
            "trades_today": len(todays),
            "max_trades": self.max_trades,
            "open_positions": open_count,
            "max_concurrent": self.max_concurrent,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "can_trade": self.can_trade(today_trades, open_count),
        }
