"""Risk manager — enforces position limits and loss controls."""

from __future__ import annotations

import logging
from datetime import date, datetime

import pytz

from app.core.config import settings
from app.core.models import Trade, TradeStatus

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")


class RiskManager:
    """Enforce risk management rules.

    Rules:
        Max trades per day: 2
        Stoploss: 25–30% of option premium
        Daily loss limit: 2% of capital
        Consecutive loss limit: 3
    """

    def __init__(self) -> None:
        self.capital = settings.initial_capital
        self.max_trades = settings.max_trades_per_day
        self.max_daily_loss_pct = settings.max_daily_loss_pct
        self.consecutive_loss_limit = settings.consecutive_loss_limit

    def can_trade(self, today_trades: list[Trade]) -> bool:
        """Check if a new trade is allowed based on risk rules."""
        # Count today's trades
        today = datetime.now(_IST).date().isoformat()
        todays = [t for t in today_trades if t.date == today]

        # Max trades per day
        if len(todays) >= self.max_trades:
            logger.warning("Max daily trades reached (%d)", self.max_trades)
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
            last_n = closed[-self.consecutive_loss_limit :]
            if all((t.pnl or 0) < 0 for t in last_n):
                logger.warning("Consecutive loss limit reached (%d)", self.consecutive_loss_limit)
                return False

        return True

    def compute_stoploss(self, entry_price: float, sl_pct: float = 0.27) -> float:
        """Compute stoploss at given percentage of premium.

        Default: ~27% (midpoint of 25-30% range).
        """
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
