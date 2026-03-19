"""Alert system — stores alerts for UI display, optionally sends via Telegram/Email."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from datetime import datetime
from typing import Optional

import pytz

from app.core.config import settings
from app.core.models import AIDecision, AlertItem, StrategySignal, Trade

logger = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

MAX_ALERTS = 200  # keep last N alerts in memory


class AlertStore:
    """In-memory alert store served to the dashboard UI."""

    def __init__(self, maxlen: int = MAX_ALERTS) -> None:
        self._alerts: deque[AlertItem] = deque(maxlen=maxlen)

    def add(self, alert: AlertItem) -> None:
        self._alerts.appendleft(alert)

    def get_all(self) -> list[AlertItem]:
        return list(self._alerts)

    def get_recent(self, n: int = 50) -> list[AlertItem]:
        return list(self._alerts)[:n]

    def clear(self) -> None:
        self._alerts.clear()


# Singleton store — shared across the app
alert_store = AlertStore()


def _format_signal_message(signal: StrategySignal, decision: AIDecision) -> str:
    inst = getattr(signal, 'instrument', 'NIFTY')
    return (
        f"Strategy: {signal.strategy.value}\n"
        f"Instrument: {inst} {int(signal.strike_price)} {signal.option_type.value}\n"
        f"Entry: {decision.entry_price:.2f}  |  SL: {decision.stoploss:.2f}\n"
        f"T1: {decision.target1:.2f}  |  T2: {decision.target2:.2f}\n"
        f"Confidence: {decision.confidence_score:.0f}%\n"
        f"{decision.reason}"
    )


def _format_exit_message(trade: Trade) -> str:
    return (
        f"Symbol: {trade.symbol}\n"
        f"Strategy: {trade.strategy.value}\n"
        f"Entry: {trade.entry_price:.2f}  →  Exit: {trade.exit_price:.2f}\n"
        f"PnL: ₹{trade.pnl:,.2f}"
    )


class TelegramAlert:
    """Send alerts via Telegram bot (optional)."""

    def __init__(self) -> None:
        self._bot = None

    async def send(self, text: str) -> None:
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return
        try:
            from telegram import Bot

            if self._bot is None:
                self._bot = Bot(token=settings.telegram_bot_token)
            await self._bot.send_message(chat_id=settings.telegram_chat_id, text=text)
            logger.info("Telegram alert sent")
        except Exception:
            logger.debug("Telegram send skipped (not configured or error)")


class EmailAlert:
    """Send alerts via email (optional)."""

    async def send(self, subject: str, body: str) -> None:
        if not settings.smtp_user or not settings.smtp_password:
            return
        try:
            from email.mime.text import MIMEText

            import aiosmtplib

            msg = MIMEText(body, "plain", "utf-8")
            msg["From"] = settings.smtp_user
            msg["To"] = settings.alert_email_to
            msg["Subject"] = subject

            await aiosmtplib.send(
                msg,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user,
                password=settings.smtp_password,
                use_tls=False,
                start_tls=True,
            )
            logger.info("Email alert sent: %s", subject)
        except Exception:
            logger.debug("Email send skipped (not configured or error)")


class AlertManager:
    """Unified alert dispatcher — always writes to UI store, optionally to Telegram/Email."""

    def __init__(self) -> None:
        self.store = alert_store
        self.telegram = TelegramAlert()
        self.email = EmailAlert()
        self._history_logger = None

    def _get_history_logger(self):
        if self._history_logger is None:
            from app.trading.history_logger import HistoryLogger
            self._history_logger = HistoryLogger()
        return self._history_logger

    async def send_signal_alert(self, signal: StrategySignal, decision: AIDecision) -> None:
        msg = _format_signal_message(signal, decision)
        inst = getattr(signal, 'instrument', 'NIFTY')
        title = f"TRADE SIGNAL — {inst} {int(signal.strike_price)} {signal.option_type.value}"

        # Always store for UI
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=title,
            message=msg,
            timestamp=datetime.now(_IST),
            strategy=signal.strategy.value,
        )
        self.store.add(alert)
        await self._get_history_logger().save_alert(alert)
        logger.info("Alert: %s", title)

        # Optional external channels
        await self.telegram.send(f"🔔 {title}\n{msg}")
        await self.email.send(f"TradeAI: {title}", msg)

    async def send_exit_alert(self, trade: Trade) -> None:
        msg = _format_exit_message(trade)
        is_win = (trade.pnl or 0) > 0
        title = f"TRADE {'WIN' if is_win else 'LOSS'} — {trade.symbol}"

        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=title,
            message=msg,
            timestamp=datetime.now(_IST),
            trade_id=trade.trade_id,
            strategy=trade.strategy.value,
            pnl=trade.pnl,
        )
        self.store.add(alert)
        await self._get_history_logger().save_alert(alert)
        logger.info("Alert: %s (PnL=%.2f)", title, trade.pnl or 0)

        await self.telegram.send(f"{'✅' if is_win else '❌'} {title}\n{msg}")
        await self.email.send(f"TradeAI: {title}", msg)

    async def send_daily_report(self, report: str) -> None:
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="report",
            title="DAILY REPORT",
            message=report,
            timestamp=datetime.now(_IST),
        )
        self.store.add(alert)
        await self._get_history_logger().save_alert(alert)
        await self.telegram.send(report)

    async def send_info(self, title: str, message: str) -> None:
        """Store an informational alert (no external push)."""
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="info",
            title=title,
            message=message,
            timestamp=datetime.now(_IST),
        )
        self.store.add(alert)
        logger.info("Alert [info]: %s", title)
        await self._get_history_logger().save_alert(alert)
