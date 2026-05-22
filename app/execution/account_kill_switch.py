"""Account-level daily-loss kill switch.

A background watchdog polls broker positions every few seconds and
computes the total realised + unrealised PnL across **every** open
position in the trading account (including ones placed manually
through the broker's web/app). When PnL crosses below the configured
loss limit:

* All open positions are force-closed via MARKET orders.
* A process-wide ``_locked`` flag is set.
* Every subsequent :py:meth:`DhanBroker.place_order` call is evaluated:
  orders that REDUCE an existing position (squareoff / partial reduce)
  are allowed; orders that open or grow a position are rejected.

The lock resets automatically at IST midnight and can be released
manually via :py:func:`reset_kill_switch`.

Profit side is intentionally uncapped.

The watchdog only activates when ``TRADING_ACCOUNT=dhan`` because it
depends on :py:class:`DhanClient.get_positions` for the per-position
PnL feed. Adding Angel/Kite is a future extension.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import pytz

from app.core.config import settings

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


@dataclass
class KillSwitchState:
    """Snapshot of the kill switch — exposed via API for the UI."""

    enabled: bool = True
    limit: float = 6000.0
    locked: bool = False
    current_pnl: float = 0.0
    tripped_at: Optional[datetime] = None
    tripped_pnl: float = 0.0
    last_poll_at: Optional[datetime] = None
    last_error: str = ""
    # IST date for which the current tripped/lock state applies. When the
    # IST date rolls over we auto-reset.
    state_date: Optional[date] = None
    # Snapshot of net qty per security at last poll. Used by the place_order
    # gate to decide whether an incoming order REDUCES or GROWS the position
    # without having to hit the broker on every order.
    net_qty_by_security: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "limit": self.limit,
            "locked": self.locked,
            "current_pnl": round(self.current_pnl, 2),
            "tripped_at": self.tripped_at.isoformat() if self.tripped_at else None,
            "tripped_pnl": round(self.tripped_pnl, 2),
            "last_poll_at": self.last_poll_at.isoformat() if self.last_poll_at else None,
            "last_error": self.last_error,
            "state_date": self.state_date.isoformat() if self.state_date else None,
        }


# ── Module-level state (process-wide singleton) ─────────────────────
_state = KillSwitchState(
    enabled=settings.account_kill_switch_enabled,
    limit=float(settings.account_max_daily_loss),
)
_state_lock = threading.RLock()  # Hot path is sync (place_order gate).
_watchdog_task: Optional[asyncio.Task] = None


# ── Public read API ─────────────────────────────────────────────────


def get_state() -> KillSwitchState:
    """Return a snapshot copy of the current kill switch state."""
    with _state_lock:
        # Shallow copy — dataclass fields are primitives / dict that we
        # rebuild on every poll, so this is safe for read-only callers.
        return KillSwitchState(
            enabled=_state.enabled,
            limit=_state.limit,
            locked=_state.locked,
            current_pnl=_state.current_pnl,
            tripped_at=_state.tripped_at,
            tripped_pnl=_state.tripped_pnl,
            last_poll_at=_state.last_poll_at,
            last_error=_state.last_error,
            state_date=_state.state_date,
            net_qty_by_security=dict(_state.net_qty_by_security),
        )


def is_locked() -> bool:
    with _state_lock:
        if _state.locked:
            _maybe_auto_reset_locked()
        return _state.locked


def is_enabled() -> bool:
    with _state_lock:
        return bool(_state.enabled)


# ── Public mutation API ─────────────────────────────────────────────


def update_settings(*, enabled: Optional[bool] = None, limit: Optional[float] = None) -> KillSwitchState:
    """Update enable/limit at runtime. Settings UI calls this."""
    with _state_lock:
        if enabled is not None:
            _state.enabled = bool(enabled)
        if limit is not None and float(limit) > 0:
            _state.limit = float(limit)
        # Note: changing the limit does NOT unlock a tripped switch within
        # the same IST day — the explicit reset is required so this can't
        # be used to escape a losing session.
        logger.info(
            "Kill switch settings updated: enabled=%s limit=%.2f locked=%s",
            _state.enabled, _state.limit, _state.locked,
        )
        return get_state()


def reset_kill_switch(reason: str = "manual_reset") -> KillSwitchState:
    """Clear a tripped lock for the current IST day."""
    with _state_lock:
        was_locked = _state.locked
        _state.locked = False
        _state.tripped_at = None
        _state.tripped_pnl = 0.0
        _state.last_error = ""
        if was_locked:
            logger.warning("Kill switch RESET (%s)", reason)
        return get_state()


# ── place_order gate (hot path — runs on every order) ───────────────


@dataclass
class GateDecision:
    allowed: bool
    reason: str
    current_qty: int = 0
    incoming_signed_qty: int = 0
    new_qty: int = 0


def evaluate_order(security_id: str, side: str, quantity: int) -> GateDecision:
    """Decide whether to allow an order while the switch is locked.

    The rule: an order is allowed iff it REDUCES the absolute value of
    the net signed position for ``security_id``. New entries, flips
    that would result in a larger absolute position, and pyramiding
    are blocked.

    When the switch is NOT locked (the common case), this returns
    ``allowed=True`` immediately so the broker hot path stays fast.
    """
    if not is_enabled() or not is_locked():
        return GateDecision(allowed=True, reason="not_locked")

    with _state_lock:
        current = int(_state.net_qty_by_security.get(str(security_id), 0))

    sgn = +1 if side.upper() == "BUY" else -1
    incoming = sgn * abs(int(quantity))
    new_qty = current + incoming
    if abs(new_qty) < abs(current):
        return GateDecision(
            allowed=True,
            reason="reducing_position",
            current_qty=current,
            incoming_signed_qty=incoming,
            new_qty=new_qty,
        )
    return GateDecision(
        allowed=False,
        reason="account_kill_switch_triggered_daily_loss",
        current_qty=current,
        incoming_signed_qty=incoming,
        new_qty=new_qty,
    )


# ── Internal: watchdog loop ─────────────────────────────────────────


def _maybe_auto_reset_locked() -> None:
    """Called under lock. Reset state on IST day rollover."""
    today = datetime.now(_IST).date()
    if _state.state_date is not None and _state.state_date != today:
        logger.info(
            "Kill switch auto-reset on IST day rollover (%s → %s)",
            _state.state_date, today,
        )
        _state.locked = False
        _state.tripped_at = None
        _state.tripped_pnl = 0.0
        _state.last_error = ""
        _state.state_date = today


def _compute_pnl_and_qty(positions: list[dict]) -> tuple[float, dict[str, int], list[dict]]:
    """Return (total_pnl, net_qty_by_security, open_positions).

    ``open_positions`` contains only rows with non-zero net quantity —
    used by the watchdog's force-close pass. PnL aggregates realised +
    unrealised across every row (including zero-qty rows that still
    carry today's realised profit).
    """
    total = 0.0
    by_sec: dict[str, int] = {}
    open_positions: list[dict] = []
    for p in positions or []:
        try:
            realised = float(p.get("realizedProfit") or p.get("realized_profit") or 0)
            unrealised = float(p.get("unrealizedProfit") or p.get("unrealized_profit") or 0)
            total += realised + unrealised
            sec_id = str(p.get("securityId") or p.get("security_id") or "")
            if not sec_id:
                continue
            net_qty = int(p.get("netQty") or p.get("net_qty") or 0)
            by_sec[sec_id] = by_sec.get(sec_id, 0) + net_qty
            if net_qty != 0:
                open_positions.append(p)
        except (TypeError, ValueError):
            continue
    return total, by_sec, open_positions


async def _force_close_all(dhan_client, positions: list[dict]) -> int:
    """MARKET-squareoff every open position. Returns the number of
    squareoff orders attempted. Each call is wrapped so one failure
    doesn't stop the rest.
    """
    closed = 0
    for p in positions:
        try:
            sec_id = str(p.get("securityId") or p.get("security_id") or "")
            seg = str(p.get("exchangeSegment") or p.get("exchange_segment") or "")
            net_qty = int(p.get("netQty") or p.get("net_qty") or 0)
            product = str(p.get("productType") or p.get("product_type") or "INTRADAY")
            if not sec_id or net_qty == 0 or not seg:
                continue
            side = "SELL" if net_qty > 0 else "BUY"
            qty = abs(net_qty)
            logger.warning(
                "KILL SWITCH FORCE-CLOSE: security_id=%s qty=%d side=%s seg=%s",
                sec_id, qty, side, seg,
            )
            resp = await asyncio.to_thread(
                dhan_client.place_order,
                security_id=sec_id,
                exchange_segment=seg,
                transaction_type=side,
                quantity=qty,
                order_type="MARKET",
                product_type=product,
                price=0.0,
                trigger_price=0.0,
            )
            logger.info("KILL SWITCH force-close response: %s", resp)
            closed += 1
        except Exception:
            logger.exception("Kill switch force-close failed for position %s", p)
    return closed


async def _send_telegram_alert(text: str) -> None:
    try:
        from app.alerts.alert_manager import AlertManager
        am = AlertManager()
        await am.telegram.send(text)
    except Exception:
        logger.debug("Kill switch telegram alert failed", exc_info=True)


async def _watchdog_loop() -> None:
    """Poll positions, compute account PnL, trip the switch if breached."""
    logger.info(
        "Account kill switch watchdog started (enabled=%s limit=%.2f interval=%.1fs)",
        _state.enabled, _state.limit, settings.account_kill_switch_poll_seconds,
    )
    while True:
        try:
            await asyncio.sleep(max(1.0, float(settings.account_kill_switch_poll_seconds)))
            if not is_enabled():
                continue
            if settings.paper_trading:
                # No real positions in paper mode; nothing to enforce.
                continue
            if (settings.trading_account or "").lower() != "dhan":
                # Currently only Dhan is wired — Angel/Kite would need
                # their own positions adapter.
                continue

            # Lazy import to avoid a circular import at module load.
            from app.execution.dhan_broker import DhanBroker

            broker = _get_or_build_dhan_broker(DhanBroker)
            if broker is None or broker.client is None:
                continue

            positions = await asyncio.to_thread(broker.client.get_positions)
            pnl, by_sec, open_positions = _compute_pnl_and_qty(positions)

            now = datetime.now(_IST)
            with _state_lock:
                _state.last_poll_at = now
                _state.current_pnl = pnl
                _state.net_qty_by_security = by_sec
                _state.last_error = ""
                if _state.state_date is None:
                    _state.state_date = now.date()
                _maybe_auto_reset_locked()
                already_locked = _state.locked
                should_trip = (
                    _state.enabled
                    and not already_locked
                    and pnl <= -abs(_state.limit)
                )
                if should_trip:
                    _state.locked = True
                    _state.tripped_at = now
                    _state.tripped_pnl = pnl
                    _state.state_date = now.date()

            if should_trip:
                logger.error(
                    "ACCOUNT KILL SWITCH TRIPPED: pnl=%.2f limit=-%.2f → "
                    "force-closing %d open positions",
                    pnl, _state.limit, len(open_positions),
                )
                await _send_telegram_alert(
                    f"🛑 ACCOUNT KILL SWITCH TRIPPED\n"
                    f"PnL: ₹{pnl:,.2f} (limit ₹-{_state.limit:,.0f})\n"
                    f"Force-closing {len(open_positions)} open position(s). "
                    f"New entries blocked for the rest of the IST day."
                )
                await _force_close_all(broker.client, open_positions)
        except asyncio.CancelledError:
            logger.info("Account kill switch watchdog cancelled")
            raise
        except Exception as exc:
            logger.exception("Kill switch watchdog iteration failed")
            with _state_lock:
                _state.last_error = str(exc)


# Tiny singleton wrapper so the watchdog reuses the same DhanBroker
# instance as the order path (and therefore picks up token rotations).
_dhan_broker_singleton = None


def _get_or_build_dhan_broker(broker_cls):
    global _dhan_broker_singleton
    if _dhan_broker_singleton is None:
        _dhan_broker_singleton = broker_cls()
    return _dhan_broker_singleton


# ── Public lifecycle hooks ──────────────────────────────────────────


def start_watchdog(loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[asyncio.Task]:
    """Start the watchdog task. Idempotent."""
    global _watchdog_task
    if _watchdog_task is not None and not _watchdog_task.done():
        return _watchdog_task
    loop = loop or asyncio.get_event_loop()
    _watchdog_task = loop.create_task(_watchdog_loop(), name="account_kill_switch_watchdog")
    return _watchdog_task


async def stop_watchdog() -> None:
    global _watchdog_task
    if _watchdog_task is None:
        return
    _watchdog_task.cancel()
    try:
        await _watchdog_task
    except (asyncio.CancelledError, Exception):
        pass
    _watchdog_task = None
