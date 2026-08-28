"""Account-level daily-loss kill switch (per BrokerAccount).

A background watchdog polls broker positions every few seconds and
computes the total realised + unrealised PnL across **every** open
position for each active broker account (including ones placed
manually through the broker's web/app). When PnL crosses below the
per-account loss limit for a given account:

* Only THAT account's open positions are force-closed via MARKET orders.
* A per-account ``locked`` flag is set.
* Every subsequent :py:meth:`DhanBroker.place_order` call routed to that
  account is evaluated: orders that REDUCE an existing position
  (squareoff / partial reduce) are allowed; orders that open or grow a
  position are rejected.

The lock resets automatically at IST midnight and can be released
manually per-account via :py:func:`reset_kill_switch`.

Profit side is intentionally uncapped.

Multi-account
-------------
State is keyed by ``account_id`` (int). ``account_id == 0`` is reserved
for the legacy env-based singleton used when ``TRADING_ACCOUNT=dhan``
is set but no ``BrokerAccount`` rows exist. Real accounts use their DB
primary key.

Currently only Dhan is wired for the watchdog — Angel/Kite adapters
don't yet expose a positions feed comparable to
:py:meth:`DhanClient.get_positions`.
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

# Sentinel account_id for the legacy env-based Dhan singleton (no DB row).
ENV_ACCOUNT_ID = 0


@dataclass
class KillSwitchState:
    """Snapshot of a single account's kill switch — exposed via API for the UI."""

    account_id: int = ENV_ACCOUNT_ID
    account_name: str = ""
    broker: str = "dhan"
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
    # Per-security pending exit qty (set when the watchdog has already
    # issued a force-close MARKET order but the fill hasn't been reflected
    # in the next /positions snapshot). Prevents the next watchdog tick
    # from firing a duplicate squareoff that would flip the net position.
    pending_exit_qty: dict[str, int] = field(default_factory=dict)
    # IST timestamp each entry in ``pending_exit_qty`` was recorded. Used
    # to expire the dedup guard after a short grace period so a rejected
    # / partially-filled force-close order gets retried instead of being
    # silently ignored forever (see 2026-08-28 stuck-kill-switch incident).
    pending_exit_at: dict[str, datetime] = field(default_factory=dict)
    # IST timestamp of the most recent force-close pass. Used as a safety
    # cooldown so we never spam squareoffs faster than fills can reflect.
    last_force_close_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "account_name": self.account_name,
            "broker": self.broker,
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


# ── Module-level state (process-wide, keyed by account_id) ──────────
_states: dict[int, KillSwitchState] = {}
_state_lock = threading.RLock()  # Hot path is sync (place_order gate).
_watchdog_task: Optional[asyncio.Task] = None


def _get_or_create_state(
    account_id: int,
    *,
    account_name: str = "",
    broker: str = "dhan",
    enabled: Optional[bool] = None,
    limit: Optional[float] = None,
) -> KillSwitchState:
    """Return the state row for ``account_id``, creating it lazily. Caller must hold ``_state_lock``."""
    st = _states.get(int(account_id))
    if st is None:
        st = KillSwitchState(
            account_id=int(account_id),
            account_name=account_name or ("env" if account_id == ENV_ACCOUNT_ID else f"account-{account_id}"),
            broker=broker or "dhan",
            enabled=bool(enabled) if enabled is not None else settings.account_kill_switch_enabled,
            limit=float(limit) if limit is not None and float(limit) > 0 else float(settings.account_max_daily_loss),
        )
        _states[int(account_id)] = st
    else:
        # Keep the display metadata in sync when the DB row changes.
        if account_name:
            st.account_name = account_name
        if broker:
            st.broker = broker
    return st


# ── Public read API ─────────────────────────────────────────────────


def _snapshot(st: KillSwitchState) -> KillSwitchState:
    """Shallow copy — safe for read-only callers."""
    return KillSwitchState(
        account_id=st.account_id,
        account_name=st.account_name,
        broker=st.broker,
        enabled=st.enabled,
        limit=st.limit,
        locked=st.locked,
        current_pnl=st.current_pnl,
        tripped_at=st.tripped_at,
        tripped_pnl=st.tripped_pnl,
        last_poll_at=st.last_poll_at,
        last_error=st.last_error,
        state_date=st.state_date,
        net_qty_by_security=dict(st.net_qty_by_security),
        pending_exit_qty=dict(st.pending_exit_qty),
        pending_exit_at=dict(st.pending_exit_at),
        last_force_close_at=st.last_force_close_at,
    )


def get_state(account_id: int = ENV_ACCOUNT_ID) -> KillSwitchState:
    """Return a snapshot of a single account's kill switch state."""
    with _state_lock:
        st = _get_or_create_state(int(account_id))
        return _snapshot(st)


def get_all_states() -> list[KillSwitchState]:
    """Return a snapshot of every tracked account's kill switch state."""
    with _state_lock:
        return [_snapshot(st) for st in _states.values()]


def is_locked(account_id: int = ENV_ACCOUNT_ID) -> bool:
    with _state_lock:
        st = _states.get(int(account_id))
        if st is None:
            return False
        if st.locked:
            _maybe_auto_reset_locked(st)
        return st.locked


def is_enabled(account_id: int = ENV_ACCOUNT_ID) -> bool:
    with _state_lock:
        st = _states.get(int(account_id))
        if st is None:
            return bool(settings.account_kill_switch_enabled)
        return bool(st.enabled)


# ── Public mutation API ─────────────────────────────────────────────


def update_settings(
    account_id: int = ENV_ACCOUNT_ID,
    *,
    enabled: Optional[bool] = None,
    limit: Optional[float] = None,
) -> KillSwitchState:
    """Update enable/limit at runtime for a single account. Settings UI calls this."""
    with _state_lock:
        st = _get_or_create_state(int(account_id))
        if enabled is not None:
            st.enabled = bool(enabled)
        if limit is not None and float(limit) > 0:
            st.limit = float(limit)
        # Note: changing the limit does NOT unlock a tripped switch within
        # the same IST day — the explicit reset is required so this can't
        # be used to escape a losing session.
        if st.state_date is None:
            st.state_date = datetime.now(_IST).date()
        logger.info(
            "Kill switch settings updated: account_id=%s enabled=%s limit=%.2f locked=%s",
            st.account_id, st.enabled, st.limit, st.locked,
        )
        return _snapshot(st)


def reset_kill_switch(
    account_id: int = ENV_ACCOUNT_ID,
    reason: str = "manual_reset",
) -> KillSwitchState:
    """Clear a tripped lock for the current IST day for one account."""
    with _state_lock:
        st = _get_or_create_state(int(account_id))
        was_locked = st.locked
        st.locked = False
        st.tripped_at = None
        st.tripped_pnl = 0.0
        st.last_error = ""
        # Clearing pending exits ensures a re-trip later in the day will
        # cleanly fire fresh force-close orders on any positions that
        # have since been re-opened.
        st.pending_exit_qty = {}
        st.pending_exit_at = {}
        st.last_force_close_at = None
        if was_locked:
            logger.warning("Kill switch RESET account_id=%s (%s)", st.account_id, reason)
        return _snapshot(st)


# ── place_order gate (hot path — runs on every order) ───────────────


@dataclass
class GateDecision:
    allowed: bool
    reason: str
    current_qty: int = 0
    incoming_signed_qty: int = 0
    new_qty: int = 0


def evaluate_order(
    security_id: str,
    side: str,
    quantity: int,
    account_id: int = ENV_ACCOUNT_ID,
) -> GateDecision:
    """Decide whether to allow an order while the switch is locked.

    The rule: an order is allowed iff it REDUCES the absolute value of
    the net signed position for ``security_id`` within the given
    ``account_id``. New entries, flips that would result in a larger
    absolute position, and pyramiding are blocked. Only the tripped
    account's orders are affected — other accounts continue trading.

    When the account's switch is NOT locked (the common case), this
    returns ``allowed=True`` immediately so the broker hot path stays
    fast.
    """
    if not is_enabled(account_id) or not is_locked(account_id):
        return GateDecision(allowed=True, reason="not_locked")

    sid = str(security_id)
    sgn = +1 if side.upper() == "BUY" else -1
    incoming = sgn * abs(int(quantity))

    with _state_lock:
        st = _get_or_create_state(int(account_id))
        current = int(st.net_qty_by_security.get(sid, 0))
        new_qty = current + incoming
        if abs(new_qty) < abs(current):
            # Apply the delta locally so the very next order sees the
            # already-reduced position, even before the next /positions
            # poll refreshes the snapshot. Without this, several reducing
            # orders fired within one poll window could collectively
            # over-reduce and flip the position to the opposite side.
            st.net_qty_by_security[sid] = new_qty
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


def _maybe_auto_reset_locked(st: KillSwitchState) -> None:
    """Called under lock. Reset state on IST day rollover for one account."""
    today = datetime.now(_IST).date()
    if st.state_date is not None and st.state_date != today:
        logger.info(
            "Kill switch auto-reset on IST day rollover account_id=%s (%s → %s)",
            st.account_id, st.state_date, today,
        )
        st.locked = False
        st.tripped_at = None
        st.tripped_pnl = 0.0
        st.last_error = ""
        st.state_date = today


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


def _freeze_qty_for_symbol(trading_symbol: str) -> int:
    """Return the exchange per-order freeze qty for a position's trading
    symbol, or 0 if the underlying can't be identified.

    Reuses the same table :py:mod:`dhan_broker` uses for live entries so
    a force-close squareoff never exceeds the exchange's iceberg limit.
    """
    from app.execution.dhan_broker import _FREEZE_QTY_BY_UNDERLYING

    sym = (trading_symbol or "").upper().strip()
    for underlying, qty in _FREEZE_QTY_BY_UNDERLYING.items():
        if sym.startswith(underlying):
            return qty
    return 0


async def _force_close_all(
    dhan_client,
    positions: list[dict],
    st: KillSwitchState,
) -> int:
    """MARKET-squareoff every open position of one account. Returns the
    number of squareoff orders attempted. Each call is wrapped so one
    failure doesn't stop the rest.

    Large positions are sliced to the exchange's per-order freeze qty
    (same table/logic as live entries) so a single oversized MARKET
    order can't silently under-fill (Dhan/NSE "iceberg" behaviour drops
    the qty above the freeze limit instead of rejecting outright).
    """
    from app.execution.dhan_broker import _split_quantity

    closed = 0
    now = datetime.now(_IST)
    # How long a "pending" force-close is trusted before we retry it
    # anyway. Must be long enough to let one poll cycle's fill reflect,
    # but short enough that a rejected/partially-filled order (e.g. from
    # a freeze-qty violation) doesn't stay stuck for the rest of the day.
    retry_grace_seconds = max(10.0, float(settings.account_kill_switch_poll_seconds) * 3)

    for p in positions:
        try:
            sec_id = str(p.get("securityId") or p.get("security_id") or "")
            seg = str(p.get("exchangeSegment") or p.get("exchange_segment") or "")
            net_qty = int(p.get("netQty") or p.get("net_qty") or 0)
            product = str(p.get("productType") or p.get("product_type") or "INTRADAY")
            if not sec_id or net_qty == 0 or not seg:
                continue

            # Skip positions that already have a pending squareoff with
            # the same sign AND that squareoff was issued recently — the
            # previous tick's exit is likely still settling. Once the
            # grace period elapses we retry regardless, so a rejected or
            # partially-filled order can never block force-close forever.
            with _state_lock:
                pending = int(st.pending_exit_qty.get(sec_id, 0))
                pending_at = st.pending_exit_at.get(sec_id)
            if (
                pending != 0
                and ((pending > 0) == (net_qty > 0))
                and pending_at is not None
                and (now - pending_at).total_seconds() < retry_grace_seconds
            ):
                logger.info(
                    "KILL SWITCH skip duplicate force-close: account_id=%s security_id=%s "
                    "net_qty=%d pending_exit=%d age=%.1fs",
                    st.account_id, sec_id, net_qty, pending, (now - pending_at).total_seconds(),
                )
                continue

            side = "SELL" if net_qty > 0 else "BUY"
            total_qty = abs(net_qty)
            trading_symbol = str(p.get("tradingSymbol") or p.get("trading_symbol") or "")
            freeze_qty = _freeze_qty_for_symbol(trading_symbol)
            lot_size = 0
            if freeze_qty and total_qty > freeze_qty:
                try:
                    lot_size = await asyncio.to_thread(
                        dhan_client.get_lot_size_for_security_id, sec_id
                    )
                except Exception:
                    logger.debug(
                        "Kill switch: lot-size lookup failed for security_id=%s", sec_id,
                        exc_info=True,
                    )
            slice_qtys = _split_quantity(total_qty, freeze_qty, lot_size or total_qty)
            if len(slice_qtys) > 1:
                logger.warning(
                    "KILL SWITCH FORCE-CLOSE SLICING account_id=%s security_id=%s "
                    "total_qty=%d freeze_qty=%d lot_size=%d → %d slices %s",
                    st.account_id, sec_id, total_qty, freeze_qty, lot_size, len(slice_qtys), slice_qtys,
                )

            filled_qty = 0
            for idx, slice_qty in enumerate(slice_qtys):
                logger.warning(
                    "KILL SWITCH FORCE-CLOSE account_id=%s: security_id=%s qty=%d "
                    "side=%s seg=%s slice=%d/%d",
                    st.account_id, sec_id, slice_qty, side, seg, idx + 1, len(slice_qtys),
                )
                resp = await asyncio.to_thread(
                    dhan_client.place_order,
                    security_id=sec_id,
                    exchange_segment=seg,
                    transaction_type=side,
                    quantity=slice_qty,
                    order_type="MARKET",
                    product_type=product,
                    price=0.0,
                    trigger_price=0.0,
                )
                logger.info("KILL SWITCH force-close response: %s", resp)
                if not isinstance(resp, dict) or resp.get("status") != "success":
                    logger.error(
                        "KILL SWITCH FORCE-CLOSE ORDER REJECTED account_id=%s security_id=%s "
                        "slice=%d/%d qty=%d resp=%s — will retry next poll",
                        st.account_id, sec_id, idx + 1, len(slice_qtys), slice_qty, resp,
                    )
                    break
                filled_qty += slice_qty

            # Record whatever we actually managed to place this pass so
            # the dedup guard reflects reality — not the originally
            # requested (possibly larger) qty. A rejected first slice
            # leaves nothing pending, so the very next poll retries
            # immediately instead of waiting out the grace period.
            with _state_lock:
                if filled_qty > 0:
                    st.pending_exit_qty[sec_id] = filled_qty if net_qty > 0 else -filled_qty
                    st.pending_exit_at[sec_id] = now
                else:
                    st.pending_exit_qty.pop(sec_id, None)
                    st.pending_exit_at.pop(sec_id, None)
                st.last_force_close_at = now
            if filled_qty > 0:
                closed += 1
        except Exception:
            logger.exception(
                "Kill switch force-close failed account_id=%s position=%s",
                st.account_id, p,
            )
    return closed


async def _send_telegram_alert(text: str) -> None:
    try:
        from app.alerts.alert_manager import AlertManager
        am = AlertManager()
        await am.telegram.send(text)
    except Exception:
        logger.debug("Kill switch telegram alert failed", exc_info=True)


async def _watchdog_loop() -> None:
    """Poll positions per Dhan BrokerAccount, compute PnL, trip switches."""
    logger.info(
        "Account kill switch watchdog started (interval=%.1fs)",
        settings.account_kill_switch_poll_seconds,
    )
    while True:
        try:
            await asyncio.sleep(max(1.0, float(settings.account_kill_switch_poll_seconds)))
            if settings.paper_trading:
                # No real positions in paper mode; nothing to enforce.
                continue

            targets = await _resolve_dhan_targets()
            if not targets:
                continue

            # Poll accounts in parallel so a slow account can't stall the
            # rest of the watchdog cycle.
            await asyncio.gather(
                *(_watchdog_poll_one(t) for t in targets),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("Account kill switch watchdog cancelled")
            raise
        except Exception:
            logger.exception("Kill switch watchdog outer iteration failed")


async def _watchdog_poll_one(target: dict) -> None:
    """Poll a single (broker, state) pair. Isolated so per-account errors
    never trip other accounts.
    """
    st: KillSwitchState = target["state"]
    broker = target["broker"]
    try:
        if not is_enabled(st.account_id):
            return
        if broker is None or getattr(broker, "client", None) is None:
            return

        positions = await asyncio.to_thread(broker.client.get_positions)
        pnl, by_sec, open_positions = _compute_pnl_and_qty(positions)

        now = datetime.now(_IST)
        with _state_lock:
            st.last_poll_at = now
            st.current_pnl = pnl
            st.net_qty_by_security = by_sec
            st.last_error = ""
            if st.state_date is None:
                st.state_date = now.date()
            _maybe_auto_reset_locked(st)
            # Drop stale pending-exit entries that the latest snapshot
            # proves are already settled.
            if st.pending_exit_qty:
                cleaned: dict[str, int] = {}
                for sid, pending_qty in st.pending_exit_qty.items():
                    cur = by_sec.get(sid, 0)
                    if cur == 0:
                        continue
                    if (pending_qty > 0) != (cur > 0):
                        continue
                    cleaned[sid] = cur if abs(cur) < abs(pending_qty) else pending_qty
                st.pending_exit_qty = cleaned
                st.pending_exit_at = {
                    sid: ts for sid, ts in st.pending_exit_at.items() if sid in cleaned
                }
            already_locked = st.locked
            should_trip = (
                st.enabled
                and not already_locked
                and pnl <= -abs(st.limit)
            )
            if should_trip:
                st.locked = True
                st.tripped_at = now
                st.tripped_pnl = pnl
                st.state_date = now.date()

        if should_trip:
            logger.error(
                "ACCOUNT KILL SWITCH TRIPPED account_id=%s (%s): pnl=%.2f limit=-%.2f → "
                "force-closing %d open positions",
                st.account_id, st.account_name, pnl, st.limit, len(open_positions),
            )
            await _send_telegram_alert(
                f"🛑 KILL SWITCH TRIPPED — {st.account_name}\n"
                f"PnL: ₹{pnl:,.2f} (limit ₹-{st.limit:,.0f})\n"
                f"Force-closing {len(open_positions)} position(s). "
                f"New entries on this account are blocked for the rest of the IST day."
            )
            await _force_close_all(broker.client, open_positions, st)
        elif already_locked and open_positions:
            logger.warning(
                "KILL SWITCH still active account_id=%s with %d open positions → "
                "continuing force-close pass",
                st.account_id, len(open_positions),
            )
            await _force_close_all(broker.client, open_positions, st)
    except Exception as exc:
        logger.exception("Kill switch poll failed for account_id=%s", st.account_id)
        with _state_lock:
            st.last_error = str(exc)


async def _resolve_dhan_targets() -> list[dict]:
    """Return the list of ``{state, broker}`` pairs to poll this tick.

    Enumerates active Dhan ``BrokerAccount`` rows first. When there are
    none, falls back to the env-based singleton (``account_id=0``) so
    legacy single-account setups keep working.
    """
    targets: list[dict] = []
    seen_ids: set[int] = set()

    try:
        rows = await _fetch_active_dhan_accounts()
    except Exception:
        logger.exception("Failed to fetch Dhan broker accounts for kill switch")
        rows = []

    if rows:
        from app.execution.broker_factory import build_broker_from_row

        for row in rows:
            account_id = int(row.get("id") or 0)
            if not account_id or account_id in seen_ids:
                continue
            seen_ids.add(account_id)
            try:
                broker = build_broker_from_row(row)
            except Exception:
                logger.exception(
                    "Kill switch: failed to build broker for account_id=%s", account_id,
                )
                broker = None
            with _state_lock:
                st = _get_or_create_state(
                    account_id,
                    account_name=str(row.get("name") or f"account-{account_id}"),
                    broker=str(row.get("broker") or "dhan"),
                )
                # Adopt DB-persisted enabled/limit as source of truth when set.
                db_enabled = row.get("kill_switch_enabled")
                db_limit = row.get("daily_loss_limit")
                if db_enabled is not None:
                    st.enabled = bool(db_enabled)
                if db_limit is not None:
                    try:
                        v = float(db_limit)
                        if v > 0:
                            st.limit = v
                    except (TypeError, ValueError):
                        pass
            targets.append({"state": st, "broker": broker})

    # Legacy env-based fallback: keep the old singleton working when no
    # BrokerAccount rows exist and TRADING_ACCOUNT=dhan is configured.
    if not rows and (settings.trading_account or "").lower() == "dhan":
        try:
            from app.execution.dhan_broker import DhanBroker

            broker = _get_or_build_env_dhan_broker(DhanBroker)
            try:
                broker._ensure_fresh_client()  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Kill switch: env broker refresh failed", exc_info=True)
            with _state_lock:
                st = _get_or_create_state(ENV_ACCOUNT_ID, account_name="env", broker="dhan")
            targets.append({"state": st, "broker": broker})
        except Exception:
            logger.exception("Kill switch: env-based Dhan fallback failed")

    return targets


async def _fetch_active_dhan_accounts() -> list[dict]:
    """Return active Dhan BrokerAccount rows as plain dicts (async-safe)."""
    from sqlalchemy import select
    from app.db.account_models import BrokerAccount
    from app.db.models import AsyncSessionLocal

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(BrokerAccount).where(
                    BrokerAccount.is_active == True,  # noqa: E712
                    BrokerAccount.broker == "dhan",
                ).order_by(BrokerAccount.id.asc())
            )
        ).scalars().all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "broker": r.broker,
                "client_id": r.client_id,
                "api_key": r.api_key,
                "api_secret": r.api_secret,
                "password": r.password,
                "mpin": r.mpin,
                "totp_secret": r.totp_secret,
                "access_token": r.access_token,
                "refresh_token": r.refresh_token,
                "proxy_url": r.proxy_url,
                "kill_switch_enabled": r.kill_switch_enabled if r.kill_switch_enabled is not None else True,
                "daily_loss_limit": r.daily_loss_limit if r.daily_loss_limit is not None else 6000.0,
                "updated_at": r.updated_at.isoformat() if r.updated_at else "",
            }
            for r in rows
        ]


# Tiny singleton wrapper so the watchdog reuses the same env-based DhanBroker
# instance as the order path (and therefore picks up token rotations).
_dhan_broker_singleton = None


def _get_or_build_env_dhan_broker(broker_cls):
    global _dhan_broker_singleton
    if _dhan_broker_singleton is None:
        _dhan_broker_singleton = broker_cls()
    return _dhan_broker_singleton


# ── Public lifecycle hooks ──────────────────────────────────────────


def start_watchdog(loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[asyncio.Task]:
    """Start the watchdog task. Idempotent.

    Must be called from inside a running event loop (FastAPI lifespan
    satisfies this). Falls back to ``get_event_loop()`` only when no
    running loop is available, to keep the legacy sync entrypoint working.
    """
    global _watchdog_task
    if _watchdog_task is not None and not _watchdog_task.done():
        return _watchdog_task
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
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
