"""Dhan broker adapter — wraps :class:`DhanClient` with :class:`BaseBroker`.

Used when ``TRADING_ACCOUNT=dhan``. Order routing only — data feeds and
analytics still come from AngelOne (mirrors the Kite adapter design).

Key behaviour:
  - Translates Angel-format option symbols → Dhan ``securityId`` via the
    scrip-master CSV. Strategies can keep emitting Angel-style symbols.
  - Polls Dhan order status briefly after placement to capture fill price
    (matches the Kite/AngelOne brokers).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pytz

from app.core.config import settings
from app.data.dhan_client import (
    DhanClient,
    _to_dhan_exchange_segment,
    _EXCHANGE_REVERSE,
)
from app.execution.broker_base import (
    BaseBroker,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
)

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

ORDER_POLL_RETRIES = 8
ORDER_POLL_DELAY_SECONDS = 1.0

# ── Exchange freeze quantities (max units per single order) ──────────
# Beyond these caps, orders MUST be split into multiple child orders.
# Dhan's UI shows them as "iceberg" but only the first slice actually
# executes — the remainder is silently dropped. We pre-slice here so
# every chunk is a normal MARKET/LIMIT order that fully fills.
# Source: NSE/BSE F&O circulars, current as of FY26.
_FREEZE_QTY_BY_UNDERLYING: dict[str, int] = {
    "NIFTY": 1800,        # lot 65  → 27 lots/slice
    "BANKNIFTY": 900,     # lot 35  → 25 lots/slice
    "FINNIFTY": 1800,     # lot 65  → 27 lots/slice
    "MIDCPNIFTY": 1800,   # lot 120 → 15 lots/slice
    "NIFTYNXT50": 1000,   # lot 25  → 40 lots/slice
    "SENSEX": 1000,       # lot 20  → 50 lots/slice
    "BANKEX": 900,        # lot 30  → 30 lots/slice
    "SENSEX50": 1000,
}


def _resolve_freeze_qty(request: OrderRequest) -> int:
    """Return the per-order freeze qty for this underlying, or 0 if unknown
    (in which case the caller falls back to single-order behaviour)."""
    underlying = (request.underlying or "").upper().strip()
    if not underlying:
        # Try to parse from trading_symbol prefix (e.g. "NIFTY26MAY...PE")
        sym = (request.trading_symbol or "").upper()
        for key in _FREEZE_QTY_BY_UNDERLYING:
            if sym.startswith(key):
                underlying = key
                break
    return int(_FREEZE_QTY_BY_UNDERLYING.get(underlying, 0))


def _split_quantity(total_qty: int, freeze_qty: int, lot_size: int) -> list[int]:
    """Split ``total_qty`` into chunks each ≤ freeze_qty and a multiple of
    lot_size. The final chunk carries the remainder."""
    if freeze_qty <= 0 or total_qty <= freeze_qty:
        return [int(total_qty)]
    ls = max(1, int(lot_size or 1))
    # Largest multiple of lot_size that is ≤ freeze_qty
    chunk = (freeze_qty // ls) * ls
    if chunk <= 0:
        return [int(total_qty)]
    slices: list[int] = []
    remaining = int(total_qty)
    while remaining > chunk:
        slices.append(chunk)
        remaining -= chunk
    if remaining > 0:
        slices.append(remaining)
    return slices


def _supports_fresh_kw() -> bool:
    """``broker_credentials.get_dhan_credentials`` only accepts ``fresh=`` in\n    newer revisions; this lets the broker work with both signatures."""
    try:
        import inspect
        from app.db.broker_credentials import get_dhan_credentials
        return "fresh" in inspect.signature(get_dhan_credentials).parameters
    except Exception:
        return False


class DhanBroker(BaseBroker):
    """DhanHQ broker implementation."""

    def __init__(self) -> None:
        self._client: Optional[DhanClient] = None
        self._authenticated = False
        self._client_id: str = ""
        # Track the access_token that built the current `_client` so we can
        # detect token rotations (UI save / .env edit) and rebuild lazily
        # instead of relying on every caller to invoke reload_credentials().
        self._client_token: str = ""
        self._init_client()

    @property
    def name(self) -> str:
        return "Dhan"

    @property
    def client(self) -> Optional[DhanClient]:
        return self._client

    def _init_client(self, *, fresh: bool = False) -> None:
        from app.db.broker_credentials import get_dhan_credentials

        creds = get_dhan_credentials(fresh=fresh) if _supports_fresh_kw() else get_dhan_credentials()
        client_id = creds.get("client_id") or ""
        access_token = creds.get("access_token") or ""
        if not client_id or not access_token:
            logger.warning(
                "DhanBroker: client_id / access_token not configured "
                "(checked DB and DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN env)"
            )
            self._client = None
            self._client_id = ""
            self._client_token = ""
            return
        try:
            self._client = DhanClient(
                client_id=client_id,
                access_token=access_token,
            )
            self._client_id = client_id
            self._client_token = access_token
            logger.info(
                "DhanBroker: client (re)built for client_id=%s token_tail=...%s",
                (client_id[:4] + "***") if client_id else "***",
                access_token[-8:] if access_token else "",
            )
        except Exception:
            logger.exception("DhanBroker: failed to initialise DhanClient")
            self._client = None
            self._client_id = ""
            self._client_token = ""

    def _ensure_fresh_client(self) -> None:
        """Self-heal: if the access_token in the DB differs from the one the
        current ``_client`` was built with, transparently rebuild it.

        This is what makes "Save credentials in the UI" actually take effect
        for the very next order, without requiring every caller (scanners,
        exit engine, …) to know about ``reload_credentials()``. The DB read
        is served from a 30s in-process cache, and the UI save invalidates
        that cache, so the rotated token is picked up on the next call and
        is then sticky for the rest of the cache window.
        """
        try:
            from app.db.broker_credentials import get_dhan_credentials

            creds = get_dhan_credentials()
            db_client_id = creds.get("client_id") or ""
            db_token = creds.get("access_token") or ""
        except Exception:
            logger.debug("DhanBroker: credential refresh check failed", exc_info=True)
            return

        if not db_client_id or not db_token:
            return
        if db_token == self._client_token and db_client_id == self._client_id and self._client is not None:
            return

        logger.info(
            "DhanBroker: detected credential change "
            "(client_id %s→%s, token tail ...%s→...%s) — rebuilding client",
            self._client_id or "-", db_client_id,
            (self._client_token or "")[-8:] or "-",
            db_token[-8:],
        )
        self._authenticated = False
        self._init_client()

    def reload_credentials(self) -> None:
        """Force-refresh the Dhan client. Kept for explicit callers (env
        rotation, /update-credentials endpoint). Day-to-day rotation is
        handled transparently by ``_ensure_fresh_client``."""
        from app.db.broker_credentials import invalidate

        invalidate("dhan")
        self._authenticated = False
        self._init_client(fresh=True)

    def authenticate(self) -> bool:
        """Validate the configured access token by hitting fund limits."""
        if not self._client:
            self._init_client()
        if not self._client:
            return False
        try:
            data = self._client.get_fund_limits()
            self._authenticated = bool(data)
            if self._authenticated:
                logger.info(
                    "DhanBroker: authenticated (client_id=%s, available=%s)",
                    (self._client_id[:4] + "***") if self._client_id else "***",
                    data.get("availabelBalance") or data.get("availableBalance"),
                )
            else:
                logger.error("DhanBroker auth failed: empty fund limits response")
            return self._authenticated
        except Exception as exc:
            logger.error("DhanBroker auth failed: %s", exc)
            self._authenticated = False
            return False

    # ── Orders ───────────────────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResponse:
        if settings.paper_trading and not getattr(request, "force_live", False):
            return self._simulate_order(request)
        # Self-heal: pick up any token rotation done via the UI / env reload
        # before we burn the order on an expired token.
        self._ensure_fresh_client()
        if not self._client:
            return OrderResponse(
                status=OrderStatus.REJECTED, message="Dhan client not initialised"
            )

        # Global execution policy: all live orders must be MARKET + CARRYFORWARD.
        request.order_type = OrderType.MARKET
        request.product_type = ProductType.CARRYFORWARD
        request.price = 0.0
        request.trigger_price = 0.0

        exchange = request.exchange or "NFO"
        exchange_segment = _to_dhan_exchange_segment(exchange)
        security_id = self._resolve_security_id(request, exchange)
        if not security_id:
            msg = (
                f"Dhan securityId not found for {request.trading_symbol} "
                f"(strike={request.strike} {request.option_type} on {exchange}). "
                "Check the Dhan scrip master via /api/broker/dhan/resolve-symbol."
            )
            logger.error(msg)
            return OrderResponse(status=OrderStatus.REJECTED, message=msg)

        # ── BSE/BFO market-protection LIMIT ─────────────────────────
        # BSE does NOT accept pure MARKET orders for SENSEX/BANKEX
        # options. Dhan auto-converts MARKET → LIMIT using a tight
        # protection band (~5%) which causes legs to fail in fast
        # markets (e.g. CALL @ LTP 519 with broker-set limit 450
        # never fills). To make BFO behave like NFO ("just fill it")
        # we send an explicit LIMIT with a wide ±20% protection: BUY
        # caps at ask × 1.20, SELL floors at bid × 0.80. NFO stays
        # on true MARKET because NSE accepts it natively.
        order_type_str = self._map_order_type(OrderType.MARKET)
        limit_price = 0.0
        if exchange_segment == "BSE_FNO":
            try:
                ltp = self._client.get_ltp(exchange_segment, security_id) or 0.0
            except Exception:
                ltp = 0.0
            if ltp and ltp > 0:
                buffer = 0.20  # ±20% market-protection band
                if request.side.value.upper() == "BUY":
                    raw = ltp * (1.0 + buffer)
                else:
                    raw = ltp * (1.0 - buffer)
                # BFO option tick size is 0.05 — round to tick.
                tick = 0.05
                limit_price = max(tick, round(raw / tick) * tick)
                order_type_str = "LIMIT"
                logger.info(
                    "DHAN BFO MARKET-as-LIMIT: side=%s ltp=%.2f limit=%.2f (buffer=%.0f%%)",
                    request.side.value, ltp, limit_price, buffer * 100,
                )
            else:
                logger.warning(
                    "DHAN BFO LTP unavailable for %s — falling back to MARKET "
                    "(broker may auto-convert to tight LIMIT)",
                    request.trading_symbol,
                )

        # ── Freeze-qty slicing ───────────────────────────────────────
        # Exchanges cap units per single order (NIFTY 1800, BANKNIFTY 900,
        # SENSEX 1000, etc.). Above that, Dhan tags the order "iceberg"
        # but only the first slice executes and the rest is dropped. To
        # avoid silent under-fills, pre-slice into chunks ≤ freeze qty
        # and place each as an independent MARKET/LIMIT.
        freeze_qty = _resolve_freeze_qty(request)
        lot_size = int(getattr(request.instrument, "lot_size", 0) or 0)
        slice_qtys = _split_quantity(int(request.quantity), freeze_qty, lot_size)
        if len(slice_qtys) > 1:
            logger.info(
                "DHAN SLICING: %s %s total_qty=%d freeze=%d → %d slices %s",
                request.side.value, request.trading_symbol,
                int(request.quantity), freeze_qty, len(slice_qtys), slice_qtys,
            )

        wait_terminal = bool(settings.wait_for_terminal_order_status) or bool(getattr(request, "wait_for_terminal", False))
        primary_id: str = ""
        all_ids: list[str] = []
        total_filled: int = 0
        weighted_price_sum: float = 0.0
        last_status: OrderStatus = OrderStatus.OPEN
        last_message: str = ""

        for idx, slice_qty in enumerate(slice_qtys):
            resp = self._client.place_order(
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=request.side.value,
                quantity=int(slice_qty),
                order_type=order_type_str,
                product_type=self._map_product(ProductType.CARRYFORWARD),
                price=float(limit_price),
                trigger_price=0.0,
            )

            if not isinstance(resp, dict) or resp.get("status") != "success":
                err = ""
                if isinstance(resp, dict):
                    err = (
                        (resp.get("remarks") or {}).get("error_message")
                        if isinstance(resp.get("remarks"), dict)
                        else resp.get("remarks") or resp.get("message") or ""
                    )
                logger.error(
                    "DHAN ORDER REJECTED (slice %d/%d qty=%d): %s | %s | %s",
                    idx + 1, len(slice_qtys), slice_qty,
                    request.trading_symbol, security_id, err,
                )
                # If first slice fails, return rejection immediately —
                # nothing was placed, so nothing to clean up.
                if idx == 0:
                    return OrderResponse(
                        status=OrderStatus.REJECTED,
                        message=str(err) or "Dhan rejected order",
                    )
                # Later slice failed AFTER earlier slices already filled.
                # Per operator policy: DO NOT auto-rollback (no flatten
                # SELL). Keep the partial position, log loudly so it can
                # be reviewed, and report partial success up the stack.
                placed_qty = sum(slice_qtys[:idx])
                missing_qty = sum(slice_qtys[idx:])
                logger.error(
                    "DHAN PARTIAL FILL — KEEPING POSITION (no rollback): "
                    "%s %s placed=%d/%d missing=%d child_ids=%s reason=%s",
                    request.side.value, request.trading_symbol,
                    placed_qty, int(request.quantity), missing_qty,
                    ",".join(all_ids), err,
                )
                last_status = OrderStatus.OPEN  # Earlier slices ARE live
                last_message = (
                    f"PARTIAL FILL: placed {placed_qty}/{int(request.quantity)} "
                    f"qty in {idx} slice(s) [{','.join(all_ids)}]. "
                    f"Slice {idx + 1} ({slice_qty} qty) rejected: {err}. "
                    f"Position kept as-is (no auto-rollback)."
                )
                break

            data = resp.get("data") or {}
            order_id = str(data.get("orderId") or data.get("order_id") or "")
            if not order_id:
                if idx == 0:
                    return OrderResponse(
                        status=OrderStatus.REJECTED, message="No orderId returned"
                    )
                last_status = OrderStatus.REJECTED
                last_message = f"Slice {idx + 1} returned no orderId"
                break

            all_ids.append(order_id)
            if not primary_id:
                primary_id = order_id

            logger.info(
                "DHAN ORDER PLACED: %s %s qty=%s id=%s (slice %d/%d)",
                request.side.value, request.trading_symbol, slice_qty,
                order_id, idx + 1, len(slice_qtys),
            )

            if wait_terminal:
                info = self._wait_terminal_status(order_id, expected_qty=int(slice_qty))
                status_raw = (info.get("orderStatus") or info.get("status") or "").upper()
                last_status = self._map_status(status_raw)
                last_message = info.get("omsErrorDescription", "") or ""
                avg_price = float(
                    info.get("averageTradedPrice", 0)
                    or info.get("avg_traded_price", 0)
                    or 0
                )
                slice_filled = int(
                    info.get("filledQty", 0) or info.get("filled_qty", 0) or 0
                )
                total_filled += slice_filled
                weighted_price_sum += avg_price * slice_filled

        # Build aggregated response
        if wait_terminal:
            avg_price_combined = (
                weighted_price_sum / total_filled if total_filled > 0 else 0.0
            )
            return OrderResponse(
                order_id=primary_id,
                status=last_status,
                message=(
                    last_message
                    if len(slice_qtys) == 1
                    else f"{last_message} | child_ids={','.join(all_ids)}"
                ).strip(" |"),
                filled_price=avg_price_combined,
                filled_quantity=total_filled,
                timestamp=datetime.now(_IST),
            )
        # Fire-and-forget path
        return OrderResponse(
            order_id=primary_id,
            status=OrderStatus.OPEN,
            message=(
                "Order accepted by broker"
                if len(slice_qtys) == 1
                else f"Sliced into {len(slice_qtys)} child orders: {','.join(all_ids)}"
            ),
            filled_price=0.0,
            filled_quantity=0,
            timestamp=datetime.now(_IST),
        )

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_trigger: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> OrderResponse:
        if settings.paper_trading or not self._client:
            return OrderResponse(
                order_id=order_id, status=OrderStatus.COMPLETE,
                message="Simulated modify",
            )
        self._ensure_fresh_client()
        ok = self._client.modify_order(
            order_id=order_id,
            quantity=new_quantity,
            price=new_price,
            trigger_price=new_trigger,
        )
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.OPEN if ok else OrderStatus.REJECTED,
            message="Modified" if ok else "Modify failed",
        )

    def cancel_order(self, order_id: str) -> bool:
        if settings.paper_trading or not self._client:
            return True
        self._ensure_fresh_client()
        return self._client.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> OrderResponse:
        if settings.paper_trading or not self._client:
            return OrderResponse(order_id=order_id, status=OrderStatus.COMPLETE)
        self._ensure_fresh_client()
        info = self._client.get_order_by_id(order_id)
        if not info:
            return OrderResponse(order_id=order_id, status=OrderStatus.PENDING)
        return OrderResponse(
            order_id=order_id,
            status=self._map_status((info.get("orderStatus") or "").upper()),
            filled_price=float(info.get("averageTradedPrice", 0) or 0),
            filled_quantity=int(info.get("filledQty", 0) or 0),
            message=info.get("omsErrorDescription", "") or "",
        )

    def get_positions(self) -> list[Position]:
        if settings.paper_trading or not self._client:
            return []
        self._ensure_fresh_client()
        out: list[Position] = []
        for p in self._client.get_positions():
            qty = int(p.get("netQty", 0) or 0)
            if qty == 0:
                continue
            seg = p.get("exchangeSegment") or ""
            out.append(Position(
                trading_symbol=p.get("tradingSymbol", ""),
                exchange=_EXCHANGE_REVERSE.get(seg, seg),
                quantity=qty,
                average_price=float(p.get("buyAvg") or p.get("sellAvg") or 0),
                ltp=float(p.get("lastTradedPrice", 0) or 0),
                pnl=float(p.get("realizedProfit", 0) or 0)
                + float(p.get("unrealizedProfit", 0) or 0),
                product_type=p.get("productType", ""),
            ))
        return out

    def get_ltp(
        self, exchange: str, symbol: str, token: str
    ) -> Optional[float]:
        if not self._client:
            return None
        if not token:
            # Try resolving from the Angel symbol
            token = self._client.resolve_from_angel_symbol(symbol, exchange) or ""
        if not token:
            return None
        return self._client.get_ltp(_to_dhan_exchange_segment(exchange), token)

    def get_margin(self) -> dict:
        if settings.paper_trading:
            return {"available": settings.initial_capital}
        if not self._client:
            return {}
        self._ensure_fresh_client()
        data = self._client.get_fund_limits() or {}
        cash = float(
            data.get("availabelBalance")  # Dhan API field is misspelt
            or data.get("availableBalance")
            or data.get("withdrawableBalance")
            or 0
        )
        return {
            "net": cash,
            "availablecash": cash,
            "available_cash": cash,
            "raw": data,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _resolve_security_id(self, request: OrderRequest, exchange: str) -> str:
        if not self._client:
            return ""

        underlying = request.underlying or (
            request.instrument.option_symbol_prefix or request.instrument.symbol
            if request.instrument else None
        )
        expiry = request.expiry_date
        if isinstance(expiry, datetime):
            expiry = expiry.date()

        if underlying and expiry and request.strike and request.option_type:
            try:
                resolved = self._client.resolve_option_security_id(
                    underlying=underlying,
                    expiry=expiry,
                    strike=float(request.strike),
                    option_type=request.option_type,
                    exchange=exchange,
                )
                if resolved:
                    return resolved
                logger.warning(
                    "Dhan structured lookup miss: %s %s %s exp=%s — "
                    "falling back to symbol parse",
                    underlying, request.strike, request.option_type, expiry,
                )
            except Exception:
                logger.exception("Dhan structured lookup failed")

        # Fallback: parse the broker-agnostic trading symbol via the Dhan
        # scrip master. The previous implementation also returned
        # ``request.symbol_token`` directly when it was numeric, but that
        # token comes from AngelOne in some exit paths and IS NOT a valid
        # Dhan securityId — using it would route the order to a random
        # contract or get it rejected. So we always resolve through the
        # scrip master when structured lookup misses.
        return self._client.resolve_from_angel_symbol(
            request.trading_symbol, exchange
        ) or ""

    def resolve_symbol_debug(
        self,
        angel_symbol: str = "",
        exchange: str = "NFO",
        underlying: str = "",
        expiry_iso: str = "",
        strike: float = 0,
        option_type: str = "",
    ) -> dict:
        """Public diagnostic for the ``/api/broker/dhan/resolve-symbol`` route."""
        if not self._client:
            return {"ok": False, "error": "Dhan client not initialised"}
        try:
            resolved = ""
            if underlying and expiry_iso and strike and option_type:
                expiry = datetime.fromisoformat(expiry_iso).date()
                resolved = self._client.resolve_option_security_id(
                    underlying=underlying, expiry=expiry, strike=float(strike),
                    option_type=option_type, exchange=exchange,
                ) or ""
            if not resolved and angel_symbol:
                resolved = self._client.resolve_from_angel_symbol(
                    angel_symbol, exchange
                ) or ""
            return {
                "ok": bool(resolved),
                "angel_symbol": angel_symbol,
                "dhan_security_id": resolved,
                "exchange": exchange,
                "exchange_segment": _to_dhan_exchange_segment(exchange),
            }
        except Exception as exc:
            logger.exception("Dhan resolve_symbol_debug error")
            return {"ok": False, "error": str(exc)}

    def _wait_terminal_status(self, order_id: str, expected_qty: int) -> dict:
        last: dict = {}
        retries = max(1, int(settings.order_status_poll_retries or ORDER_POLL_RETRIES))
        delay = max(0.05, float(settings.order_status_poll_delay_seconds or ORDER_POLL_DELAY_SECONDS))
        for _ in range(retries):
            try:
                info = self._client.get_order_by_id(order_id) if self._client else {}
                if info:
                    last = info
                    status = (info.get("orderStatus") or "").upper()
                    if status in {"TRADED", "REJECTED", "CANCELLED", "EXPIRED"}:
                        return last
                    filled = int(info.get("filledQty", 0) or 0)
                    avg = float(info.get("averageTradedPrice", 0) or 0)
                    if expected_qty > 0 and filled >= expected_qty and avg > 0:
                        last["orderStatus"] = "TRADED"
                        return last
            except Exception:
                logger.debug("Dhan order poll error for %s", order_id)
            time.sleep(delay)
        return last

    def _simulate_order(self, request: OrderRequest) -> OrderResponse:
        import uuid
        return OrderResponse(
            order_id=f"PAPER-D-{uuid.uuid4().hex[:8].upper()}",
            status=OrderStatus.COMPLETE,
            message="Paper trade executed (Dhan)",
            filled_price=request.price if request.price > 0 else 0,
            filled_quantity=request.quantity,
            timestamp=datetime.now(_IST),
        )

    @staticmethod
    def _map_product(product: ProductType) -> str:
        return {
            ProductType.INTRADAY: "INTRADAY",
            ProductType.CARRYFORWARD: "MARGIN",
            ProductType.DELIVERY: "CNC",
        }.get(product, "INTRADAY")

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        return {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.SL: "STOP_LOSS",
            OrderType.SL_MARKET: "STOP_LOSS_MARKET",
        }.get(order_type, "MARKET")

    @staticmethod
    def _map_status(status_str: str) -> OrderStatus:
        s = (status_str or "").upper()
        mapping = {
            "TRADED": OrderStatus.COMPLETE,
            "TRANSIT": OrderStatus.PENDING,
            "PENDING": OrderStatus.OPEN,
            "TRIGGER PENDING": OrderStatus.OPEN,
            "PART_TRADED": OrderStatus.OPEN,
            "MODIFIED": OrderStatus.OPEN,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.CANCELLED,
        }
        return mapping.get(s, OrderStatus.PENDING)
