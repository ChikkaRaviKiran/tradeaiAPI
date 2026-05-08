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


class DhanBroker(BaseBroker):
    """DhanHQ broker implementation."""

    def __init__(self) -> None:
        self._client: Optional[DhanClient] = None
        self._authenticated = False
        self._init_client()

    @property
    def name(self) -> str:
        return "Dhan"

    @property
    def client(self) -> Optional[DhanClient]:
        return self._client

    def _init_client(self) -> None:
        if not settings.dhan_client_id or not settings.dhan_access_token:
            logger.warning(
                "DhanBroker: DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not configured"
            )
            return
        try:
            self._client = DhanClient(
                client_id=settings.dhan_client_id,
                access_token=settings.dhan_access_token,
            )
        except Exception:
            logger.exception("DhanBroker: failed to initialise DhanClient")
            self._client = None

    def reload_credentials(self) -> None:
        """Re-create the Dhan client after settings change at runtime."""
        self._authenticated = False
        self._init_client()

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
                    settings.dhan_client_id[:4] + "***",
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
        if settings.paper_trading:
            return self._simulate_order(request)
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

        resp = self._client.place_order(
            security_id=security_id,
            exchange_segment=exchange_segment,
            transaction_type=request.side.value,
            quantity=int(request.quantity),
            order_type=self._map_order_type(request.order_type),
            product_type=self._map_product(request.product_type),
            price=float(request.price or 0),
            trigger_price=float(request.trigger_price or 0),
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
                "DHAN ORDER REJECTED: %s | %s | %s",
                request.trading_symbol, security_id, err,
            )
            return OrderResponse(
                status=OrderStatus.REJECTED, message=str(err) or "Dhan rejected order"
            )

        data = resp.get("data") or {}
        order_id = str(data.get("orderId") or data.get("order_id") or "")
        if not order_id:
            return OrderResponse(
                status=OrderStatus.REJECTED, message="No orderId returned"
            )

        logger.info(
            "DHAN ORDER PLACED: %s %s qty=%s id=%s",
            request.side.value, request.trading_symbol, request.quantity, order_id,
        )
        if not settings.wait_for_terminal_order_status:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.OPEN,
                message="Order accepted by broker",
                filled_price=0.0,
                filled_quantity=0,
                timestamp=datetime.now(_IST),
            )
        info = self._wait_terminal_status(
            order_id, expected_qty=int(request.quantity)
        )
        status_raw = (info.get("orderStatus") or info.get("status") or "").upper()
        avg_price = float(
            info.get("averageTradedPrice", 0)
            or info.get("avg_traded_price", 0)
            or 0
        )
        filled_qty = int(
            info.get("filledQty", 0) or info.get("filled_qty", 0) or 0
        )
        return OrderResponse(
            order_id=order_id,
            status=self._map_status(status_raw),
            message=info.get("omsErrorDescription", "") or "",
            filled_price=avg_price,
            filled_quantity=filled_qty,
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
        return self._client.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> OrderResponse:
        if settings.paper_trading or not self._client:
            return OrderResponse(order_id=order_id, status=OrderStatus.COMPLETE)
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

        # Fallback: parse Angel symbol
        if request.symbol_token and request.symbol_token.isdigit():
            return request.symbol_token  # caller may have passed Dhan securityId
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
