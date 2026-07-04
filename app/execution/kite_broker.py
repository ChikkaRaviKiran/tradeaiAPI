"""Kite (Zerodha) broker adapter — wraps KiteClient with BaseBroker interface.

Used when the AngelOne trading account is blocked. The orchestrator switches
to this broker via the `USE_KITE_FOR_ORDERS` setting; data feeds and
analytics continue to use AngelOne.

Key behaviour:
  - Translates Angel-format option symbols → Kite tradingsymbols on the fly
    using `KiteClient.resolve_from_angel_symbol`. Strategies can keep emitting
    Angel-style symbols.
  - Polls Kite order_history briefly after placement to capture average fill
    price (matches the AngelOne broker's behaviour and the
    `atl_live_execution_lessons` rule of waiting for terminal status).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pytz

from app.core.config import settings
from app.data.kite_client import KiteClient
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


class KiteBroker(BaseBroker):
    """Kite Connect (Zerodha) broker implementation."""

    def __init__(self, client: Optional[KiteClient] = None) -> None:
        # If ``client`` is supplied (multi-account path) it is used verbatim
        # so a per-account KiteClient can be injected by the broker factory.
        # Otherwise the client is built from global settings (legacy path).
        self._client: Optional[KiteClient] = client
        self._authenticated = False
        if client is None:
            self._init_client()

    @property
    def name(self) -> str:
        return "Kite"

    @property
    def client(self) -> Optional[KiteClient]:
        return self._client

    def _init_client(self) -> None:
        if not settings.kite_api_key:
            logger.warning("KiteBroker: KITE_API_KEY not configured")
            return
        try:
            self._client = KiteClient(
                api_key=settings.kite_api_key,
                access_token=settings.kite_access_token,
                proxy_url=settings.kite_proxy_url,
                market_protection_pct=float(settings.kite_market_protection_pct or 0),
            )
        except Exception:
            logger.exception("KiteBroker: failed to initialise KiteClient")
            self._client = None

    def reload_credentials(self) -> None:
        """Re-create the Kite client after settings are updated at runtime."""
        self._authenticated = False
        self._init_client()

    def authenticate(self) -> bool:
        """Validate the configured access token by hitting Kite profile."""
        if not self._client:
            self._init_client()
        if not self._client:
            return False
        if not settings.kite_access_token:
            logger.warning(
                "KiteBroker: no access token set. Login at /api/auth/kite/login-url first."
            )
            return False
        try:
            self._client.set_access_token(settings.kite_access_token)
            profile = self._client.get_profile()
            self._authenticated = bool(profile and profile.get("user_id"))
            if self._authenticated:
                logger.info(
                    "KiteBroker: authenticated as %s (%s)",
                    profile.get("user_name", ""), profile.get("user_id", ""),
                )
            return self._authenticated
        except Exception as exc:
            logger.error("KiteBroker auth failed: %s", exc)
            self._authenticated = False
            return False

    # ── Orders ────────────────────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResponse:
        if settings.paper_trading and not getattr(request, "force_live", False):
            return self._simulate_order(request)
        if not self._client:
            return OrderResponse(status=OrderStatus.REJECTED, message="Kite client not initialised")

        # Global execution policy: all live orders must be MARKET + CARRYFORWARD.
        request.order_type = OrderType.MARKET
        request.product_type = ProductType.CARRYFORWARD
        request.price = 0.0
        request.trigger_price = 0.0

        exchange = request.exchange or "NFO"
        kite_symbol = self._resolve_symbol(request, exchange)
        if not kite_symbol:
            msg = (
                f"Kite tradingsymbol not found for {request.trading_symbol} "
                f"(strike={request.strike} {request.option_type} on {exchange}). "
                "Check the Kite instrument master via /api/broker/kite/resolve-symbol."
            )
            logger.error(msg)
            return OrderResponse(status=OrderStatus.REJECTED, message=msg)

        try:
            order_id = self._client.place_order(
                tradingsymbol=kite_symbol,
                exchange=exchange,
                transaction_type=request.side.value,
                quantity=int(request.quantity),
                order_type=self._map_order_type(OrderType.MARKET),
                product=self._map_product(ProductType.CARRYFORWARD),
                price=float(request.price or 0),
                trigger_price=float(request.trigger_price or 0),
            )
            if not order_id:
                return OrderResponse(status=OrderStatus.REJECTED, message="No order_id returned")

            logger.info(
                "KITE ORDER PLACED: %s %s qty=%s id=%s",
                request.side.value, kite_symbol, request.quantity, order_id,
            )
            wait_terminal = bool(settings.wait_for_terminal_order_status) or bool(getattr(request, "wait_for_terminal", False))
            if not wait_terminal:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.OPEN,
                    message="Order accepted by broker",
                    filled_price=0.0,
                    filled_quantity=0,
                    timestamp=datetime.now(_IST),
                )

            info = self._wait_terminal_status(order_id, expected_qty=int(request.quantity))
            status_raw = (info.get("status") or "").upper()
            avg_price = float(info.get("average_price", 0) or 0)
            filled_qty = int(info.get("filled_quantity", 0) or 0)
            mapped = self._map_status(status_raw)
            if mapped == OrderStatus.REJECTED:
                msg = info.get("status_message", "") or "rejected by broker"
                logger.error("KITE ORDER REJECTED: %s | %s | %s", kite_symbol, order_id, msg)
            return OrderResponse(
                order_id=order_id,
                status=mapped,
                message=info.get("status_message", ""),
                filled_price=avg_price,
                filled_quantity=filled_qty,
                timestamp=datetime.now(_IST),
            )
        except Exception as exc:
            logger.exception("Kite order placement error")
            return OrderResponse(status=OrderStatus.REJECTED, message=str(exc))

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_trigger: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> OrderResponse:
        if settings.paper_trading or not self._client:
            return OrderResponse(order_id=order_id, status=OrderStatus.COMPLETE, message="Simulated modify")
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
        try:
            history = self._client.get_order_history(order_id)
            if history:
                latest = history[-1]
                return OrderResponse(
                    order_id=order_id,
                    status=self._map_status((latest.get("status") or "").upper()),
                    filled_price=float(latest.get("average_price", 0) or 0),
                    filled_quantity=int(latest.get("filled_quantity", 0) or 0),
                    message=latest.get("status_message", ""),
                )
        except Exception:
            logger.exception("Kite order status query failed for %s", order_id)
        return OrderResponse(order_id=order_id, status=OrderStatus.PENDING)

    def get_positions(self) -> list[Position]:
        if settings.paper_trading or not self._client:
            return []
        try:
            raw = self._client.get_positions() or {}
            net = raw.get("net", []) or []
            out: list[Position] = []
            for p in net:
                qty = int(p.get("quantity", 0) or 0)
                if qty == 0:
                    continue
                out.append(Position(
                    trading_symbol=p.get("tradingsymbol", ""),
                    exchange=p.get("exchange", ""),
                    quantity=qty,
                    average_price=float(p.get("average_price", 0) or 0),
                    ltp=float(p.get("last_price", 0) or 0),
                    pnl=float(p.get("pnl", 0) or 0),
                    product_type=p.get("product", ""),
                ))
            return out
        except Exception:
            logger.exception("Kite positions query failed")
            return []

    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        if not self._client:
            return None
        try:
            kite_symbol = self._client.resolve_from_angel_symbol(symbol, exchange) or symbol
            return self._client.get_ltp(exchange, kite_symbol)
        except Exception:
            logger.exception("Kite LTP fetch failed for %s", symbol)
            return None

    def get_margin(self) -> dict:
        if settings.paper_trading:
            return {"available": settings.initial_capital}
        if not self._client:
            return {}
        try:
            data = self._client.get_margins() or {}
            # Normalise to expose `availablecash` / `net` (matches AngelOne shape
            # used by lot_sizer.get_available_funds and orchestrator margin check)
            equity = data.get("equity", {}) if isinstance(data, dict) else {}
            available = equity.get("available", {}) if isinstance(equity, dict) else {}
            cash = float(available.get("live_balance", 0) or available.get("cash", 0) or 0)
            return {
                "net": cash,
                "availablecash": cash,
                "available_cash": cash,
                "raw": data,
            }
        except Exception:
            logger.exception("Kite margins query failed")
            return {}

    # ── Internal helpers ─────────────────────────────────────────────

    def _resolve_symbol(self, request: OrderRequest, exchange: str) -> str:
        """Resolve Kite tradingsymbol with structured fields preferred.

        Order of preference:
          1. Structured fields on `OrderRequest` (underlying / expiry_date /
             strike / option_type) — this never depends on string formats.
          2. Parse the AngelOne-format `trading_symbol` and translate.
          3. Direct match (Kite/Angel symbol equality, rare but cheap).
        """
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
                resolved = self._client.resolve_option_symbol(
                    name=underlying,
                    expiry=expiry,
                    strike=float(request.strike),
                    option_type=request.option_type,
                    exchange=exchange,
                )
                if resolved:
                    return resolved
                logger.warning(
                    "Kite structured lookup miss: %s %s %s exp=%s — falling back to symbol parse",
                    underlying, request.strike, request.option_type, expiry,
                )
            except Exception:
                logger.exception("Kite structured lookup failed")

        return self._client.resolve_from_angel_symbol(request.trading_symbol, exchange)

    def resolve_symbol_debug(
        self,
        angel_symbol: str = "",
        exchange: str = "NFO",
        underlying: str = "",
        expiry_iso: str = "",
        strike: float = 0,
        option_type: str = "",
    ) -> dict:
        """Public diagnostic helper used by `/api/broker/kite/resolve-symbol`.

        Returns the resolved Kite tradingsymbol and instrument_token for
        manual verification against the Kite instrument master.
        """
        if not self._client:
            return {"ok": False, "error": "Kite client not initialised"}
        try:
            resolved = ""
            if underlying and expiry_iso and strike and option_type:
                expiry = datetime.fromisoformat(expiry_iso).date()
                resolved = self._client.resolve_option_symbol(
                    name=underlying, expiry=expiry, strike=float(strike),
                    option_type=option_type, exchange=exchange,
                ) or ""
            if not resolved and angel_symbol:
                resolved = self._client.resolve_from_angel_symbol(angel_symbol, exchange)
            token = self._client.get_instrument_token(resolved, exchange) if resolved else None
            return {
                "ok": bool(resolved),
                "angel_symbol": angel_symbol,
                "kite_tradingsymbol": resolved,
                "kite_instrument_token": token,
                "exchange": exchange,
            }
        except Exception as exc:
            logger.exception("Kite resolve_symbol_debug error")
            return {"ok": False, "error": str(exc)}

    def _wait_terminal_status(self, order_id: str, expected_qty: int) -> dict:
        """Poll Kite order_history until COMPLETE/REJECTED/CANCELLED or timeout."""
        last: dict = {}
        retries = max(1, int(settings.order_status_poll_retries or ORDER_POLL_RETRIES))
        delay = max(0.05, float(settings.order_status_poll_delay_seconds or ORDER_POLL_DELAY_SECONDS))
        for _ in range(retries):
            try:
                history = self._client.get_order_history(order_id) if self._client else []
                if history:
                    last = history[-1]
                    status = (last.get("status") or "").upper()
                    if status in {"COMPLETE", "REJECTED", "CANCELLED"}:
                        return last
                    filled = int(last.get("filled_quantity", 0) or 0)
                    avg = float(last.get("average_price", 0) or 0)
                    if filled >= expected_qty and avg > 0:
                        last["status"] = "COMPLETE"
                        return last
            except Exception:
                logger.debug("Kite order poll error for %s", order_id)
            time.sleep(delay)
        return last

    def _simulate_order(self, request: OrderRequest) -> OrderResponse:
        import uuid
        return OrderResponse(
            order_id=f"PAPER-K-{uuid.uuid4().hex[:8].upper()}",
            status=OrderStatus.COMPLETE,
            message="Paper trade executed (Kite)",
            filled_price=request.price if request.price > 0 else 0,
            filled_quantity=request.quantity,
            timestamp=datetime.now(_IST),
        )

    @staticmethod
    def _map_product(product: ProductType) -> str:
        return {
            ProductType.INTRADAY: "MIS",
            ProductType.CARRYFORWARD: "NRML",
            ProductType.DELIVERY: "CNC",
        }.get(product, "MIS")

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        return {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.SL: "SL",
            OrderType.SL_MARKET: "SL-M",
        }.get(order_type, "MARKET")

    @staticmethod
    def _map_status(status_str: str) -> OrderStatus:
        s = (status_str or "").upper()
        mapping = {
            "COMPLETE": OrderStatus.COMPLETE,
            "OPEN": OrderStatus.OPEN,
            "TRIGGER PENDING": OrderStatus.OPEN,
            "VALIDATION PENDING": OrderStatus.PENDING,
            "PUT ORDER REQ RECEIVED": OrderStatus.PENDING,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
        }
        return mapping.get(s, OrderStatus.PENDING)
