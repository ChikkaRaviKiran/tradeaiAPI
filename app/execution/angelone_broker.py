"""AngelOne broker adapter — wraps AngelOneClient with BaseBroker interface.

SRS Module 3.8: Trade Execution Engine — AngelOne implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pytz

from app.core.config import settings
from app.core.instruments import InstrumentConfig
from app.data.angelone_client import AngelOneClient
from app.execution.broker_base import (
    BaseBroker,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
)

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")


class AngelOneBroker(BaseBroker):
    """AngelOne SmartAPI broker implementation.

    For paper trading, orders are simulated and not sent to exchange.
    For live trading (paper_trading=False), orders hit real exchange.
    """

    def __init__(self) -> None:
        self.client = AngelOneClient()
        self._authenticated = False

    @property
    def name(self) -> str:
        return "AngelOne"

    def authenticate(self) -> bool:
        self._authenticated = self.client.authenticate()
        return self._authenticated

    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place order via SmartAPI."""
        if settings.paper_trading:
            return self._simulate_order(request)

        self.client.ensure_authenticated()
        try:
            params = {
                "variety": "NORMAL",
                "tradingsymbol": request.trading_symbol,
                "symboltoken": request.symbol_token,
                "transactiontype": request.side.value,
                "exchange": request.exchange,
                "ordertype": request.order_type.value,
                "producttype": self._map_product(request.product_type),
                "duration": "DAY",
                "quantity": str(request.quantity),
                "price": str(request.price) if request.order_type == OrderType.LIMIT else "0",
                "triggerprice": str(request.trigger_price) if request.trigger_price > 0 else "0",
            }
            resp = self.client._smart_api.placeOrder(params)
            if resp and resp.get("status"):
                order_id = resp.get("data", {}).get("orderid", "")
                logger.info("Order placed: %s | %s", order_id, request.trading_symbol)
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.OPEN,
                    message="Order placed successfully",
                    timestamp=datetime.now(_IST),
                )
            else:
                msg = resp.get("message", "Unknown error") if resp else "No response"
                logger.error("Order failed: %s | %s", msg, request.trading_symbol)
                return OrderResponse(status=OrderStatus.REJECTED, message=msg)
        except Exception as e:
            logger.exception("Order placement error")
            return OrderResponse(status=OrderStatus.REJECTED, message=str(e))

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_trigger: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> OrderResponse:
        if settings.paper_trading:
            return OrderResponse(order_id=order_id, status=OrderStatus.COMPLETE, message="Simulated modify")

        self.client.ensure_authenticated()
        try:
            params = {"variety": "NORMAL", "orderid": order_id}
            if new_price is not None:
                params["price"] = str(new_price)
            if new_trigger is not None:
                params["triggerprice"] = str(new_trigger)
            if new_quantity is not None:
                params["quantity"] = str(new_quantity)
            resp = self.client._smart_api.modifyOrder(params)
            if resp and resp.get("status"):
                return OrderResponse(order_id=order_id, status=OrderStatus.OPEN, message="Modified")
            return OrderResponse(order_id=order_id, status=OrderStatus.REJECTED, message=str(resp))
        except Exception as e:
            logger.exception("Order modify error")
            return OrderResponse(order_id=order_id, status=OrderStatus.REJECTED, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if settings.paper_trading:
            return True
        self.client.ensure_authenticated()
        try:
            resp = self.client._smart_api.cancelOrder(order_id, "NORMAL")
            return bool(resp and resp.get("status"))
        except Exception:
            logger.exception("Order cancel error")
            return False

    def get_order_status(self, order_id: str) -> OrderResponse:
        if settings.paper_trading:
            return OrderResponse(order_id=order_id, status=OrderStatus.COMPLETE)
        self.client.ensure_authenticated()
        try:
            resp = self.client._smart_api.orderBook()
            if resp and resp.get("data"):
                for order in resp["data"]:
                    if order.get("orderid") == order_id:
                        return OrderResponse(
                            order_id=order_id,
                            status=self._map_status(order.get("orderstatus", "")),
                            filled_price=float(order.get("averageprice", 0)),
                            filled_quantity=int(order.get("filledshares", 0)),
                        )
        except Exception:
            logger.exception("Order status query error")
        return OrderResponse(order_id=order_id, status=OrderStatus.PENDING)

    def get_positions(self) -> list[Position]:
        if settings.paper_trading:
            return []
        self.client.ensure_authenticated()
        try:
            resp = self.client._smart_api.position()
            positions = []
            if resp and resp.get("data"):
                for p in resp["data"]:
                    positions.append(Position(
                        trading_symbol=p.get("tradingsymbol", ""),
                        exchange=p.get("exchange", ""),
                        quantity=int(p.get("netqty", 0)),
                        average_price=float(p.get("averageprice", 0)),
                        ltp=float(p.get("ltp", 0)),
                        pnl=float(p.get("pnl", 0)),
                        product_type=p.get("producttype", ""),
                    ))
            return positions
        except Exception:
            logger.exception("Position query error")
            return []

    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        return self.client.get_ltp(exchange, symbol, token)

    def get_margin(self) -> dict:
        if settings.paper_trading:
            return {"available": settings.initial_capital}
        self.client.ensure_authenticated()
        try:
            resp = self.client._smart_api.rmsLimit()
            if resp and resp.get("data"):
                return resp["data"]
        except Exception:
            logger.exception("Margin query error")
        return {}

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _simulate_order(self, request: OrderRequest) -> OrderResponse:
        """Simulate order for paper trading."""
        import uuid
        return OrderResponse(
            order_id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            status=OrderStatus.COMPLETE,
            message="Paper trade executed",
            filled_price=request.price if request.price > 0 else 0,
            filled_quantity=request.quantity,
            timestamp=datetime.now(_IST),
        )

    @staticmethod
    def _map_product(product: ProductType) -> str:
        return {
            ProductType.INTRADAY: "INTRADAY",
            ProductType.CARRYFORWARD: "CARRYFORWARD",
            ProductType.DELIVERY: "DELIVERY",
        }.get(product, "INTRADAY")

    @staticmethod
    def _map_status(status_str: str) -> OrderStatus:
        mapping = {
            "complete": OrderStatus.COMPLETE,
            "open": OrderStatus.OPEN,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(status_str.lower(), OrderStatus.PENDING)
