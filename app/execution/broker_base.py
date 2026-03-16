"""Abstract broker interface — SRS Module 3.8.

All brokers (AngelOne, Zerodha, etc.) implement this interface for:
  - Order placement (market, limit, bracket)
  - Order modification / cancellation
  - Position and order status queries
  - Portfolio / margin queries
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from app.core.instruments import InstrumentConfig


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_MARKET = "SL-M"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProductType(str, Enum):
    INTRADAY = "INTRADAY"    # MIS
    CARRYFORWARD = "CARRYFORWARD"  # NRML
    DELIVERY = "DELIVERY"    # CNC


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OrderRequest:
    """Order placement request."""
    instrument: InstrumentConfig
    trading_symbol: str
    symbol_token: str
    exchange: str = "NFO"
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    product_type: ProductType = ProductType.INTRADAY
    quantity: int = 0
    price: float = 0.0            # For LIMIT orders
    trigger_price: float = 0.0    # For SL / SL-M orders


@dataclass
class OrderResponse:
    """Broker response after placing an order."""
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    message: str = ""
    filled_price: float = 0.0
    filled_quantity: int = 0
    timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Current open position from broker."""
    trading_symbol: str = ""
    exchange: str = ""
    quantity: int = 0
    average_price: float = 0.0
    ltp: float = 0.0
    pnl: float = 0.0
    product_type: str = ""


class BaseBroker(ABC):
    """Interface all broker implementations must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name for logging."""
        ...

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate / create session. Returns True on success."""
        ...

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place an order. Returns OrderResponse with order_id and status."""
        ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_trigger: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> OrderResponse:
        """Modify a pending order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get current status of an order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        """Fetch last traded price for a contract."""
        ...

    @abstractmethod
    def get_margin(self) -> dict:
        """Get available margin / funds."""
        ...
