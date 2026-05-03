"""Dynamic lot sizer for option-buying strategies.

Computes how many lots can be afforded given current broker margin /
account funds. For BUYING options the required capital is simply
``premium × lot_size`` per lot (no SPAN/exposure margin), so we can
compute lots directly from premium and available cash.

For SELLING options (margin-intensive), this module attempts the
AngelOne SmartAPI ``getMarginApi`` (POST
``/rest/secure/angelbroking/margin/v1/batch``) which returns SPAN +
exposure margin per leg. When that endpoint is unavailable we fall
back to a heuristic and let the caller decide.

Public helpers
--------------
- ``get_available_funds(broker)``: read net/available cash from rmsLimit.
- ``compute_buy_option_lots(...)``: max lots for a long option entry.
- ``estimate_sell_margin_per_lot(...)``: try broker margin-calculator API.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_available_funds(broker) -> float:
    """Return cash available for new trades from the broker (₹).

    For paper trading returns the configured initial_capital. For live
    trading returns the AngelOne ``rmsLimit`` ``net``/``availablecash``
    value, whichever is present first.
    """
    try:
        data = broker.get_margin() or {}
    except Exception:
        logger.exception("get_available_funds: broker.get_margin failed")
        return 0.0

    if not isinstance(data, dict):
        return 0.0

    # Paper-trading shape: {"available": <capital>}
    if "available" in data:
        try:
            return float(data["available"])
        except (TypeError, ValueError):
            return 0.0

    for key in ("availablecash", "available_cash", "net", "availableintradaypayin"):
        val = data.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def compute_buy_option_lots(
    broker,
    premium: float,
    lot_size: int,
    *,
    max_funds_cap: float = 0.0,
    buffer_pct: float = 5.0,
    max_lots_cap: int = 20,
    min_lots: int = 1,
) -> tuple[int, dict]:
    """Compute maximum affordable lots for a long-option entry.

    Parameters
    ----------
    broker : BaseBroker
        Used to fetch available funds via ``get_margin``.
    premium : float
        Current option premium (₹ per unit).
    lot_size : int
        Contract lot size (e.g. 75 for NIFTY).
    max_funds_cap : float
        If > 0, cap usable funds to this amount (e.g. ₹1,50,000).
        ``0`` means use full available cash.
    buffer_pct : float
        Safety buffer (% of usable funds reserved for slippage / fees).
    max_lots_cap : int
        Hard ceiling on lots regardless of funds.
    min_lots : int
        Minimum lots to return when funds are sufficient (default 1).

    Returns
    -------
    (lots, details) where ``details`` includes the math for logging.
    """
    if premium <= 0 or lot_size <= 0:
        return 0, {"reason": "invalid_premium_or_lot_size", "premium": premium, "lot_size": lot_size}

    available = get_available_funds(broker)
    usable = available
    if max_funds_cap and max_funds_cap > 0:
        usable = min(usable, float(max_funds_cap))

    # Reserve safety buffer for slippage + fees + margin haircut
    buf = max(0.0, float(buffer_pct)) / 100.0
    deployable = usable * (1.0 - buf)

    cost_per_lot = premium * lot_size
    if cost_per_lot <= 0:
        return 0, {"reason": "zero_cost_per_lot"}

    raw_lots = int(math.floor(deployable / cost_per_lot))
    lots = max(0, min(raw_lots, int(max_lots_cap)))

    if lots < min_lots:
        # Funds too low to honour the requested minimum
        return 0, {
            "reason": "insufficient_funds",
            "available": round(available, 2),
            "usable": round(usable, 2),
            "deployable": round(deployable, 2),
            "cost_per_lot": round(cost_per_lot, 2),
            "max_funds_cap": max_funds_cap,
            "buffer_pct": buffer_pct,
        }

    return lots, {
        "available": round(available, 2),
        "usable": round(usable, 2),
        "deployable": round(deployable, 2),
        "cost_per_lot": round(cost_per_lot, 2),
        "premium": round(premium, 2),
        "lot_size": lot_size,
        "max_funds_cap": max_funds_cap,
        "buffer_pct": buffer_pct,
        "max_lots_cap": max_lots_cap,
        "lots": lots,
        "deployed_capital": round(lots * cost_per_lot, 2),
    }


async def estimate_sell_margin_per_lot(
    client,
    *,
    exchange: str,
    trading_symbol: str,
    symbol_token: str,
    lot_size: int,
    premium: float,
) -> Optional[float]:
    """Estimate per-lot margin for a SHORT option leg via SmartAPI batch
    margin calculator. Returns ``None`` if the endpoint is unavailable.

    See https://smartapi.angelbroking.com/docs/Margin — endpoint
    ``/rest/secure/angelbroking/margin/v1/batch``.
    """
    if settings.paper_trading:
        return None
    if not client or lot_size <= 0:
        return None

    smart_api = getattr(client, "_smart_api", None)
    if smart_api is None:
        return None

    # SmartAPI exposes this as ``getMarginApi`` in newer SDK versions.
    fn = getattr(smart_api, "getMarginApi", None) or getattr(smart_api, "marginCalculator", None)
    if fn is None:
        return None

    payload = {
        "positions": [
            {
                "exchange": exchange or "NFO",
                "qty": str(lot_size),
                "price": str(premium),
                "productType": "INTRADAY",
                "token": str(symbol_token),
                "tradeType": "SELL",
            }
        ]
    }
    try:
        client.ensure_authenticated()
        resp = await asyncio.to_thread(fn, payload)
    except Exception:
        logger.exception("getMarginApi call failed for %s", trading_symbol)
        return None

    if not isinstance(resp, dict):
        return None

    data = resp.get("data") or {}
    # Common keys returned by AngelOne batch margin response
    for key in ("totalMarginRequired", "margin_required", "margin", "totalMargin"):
        val = data.get(key) if isinstance(data, dict) else None
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None
