"""Kite Connect client wrapper for Zerodha order execution.

Used when the AngelOne trading account is blocked. Data continues to flow
from AngelOne (quotes, candles, option chain); only order placement and
exits are routed through Kite.

Mirrors the design used in the OptionSelling project — instrument cache is
refreshed once per IST trading day and queried by structured fields
(name + expiry + strike + option_type) so we don't depend on broker-specific
symbol-string formatting.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pytz

try:
    from kiteconnect import KiteConnect
except Exception:  # pragma: no cover — soft import so app boots without kiteconnect installed
    KiteConnect = None  # type: ignore

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

EXCHANGE_NFO = "NFO"
EXCHANGE_BFO = "BFO"


class KiteClient:
    """Thin wrapper around KiteConnect for order placement.

    Threading note: KiteConnect itself is sync. Wrap calls in
    `asyncio.to_thread(...)` from the orchestrator's async context.
    """

    def __init__(
        self,
        api_key: str,
        access_token: str = "",
        account_name: str = "",
        proxy_url: str = "",
        market_protection_pct: float = 0.0,
    ):
        if KiteConnect is None:
            raise RuntimeError(
                "kiteconnect is not installed. Run `pip install kiteconnect==5.1.0`."
            )
        self.api_key = api_key
        self.account_name = account_name or "kite"
        self.proxy_url = proxy_url or ""
        kite_kwargs: dict = {"api_key": api_key}
        if self.proxy_url:
            kite_kwargs["proxies"] = {"http": self.proxy_url, "https": self.proxy_url}
            logger.info(
                "KiteClient[%s] routing via proxy %s",
                self.account_name,
                self._safe_proxy_repr(self.proxy_url),
            )
        self._kite = KiteConnect(**kite_kwargs)
        if access_token:
            self._kite.set_access_token(access_token)
        self._instrument_cache_date = None
        self._instruments_by_exchange: dict[str, list[dict]] = {}
        self.market_protection_pct = max(0.0, float(market_protection_pct or 0.0))

    @staticmethod
    def _safe_proxy_repr(url: str) -> str:
        """Mask credentials in proxy URL for logs."""
        try:
            from urllib.parse import urlparse, urlunparse
            p = urlparse(url)
            if p.username:
                netloc = f"***:***@{p.hostname}"
                if p.port:
                    netloc += f":{p.port}"
                return urlunparse((p.scheme, netloc, p.path, "", "", ""))
        except Exception:
            pass
        return url

    @property
    def kite(self):
        return self._kite

    def set_access_token(self, token: str) -> None:
        self._kite.set_access_token(token)

    def generate_session(self, request_token: str, api_secret: str) -> dict:
        return self._kite.generate_session(request_token, api_secret=api_secret)

    # ── Orders ────────────────────────────────────────────────────────

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        product: str = "NRML",
        variety: str = "regular",
        price: float = 0.0,
        trigger_price: float = 0.0,
    ) -> str:
        kwargs: dict = {
            "variety": variety,
            "exchange": exchange,
            "tradingsymbol": tradingsymbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "product": product,
        }
        if order_type == "MARKET" and self.market_protection_pct > 0:
            kwargs["market_protection"] = self.market_protection_pct
        if order_type in ("LIMIT", "SL"):
            kwargs["price"] = price
        if order_type in ("SL", "SL-M"):
            kwargs["trigger_price"] = trigger_price
        order_id = self._kite.place_order(**kwargs)
        return str(order_id)

    def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        try:
            self._kite.cancel_order(variety=variety, order_id=order_id)
            return True
        except Exception:
            logger.exception("Kite cancel_order failed for %s", order_id)
            return False

    def modify_order(
        self,
        order_id: str,
        variety: str = "regular",
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        kwargs: dict = {"variety": variety, "order_id": order_id}
        if quantity is not None:
            kwargs["quantity"] = quantity
        if price is not None:
            kwargs["price"] = price
        if trigger_price is not None:
            kwargs["trigger_price"] = trigger_price
        if order_type is not None:
            kwargs["order_type"] = order_type
        try:
            self._kite.modify_order(**kwargs)
            return True
        except Exception:
            logger.exception("Kite modify_order failed for %s", order_id)
            return False

    def get_order_history(self, order_id: str) -> list[dict]:
        return self._kite.order_history(order_id)

    def get_orders(self) -> list[dict]:
        return self._kite.orders()

    def get_positions(self) -> dict:
        return self._kite.positions()

    def get_ltp(self, exchange: str, tradingsymbol: str) -> float:
        key = f"{exchange}:{tradingsymbol}"
        data = self._kite.ltp(key)
        if data and key in data:
            return float(data[key].get("last_price", 0))
        return 0.0

    def get_profile(self) -> dict:
        return self._kite.profile()

    def get_margins(self) -> dict:
        return self._kite.margins()

    # ── Instruments ───────────────────────────────────────────────────

    def _load_instruments(self, exchange: str) -> list[dict]:
        today = datetime.now(_IST).date()
        if self._instrument_cache_date != today:
            self._instruments_by_exchange = {}
            self._instrument_cache_date = today

        if exchange in self._instruments_by_exchange:
            return self._instruments_by_exchange[exchange]

        instruments = self._kite.instruments(exchange) or []
        self._instruments_by_exchange[exchange] = instruments
        logger.info("Loaded Kite %s instruments: %d", exchange, len(instruments))
        return instruments

    def refresh_instruments(self, exchange: str, force: bool = False) -> list[dict]:
        if force:
            self._instruments_by_exchange.pop(exchange, None)
        return self._load_instruments(exchange)

    def resolve_option_symbol(
        self,
        name: str,
        expiry,
        strike: float,
        option_type: str,
        exchange: str,
    ) -> Optional[str]:
        """Resolve exact Kite tradingsymbol for an option contract.

        Args:
            name: index / underlying name (e.g. "NIFTY", "SENSEX", "BANKNIFTY").
            expiry: `datetime.date` of contract expiry.
            strike: strike price.
            option_type: "CE" or "PE".
            exchange: "NFO" or "BFO".
        """
        instruments = self._load_instruments(exchange)
        target_strike = float(strike)
        name_u = name.upper()

        for inst in instruments:
            if (inst.get("name") or "").upper() != name_u:
                continue
            if inst.get("instrument_type") != option_type:
                continue
            if float(inst.get("strike", 0) or 0) != target_strike:
                continue
            inst_expiry = inst.get("expiry")
            if not inst_expiry:
                continue
            try:
                inst_date = datetime.strptime(str(inst_expiry), "%Y-%m-%d").date()
            except ValueError:
                continue
            if inst_date == expiry:
                return inst.get("tradingsymbol")
        return None

    def resolve_from_angel_symbol(self, angel_symbol: str, exchange: str) -> str:
        """Convert an AngelOne-format option symbol to a Kite tradingsymbol.

        Symbol formats vary by exchange — Kite's tradingsymbol is *not* the same
        string as Angel's, so we always look up by structured fields parsed
        from the Angel symbol rather than relying on string equality.

        Supported Angel formats:
          - NFO:  ``NIFTY09APR2622500CE``  (DDMMMYY + strike + CE/PE)
          - NFO:  ``BANKNIFTY09APR2651000CE``
          - BFO:  ``SENSEX2640923900CE``   (YY + month_int(1–12) + DD + strike + CE/PE)
                  Month is *unpadded* (1, 2, ... 9, 10, 11, 12).

        Returns "" when the contract cannot be located in Kite's instrument list.
        """
        if not angel_symbol:
            return ""

        import re

        # 1. Direct match — useful only when the broker formats overlap. Kept
        #    cheap (one pass over cached instruments) and safe.
        try:
            instruments = self._load_instruments(exchange)
            for inst in instruments:
                if inst.get("tradingsymbol") == angel_symbol:
                    return angel_symbol
        except Exception:
            logger.debug("Kite instrument lookup failed for %s", angel_symbol)

        # 2. NFO format — DDMMMYY
        m = re.match(
            r"^(NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$",
            angel_symbol,
        )
        if m:
            name, day, mon, year, strike, opt = m.groups()
            try:
                expiry = datetime.strptime(f"{day}{mon}{year}", "%d%b%y").date()
            except ValueError:
                return ""
            resolved = self.resolve_option_symbol(
                name=name, expiry=expiry, strike=float(strike),
                option_type=opt, exchange=exchange,
            )
            if resolved:
                return resolved

        # 3. BFO (SENSEX/BANKEX) format — YY + month_int(1–12, unpadded) + DD
        #    Strike comes after the YYM[M]DD prefix. We greedy-match strike
        #    using the trailing CE/PE; the prefix length determines the month
        #    width (1 or 2 digits).
        m = re.match(r"^(SENSEX|BANKEX)(\d+)(CE|PE)$", angel_symbol)
        if m:
            name, mid, opt = m.groups()
            for month_width in (2, 1):
                # Layout: YY(2) + MM(month_width) + DD(2) + STRIKE(rest)
                prefix_len = 2 + month_width + 2
                if len(mid) <= prefix_len:
                    continue
                try:
                    yy = int(mid[0:2])
                    mm = int(mid[2 : 2 + month_width])
                    dd = int(mid[2 + month_width : 4 + month_width])
                    strike_part = mid[prefix_len:]
                    if not (1 <= mm <= 12) or not (1 <= dd <= 31) or not strike_part.isdigit():
                        continue
                    expiry = datetime(2000 + yy, mm, dd).date()
                    resolved = self.resolve_option_symbol(
                        name=name, expiry=expiry, strike=float(strike_part),
                        option_type=opt, exchange=exchange,
                    )
                    if resolved:
                        return resolved
                except (ValueError, TypeError):
                    continue

        logger.warning(
            "Could not resolve Kite tradingsymbol for Angel symbol %s on %s",
            angel_symbol, exchange,
        )
        return ""

    def get_instrument_token(
        self,
        tradingsymbol: str,
        exchange: str,
    ) -> Optional[int]:
        """Return Kite's `instrument_token` for a Kite-format tradingsymbol.

        Kite uses its own numeric instrument_token rather than Angel's
        `symboltoken`. Most order-placement calls do not require this — they
        accept tradingsymbol + exchange — but the WebSocket feed and some
        analytics endpoints do.
        """
        try:
            for inst in self._load_instruments(exchange):
                if inst.get("tradingsymbol") == tradingsymbol:
                    tok = inst.get("instrument_token")
                    return int(tok) if tok else None
        except Exception:
            logger.exception("Kite instrument_token lookup failed for %s", tradingsymbol)
        return None
