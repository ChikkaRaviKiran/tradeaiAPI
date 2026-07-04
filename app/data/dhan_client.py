"""DhanHQ client wrapper for order execution.

Used when ``TRADING_ACCOUNT=dhan``. Data continues to flow from AngelOne
(quotes, candles, option chain); only order placement and exits are routed
through Dhan. Mirrors the design of :mod:`app.data.kite_client`.

Authentication: Dhan uses a static access token (JWT) generated from the
Dhan web portal (no OAuth flow). Combined with ``DHAN_CLIENT_ID``.

Symbol resolution: Dhan orders require a numeric ``securityId`` rather than
a tradingsymbol. We download the published scrip-master CSV once per IST
trading day and resolve by structured fields (underlying + expiry + strike
+ option_type + exchange) so we never depend on broker-specific symbol
formats.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, date
from typing import Optional

import pytz
import requests

try:
    from dhanhq import dhanhq as _DhanSDK  # type: ignore
except Exception:  # pragma: no cover — soft import so app boots without dhanhq installed
    _DhanSDK = None  # type: ignore

# v3.x of the SDK requires a DhanContext wrapper rather than passing
# (client_id, access_token) directly to dhanhq(). Import is optional so
# older v2.x installs still work.
try:
    from dhanhq import DhanContext as _DhanContext  # type: ignore
except Exception:  # pragma: no cover
    _DhanContext = None  # type: ignore

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

EXCHANGE_NSE_FNO = "NSE_FNO"
EXCHANGE_BSE_FNO = "BSE_FNO"
EXCHANGE_NSE_EQ = "NSE_EQ"
EXCHANGE_BSE_EQ = "BSE_EQ"

# Dhan publishes the scrip master daily. The "detailed" CSV contains
# expiry/strike/option_type fields needed for option resolution.
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
SCRIP_MASTER_CACHE_SECONDS = 12 * 3600

# Map our generic exchange tokens → Dhan exchange-segment values
_EXCHANGE_SEGMENT_MAP = {
    "NFO": "NSE_FNO",
    "BFO": "BSE_FNO",
    "NSE": "NSE_EQ",
    "BSE": "BSE_EQ",
    "NSE_FNO": "NSE_FNO",
    "BSE_FNO": "BSE_FNO",
    "NSE_EQ": "NSE_EQ",
    "BSE_EQ": "BSE_EQ",
}

# Reverse: dhan exchange-segment → BaseBroker exchange token
_EXCHANGE_REVERSE = {
    "NSE_FNO": "NFO",
    "BSE_FNO": "BFO",
    "NSE_EQ": "NSE",
    "BSE_EQ": "BSE",
}


def _to_dhan_exchange_segment(exchange: str) -> str:
    return _EXCHANGE_SEGMENT_MAP.get((exchange or "").upper(), "NSE_FNO")


def _mask_dhan_proxy(url: str) -> str:
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


class DhanClient:
    """Thin wrapper around the ``dhanhq`` SDK for order placement.

    Threading note: the SDK is synchronous. Wrap calls in
    ``asyncio.to_thread(...)`` from async contexts.
    """

    def __init__(self, client_id: str, access_token: str, proxy_url: str = "") -> None:
        if _DhanSDK is None:
            raise RuntimeError(
                "dhanhq is not installed. Run `pip install dhanhq`."
            )
        if not client_id or not access_token:
            raise RuntimeError("DhanClient requires client_id and access_token")
        self.client_id = client_id
        self.access_token = access_token
        self.proxy_url = (proxy_url or "").strip()
        # SDK shape changed in v3.x: dhanhq(client_id, token) → dhanhq(DhanContext(...))
        # Try the new style first, fall back to legacy positional args.
        if _DhanContext is not None:
            try:
                ctx = _DhanContext(client_id, access_token)
                self._dhan = _DhanSDK(ctx)
            except TypeError:
                self._dhan = _DhanSDK(client_id, access_token)
        else:
            self._dhan = _DhanSDK(client_id, access_token)
        # Route all outbound HTTP through the per-account proxy so this Dhan
        # login uses its dedicated Lightsail IP (matches the OptionSelling
        # multi-account isolation pattern).
        if self.proxy_url:
            try:
                self._install_proxy_on_sdk_session(self.proxy_url)
                logger.info(
                    "DhanClient[%s] routing via proxy %s",
                    self.client_id, _mask_dhan_proxy(self.proxy_url),
                )
            except Exception:
                logger.exception("DhanClient: failed to install proxy on SDK session (proceeding without proxy)")
        self._instrument_cache: list[dict] = []
        self._instrument_cache_date: Optional[date] = None

    def _install_proxy_on_sdk_session(self, proxy_url: str) -> None:
        """Best-effort: locate the internal requests.Session on the dhanhq SDK
        instance and attach a proxies dict. Structure varies across SDK
        versions so we probe common attribute paths.
        """
        proxies = {"http": proxy_url, "https": proxy_url}
        candidates: list = []
        for obj in (self._dhan, getattr(self._dhan, "dhan_context", None),
                    getattr(self._dhan, "context", None)):
            if obj is None:
                continue
            candidates.append(obj)
            # SDKs commonly hold the session on one of these attributes
            for attr in ("session", "_session", "http_session", "_http", "api", "_api"):
                child = getattr(obj, attr, None)
                if child is not None:
                    candidates.append(child)
        for c in candidates:
            sess = getattr(c, "session", None) if not hasattr(c, "proxies") else c
            if sess is None:
                sess = getattr(c, "_session", None)
            if sess is not None and hasattr(sess, "proxies"):
                try:
                    sess.proxies.update(proxies)
                except Exception:
                    try:
                        sess.proxies = proxies
                    except Exception:
                        continue
        # Absolute fallback: process-wide env vars scoped to this call site.
        # We deliberately do NOT set os.environ here — that would leak to
        # every other broker's requests. The SDK-session patching above is
        # the isolated per-account path.

    # ── Profile / health ─────────────────────────────────────────────

    def get_fund_limits(self) -> dict:
        """Return account fund limits / available margin."""
        try:
            resp = self._dhan.get_fund_limits()
            if isinstance(resp, dict) and resp.get("status") == "success":
                return resp.get("data", {}) or {}
        except Exception:
            logger.exception("Dhan get_fund_limits failed")
        return {}

    # ── Orders ───────────────────────────────────────────────────────

    def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        product_type: str = "INTRADAY",
        price: float = 0.0,
        trigger_price: float = 0.0,
        validity: str = "DAY",
    ) -> dict:
        """Place an order. Returns the raw Dhan response dict.

        Dhan expects string constants for transaction/order/product types.
        Both the order_type and price are normalised here as a safety
        net so a bad caller can never accidentally place a LIMIT order
        when MARKET was intended (Dhan treats any non-zero price on a
        MARKET payload as a LIMIT order in some account configs, leaving
        the order pending in the order book).
        """
        order_type_norm = (order_type or "MARKET").upper().strip()
        if order_type_norm not in {"LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_MARKET"}:
            order_type_norm = "MARKET"
        # MARKET / SL-MARKET must have price=0 and trigger_price=0 (for MARKET)
        if order_type_norm == "MARKET":
            price = 0.0
            trigger_price = 0.0
        elif order_type_norm == "STOP_LOSS_MARKET":
            price = 0.0

        payload_log = {
            "security_id": str(security_id),
            "exchange_segment": exchange_segment,
            "transaction_type": transaction_type,
            "quantity": int(quantity),
            "order_type": order_type_norm,
            "product_type": product_type,
            "price": float(price or 0),
            "trigger_price": float(trigger_price or 0),
            "validity": validity,
        }
        logger.info("DHAN PLACE_ORDER payload: %s", payload_log)

        try:
            resp = self._dhan.place_order(
                security_id=str(security_id),
                exchange_segment=exchange_segment,
                transaction_type=transaction_type,
                quantity=int(quantity),
                order_type=order_type_norm,
                product_type=product_type,
                price=float(price or 0),
                trigger_price=float(trigger_price or 0),
                validity=validity,
            )
            logger.info("DHAN PLACE_ORDER response: %s", resp)
            return resp if isinstance(resp, dict) else {"status": "failure", "raw": str(resp)}
        except Exception as exc:
            logger.exception("Dhan place_order failed")
            return {"status": "failure", "remarks": {"error_message": str(exc)}}

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = self._dhan.cancel_order(order_id)
            return isinstance(resp, dict) and resp.get("status") == "success"
        except Exception:
            logger.exception("Dhan cancel_order failed for %s", order_id)
            return False

    def modify_order(
        self,
        order_id: str,
        order_type: Optional[str] = None,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        validity: str = "DAY",
        leg_name: str = "ENTRY_LEG",
    ) -> bool:
        try:
            resp = self._dhan.modify_order(
                order_id=order_id,
                order_type=order_type or "MARKET",
                leg_name=leg_name,
                quantity=int(quantity) if quantity is not None else 0,
                price=float(price) if price is not None else 0.0,
                trigger_price=float(trigger_price) if trigger_price is not None else 0.0,
                disclosed_quantity=0,
                validity=validity,
            )
            return isinstance(resp, dict) and resp.get("status") == "success"
        except Exception:
            logger.exception("Dhan modify_order failed for %s", order_id)
            return False

    def get_order_by_id(self, order_id: str) -> dict:
        try:
            resp = self._dhan.get_order_by_id(order_id)
            if isinstance(resp, dict) and resp.get("status") == "success":
                data = resp.get("data")
                if isinstance(data, list) and data:
                    return data[0]
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.exception("Dhan get_order_by_id failed for %s", order_id)
        return {}

    def get_order_list(self) -> list[dict]:
        try:
            resp = self._dhan.get_order_list()
            if isinstance(resp, dict) and resp.get("status") == "success":
                data = resp.get("data") or []
                return data if isinstance(data, list) else []
        except Exception:
            logger.exception("Dhan get_order_list failed")
        return []

    def get_positions(self) -> list[dict]:
        try:
            resp = self._dhan.get_positions()
            if isinstance(resp, dict) and resp.get("status") == "success":
                data = resp.get("data") or []
                return data if isinstance(data, list) else []
        except Exception:
            logger.exception("Dhan get_positions failed")
        return []

    def get_ltp(self, exchange_segment: str, security_id: str) -> Optional[float]:
        """Best-effort LTP fetch using the SDK's quote endpoint."""
        try:
            payload = {exchange_segment: [int(security_id)]}
            if hasattr(self._dhan, "ticker_data"):
                resp = self._dhan.ticker_data(payload)
            elif hasattr(self._dhan, "quote_data"):
                resp = self._dhan.quote_data(payload)
            else:
                return None
            if isinstance(resp, dict) and resp.get("status") == "success":
                data = (resp.get("data") or {}).get("data") or resp.get("data") or {}
                seg = data.get(exchange_segment) if isinstance(data, dict) else None
                if isinstance(seg, dict):
                    row = seg.get(str(security_id)) or seg.get(int(security_id))
                    if isinstance(row, dict):
                        ltp = row.get("last_price") or row.get("LTP") or row.get("ltp")
                        if ltp is not None:
                            return float(ltp)
        except Exception:
            logger.debug("Dhan LTP fetch failed for %s/%s", exchange_segment, security_id)
        return None

    # ── Instrument master ────────────────────────────────────────────

    def _load_scrip_master(self, force: bool = False) -> list[dict]:
        today = datetime.now(_IST).date()
        if not force and self._instrument_cache and self._instrument_cache_date == today:
            return self._instrument_cache
        try:
            r = requests.get(SCRIP_MASTER_URL, timeout=60)
            r.raise_for_status()
            reader = csv.DictReader(io.StringIO(r.text))
            rows = [row for row in reader]
            self._instrument_cache = rows
            self._instrument_cache_date = today
            logger.info("Dhan scrip master loaded: %d rows", len(rows))
            return rows
        except Exception:
            logger.exception("Dhan scrip master download failed")
            return self._instrument_cache  # may be empty

    def refresh_scrip_master(self, force: bool = True) -> int:
        rows = self._load_scrip_master(force=force)
        return len(rows)

    @staticmethod
    def _row_get(row: dict, *keys: str) -> str:
        for k in keys:
            v = row.get(k)
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        return ""

    def resolve_option_security_id(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: str,
        exchange: str,
    ) -> Optional[str]:
        """Find Dhan ``securityId`` for an option contract."""
        row = self._find_option_row(underlying, expiry, strike, option_type, exchange)
        if not row:
            return None
        sec_id = self._row_get(row, "SEM_SMST_SECURITY_ID", "SECURITY_ID")
        return sec_id or None

    def resolve_option_lot_size(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: str,
        exchange: str,
    ) -> Optional[int]:
        """Return the exchange-published lot size for an option contract.

        Critical for expiries straddling SEBI lot-size revisions: existing
        (older-issued) contracts retain their original lot size while newly
        issued ones use the revised value. Using a stale static lot_size
        from instrument config causes "Invalid Quantity" rejections.
        """
        row = self._find_option_row(underlying, expiry, strike, option_type, exchange)
        if not row:
            return None
        raw = self._row_get(row, "SEM_LOT_UNITS", "LOT_SIZE", "LOT_UNITS")
        try:
            ls = int(float(raw)) if raw else 0
            return ls if ls > 0 else None
        except (ValueError, TypeError):
            return None

    def _find_option_row(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: str,
        exchange: str,
    ) -> Optional[dict]:
        """Locate the scrip-master row for a specific option contract.

        Args:
            underlying: e.g. "NIFTY", "BANKNIFTY", "SENSEX".
            expiry: contract expiry as ``date``.
            strike: numeric strike price.
            option_type: "CE" or "PE".
            exchange: "NFO", "BFO", or Dhan-segment string.
        """
        if not underlying or not expiry or not strike or not option_type:
            return None
        target_segment = _to_dhan_exchange_segment(exchange)
        target_strike = float(strike)
        target_opt = option_type.upper()
        target_under = underlying.upper()

        for row in self._load_scrip_master():
            seg = self._row_get(row, "SEM_EXM_EXCH_ID", "EXCH_ID", "exchange_segment")
            # SEM_EXM_EXCH_ID is "NSE"/"BSE"; SEM_SEGMENT denotes "D" (derivative).
            # Dhan's "detailed" CSV uses SEGMENT (no SEM_ prefix), so fall
            # back to that — without the fallback every option row was
            # silently rejected and the resolver returned None for all
            # contracts (NIFTY+SENSEX) starting from the day Dhan changed
            # the master schema.
            segment = self._row_get(row, "SEM_SEGMENT", "SEGMENT")
            instr = self._row_get(row, "SEM_INSTRUMENT_NAME", "INSTRUMENT", "INSTRUMENT_TYPE")
            if instr.upper() not in {"OPTIDX", "OPTSTK", "OPTFUT"}:
                continue
            # Map row to a dhan exchange segment
            if seg == "NSE" and segment in {"D", "E"}:
                row_segment = "NSE_FNO"
            elif seg == "BSE" and segment in {"D", "E"}:
                row_segment = "BSE_FNO"
            else:
                row_segment = ""
            if row_segment != target_segment:
                continue

            # Exact-match against the dedicated underlying-symbol column.
            # A substring match would wrongly route NIFTY → FINNIFTY /
            # MIDCPNIFTY / BANKNIFTY because "NIFTY" is contained in all
            # of them. Compare against SM_SYMBOL_NAME (or equivalents)
            # exactly; fall back to a trading-symbol PREFIX match only
            # when the underlying column is missing so older master
            # schemas still resolve.
            under_exact = self._row_get(
                row, "SM_SYMBOL_NAME", "UNDERLYING_SYMBOL",
                "SYMBOL_NAME", "DISPLAY_NAME",
            ).upper()
            if under_exact:
                if under_exact != target_under:
                    continue
            else:
                tsym = self._row_get(row, "SEM_TRADING_SYMBOL").upper()
                if not tsym.startswith(target_under):
                    continue
                # Ensure prefix isn't accidentally matching a longer
                # underlying (e.g. NIFTY vs FINNIFTY). The character
                # right after the prefix must NOT be a letter — option
                # trading symbols put the expiry digits/format next.
                tail = tsym[len(target_under):len(target_under) + 1]
                if tail.isalpha():
                    continue
            opt = self._row_get(row, "SEM_OPTION_TYPE", "OPTION_TYPE").upper()
            if opt != target_opt:
                continue
            try:
                row_strike = float(
                    self._row_get(row, "SEM_STRIKE_PRICE", "STRIKE_PRICE") or 0
                )
            except ValueError:
                continue
            if row_strike != target_strike:
                continue
            exp_str = self._row_get(row, "SEM_EXPIRY_DATE", "SM_EXPIRY_DATE", "EXPIRY_DATE")
            if not exp_str:
                continue
            row_expiry = self._parse_expiry(exp_str)
            if not row_expiry or row_expiry != expiry:
                continue
            return row
        return None

    @staticmethod
    def _parse_expiry(s: str) -> Optional[date]:
        s = s.strip()
        # Dhan typically uses "2026-04-09" or "2026-04-09 14:30:00"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(s.split(".")[0], fmt).date()
            except ValueError:
                continue
        return None

    def resolve_from_angel_symbol(
        self, angel_symbol: str, exchange: str
    ) -> Optional[str]:
        """Convert AngelOne-format option symbol to Dhan ``securityId``.

        Supported formats mirror :meth:`KiteClient.resolve_from_angel_symbol`:
          - NFO:  ``NIFTY09APR2622500CE`` (DDMMMYY + strike + CE/PE)
          - BFO:  ``SENSEX2640923900CE``  (YY + month_int + DD + strike + CE/PE)
        """
        if not angel_symbol:
            return None
        import re

        m = re.match(
            r"^(NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$",
            angel_symbol,
        )
        if m:
            name, day, mon, year, strike, opt = m.groups()
            try:
                expiry = datetime.strptime(f"{day}{mon}{year}", "%d%b%y").date()
            except ValueError:
                return None
            return self.resolve_option_security_id(
                underlying=name, expiry=expiry, strike=float(strike),
                option_type=opt, exchange=exchange,
            )

        m = re.match(r"^(SENSEX|BANKEX)(\d+)(CE|PE)$", angel_symbol)
        if m:
            name, mid, opt = m.groups()
            for month_width in (2, 1):
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
                    sec = self.resolve_option_security_id(
                        underlying=name, expiry=expiry, strike=float(strike_part),
                        option_type=opt, exchange=exchange,
                    )
                    if sec:
                        return sec
                except (ValueError, TypeError):
                    continue
        return None
