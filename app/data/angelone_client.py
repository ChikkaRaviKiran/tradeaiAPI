"""AngelOne SmartAPI client for market data fetching."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyotp
from SmartApi import SmartConnect

from app.core.config import settings
from app.core.models import Candle, GlobalIndex, OptionsChainRow

logger = logging.getLogger(__name__)


class AngelOneClient:
    """Wrapper around AngelOne SmartAPI for authentication and data retrieval."""

    def __init__(self) -> None:
        self._smart_api: Optional[SmartConnect] = None
        self._auth_token: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._last_auth: Optional[datetime] = None
        # NIFTY token on AngelOne
        self.nifty_token = "99926000"
        self.nifty_symbol = "NIFTY"
        self.exchange = "NSE"
        self.nfo_exchange = "NFO"

    # ── Authentication ────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Authenticate with AngelOne SmartAPI using TOTP.

        Uses MPIN if configured (required by AngelOne since 2025),
        otherwise falls back to password.
        """
        try:
            self._smart_api = SmartConnect(api_key=settings.angelone_api_key)
            totp = pyotp.TOTP(settings.angelone_totp_secret).now()
            # AngelOne now requires MPIN instead of password
            credential = settings.angelone_mpin or settings.angelone_password
            data = self._smart_api.generateSession(
                settings.angelone_client_id,
                credential,
                totp,
            )
            if not data or data.get("status") is False:
                logger.error("AngelOne authentication failed: %s", data)
                return False

            self._auth_token = data["data"]["jwtToken"]
            self._feed_token = self._smart_api.getfeedToken()
            self._last_auth = datetime.now()
            logger.info("AngelOne authenticated successfully.")
            return True
        except Exception:
            logger.exception("AngelOne authentication error")
            return False

    def ensure_authenticated(self) -> None:
        """Re-authenticate if token is stale (>6 hours)."""
        if (
            self._smart_api is None
            or self._last_auth is None
            or (datetime.now() - self._last_auth) > timedelta(hours=6)
        ):
            if not self.authenticate():
                raise ConnectionError("Failed to authenticate with AngelOne")

    # ── Historical Candle Data ────────────────────────────────────────────

    def get_candle_data(
        self,
        symbol_token: str,
        exchange: str,
        interval: str = "ONE_MINUTE",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[Candle]:
        """Fetch historical candle data.

        Args:
            symbol_token: AngelOne symbol token.
            exchange: Exchange (NSE, NFO).
            interval: Candle interval (ONE_MINUTE, FIVE_MINUTE, etc.).
            from_date: Start date in YYYY-MM-DD HH:MM format.
            to_date: End date in YYYY-MM-DD HH:MM format.
        """
        self.ensure_authenticated()
        now = datetime.now()
        if to_date is None:
            to_date = now.strftime("%Y-%m-%d %H:%M")
        if from_date is None:
            from_date = (now - timedelta(days=1)).strftime("%Y-%m-%d 09:15")

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        try:
            # Retry up to 3 times — AngelOne occasionally returns AB1004
            for attempt in range(3):
                resp = self._smart_api.getCandleData(params)
                if resp and resp.get("status") is True:
                    break
                logger.warning(
                    "Candle data attempt %d failed: %s", attempt + 1, resp
                )
                time.sleep(1)
            else:
                logger.error("Failed to fetch candle data after 3 attempts")
                return []

            candles: list[Candle] = []
            for row in resp.get("data", []):
                candles.append(
                    Candle(
                        symbol=symbol_token,
                        timestamp=datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S%z"),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=int(row[5]),
                    )
                )
            return candles
        except Exception:
            logger.exception("Error fetching candle data")
            return []

    def get_nifty_candles(
        self,
        interval: str = "ONE_MINUTE",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[Candle]:
        """Fetch NIFTY index candle data."""
        return self.get_candle_data(
            self.nifty_token, self.exchange, interval, from_date, to_date
        )

    # ── LTP / Quote ──────────────────────────────────────────────────────

    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        """Get last traded price for a symbol."""
        self.ensure_authenticated()
        try:
            data = self._smart_api.ltpData(exchange, symbol, token)
            if data and data.get("status"):
                return float(data["data"]["ltp"])
        except Exception:
            logger.exception("Error fetching LTP for %s", symbol)
        return None

    # ── Options Chain ────────────────────────────────────────────────────

    def get_option_chain(self, expiry_date: str) -> list[OptionsChainRow]:
        """Fetch NIFTY options chain for a given expiry.

        Uses individual strike data since SmartAPI doesn't have a native
        options chain endpoint—we build it from LTP + OI calls.

        Args:
            expiry_date: Expiry in DDMMMYYYY format (e.g., '06MAR2026').
        """
        self.ensure_authenticated()
        rows: list[OptionsChainRow] = []

        nifty_ltp = self.get_ltp(self.exchange, self.nifty_symbol, self.nifty_token)
        if nifty_ltp is None:
            return rows

        # Build strikes around current price (±1000 range, step 50)
        base = int(round(nifty_ltp / 50) * 50)
        strikes = range(base - 1000, base + 1050, 50)

        for strike in strikes:
            try:
                ce_symbol = f"NIFTY{expiry_date}{strike}CE"
                pe_symbol = f"NIFTY{expiry_date}{strike}PE"

                # Search for tokens
                ce_info = self._search_symbol(ce_symbol)
                pe_info = self._search_symbol(pe_symbol)

                row = OptionsChainRow(strike_price=float(strike))

                if ce_info:
                    ce_quote = self._get_quote(self.nfo_exchange, ce_info)
                    if ce_quote:
                        row.call_ltp = ce_quote.get("ltp", 0.0)
                        row.call_oi = ce_quote.get("opnInterest", 0)
                        row.call_volume = ce_quote.get("exchTradVol", 0)

                if pe_info:
                    pe_quote = self._get_quote(self.nfo_exchange, pe_info)
                    if pe_quote:
                        row.put_ltp = pe_quote.get("ltp", 0.0)
                        row.put_oi = pe_quote.get("opnInterest", 0)
                        row.put_volume = pe_quote.get("exchTradVol", 0)

                rows.append(row)
                time.sleep(0.1)  # Rate limiting
            except Exception:
                logger.debug("Skipping strike %s", strike)
                continue

        return rows

    def _search_symbol(self, trading_symbol: str) -> Optional[dict]:
        """Search for a symbol token using the search API."""
        try:
            result = self._smart_api.searchScrip(self.nfo_exchange, trading_symbol)
            if result and result.get("status") and result.get("data"):
                for item in result["data"]:
                    if item.get("tradingsymbol") == trading_symbol:
                        return item
        except Exception:
            pass
        return None

    def _get_quote(self, exchange: str, symbol_info: dict) -> Optional[dict]:
        """Get full quote for a symbol."""
        try:
            token = symbol_info.get("symboltoken", "")
            tsym = symbol_info.get("tradingsymbol", "")
            data = self._smart_api.ltpData(exchange, tsym, token)
            if data and data.get("status"):
                return data.get("data", {})
        except Exception:
            pass
        return None

    # ── Global Market Data ───────────────────────────────────────────────

    def get_global_indices(self) -> list[GlobalIndex]:
        """Fetch global index data (approximate via AngelOne or fallback).

        AngelOne primarily covers Indian markets. For global indices,
        we use the available index data and can be extended with other APIs.
        """
        # These are indicative global index tokens on AngelOne (where available)
        # In production, supplement with a global markets data API
        indices: list[GlobalIndex] = []
        global_symbols = [
            ("DOW JONES", "^DJI"),
            ("NASDAQ", "^IXIC"),
            ("S&P 500", "^GSPC"),
            ("FTSE", "^FTSE"),
            ("DAX", "^GDAXI"),
            ("NIKKEI", "^N225"),
            ("HANG SENG", "^HSI"),
        ]
        # Fallback: return neutral if we can't fetch
        for name, symbol in global_symbols:
            indices.append(GlobalIndex(symbol=name, change_pct=0.0, last_price=0.0))

        logger.info(
            "Global indices fetched (placeholder - integrate dedicated global API)"
        )
        return indices

    # ── Instrument List ──────────────────────────────────────────────────

    def get_nifty_option_tokens(
        self, expiry: str, strike: float, option_type: str
    ) -> Optional[dict]:
        """Get token info for a specific NIFTY option.

        Args:
            expiry: Expiry in DDMMMYYYY (e.g. '06MAR2026').
            strike: Strike price.
            option_type: 'CE' or 'PE'.
        """
        symbol = f"NIFTY{expiry}{int(strike)}{option_type}"
        return self._search_symbol(symbol)

    # ── Expiry Discovery ─────────────────────────────────────────────────

    def get_nearest_weekly_expiry(self) -> Optional[str]:
        """Find the nearest valid NIFTY weekly expiry from the AngelOne instrument list.

        Downloads the OpenAPI instrument master CSV, filters for NIFTY options,
        and returns the nearest expiry in DDMMMYYYY format.
        """
        self.ensure_authenticated()
        try:
            import io
            import csv
            from urllib.request import urlopen

            url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
            logger.info("Fetching AngelOne instrument master for expiry discovery...")

            import json
            with urlopen(url, timeout=30) as resp:
                instruments = json.loads(resp.read().decode())

            today = datetime.now().date()
            expiry_dates: set[datetime] = set()

            for inst in instruments:
                name = inst.get("name", "")
                exch_seg = inst.get("exch_seg", "")
                symbol = inst.get("symbol", "")
                expiry_str = inst.get("expiry", "")

                if exch_seg != "NFO" or name != "NIFTY":
                    continue
                # Only option contracts (CE/PE in symbol)
                if "CE" not in symbol and "PE" not in symbol:
                    continue
                if not expiry_str:
                    continue

                try:
                    exp_date = datetime.strptime(expiry_str, "%d%b%Y").date()
                    if exp_date >= today:
                        expiry_dates.add(exp_date)
                except ValueError:
                    continue

            if not expiry_dates:
                logger.warning("No NIFTY expiry dates found in instrument master")
                return None

            nearest = min(expiry_dates)
            result = nearest.strftime("%d%b%Y").upper()
            logger.info("Nearest NIFTY weekly expiry from instrument master: %s", result)
            return result

        except Exception:
            logger.exception("Error fetching expiry from instrument master")
            return None

    def candles_to_dataframe(self, candles: list[Candle]) -> pd.DataFrame:
        """Convert candle list to a pandas DataFrame."""
        if not candles:
            return pd.DataFrame()
        data = [c.model_dump() for c in candles]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
