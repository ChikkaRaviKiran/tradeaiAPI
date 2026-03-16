"""AngelOne SmartAPI client for market data fetching."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyotp
import pytz
from SmartApi import SmartConnect

_IST = pytz.timezone("Asia/Kolkata")

from app.core.config import settings
from app.core.models import Candle, OptionsChainRow

logger = logging.getLogger(__name__)


class AngelOneClient:
    """Wrapper around AngelOne SmartAPI for authentication and data retrieval."""

    def __init__(self) -> None:
        self._smart_api: Optional[SmartConnect] = None
        self._auth_token: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._last_auth: Optional[datetime] = None
        self._instrument_master: Optional[list] = None  # Cached instrument master
        # NIFTY token on AngelOne
        self.nifty_token = "99926000"
        self.nifty_symbol = "NIFTY"
        self.exchange = "NSE"
        self.nfo_exchange = "NFO"

    # ── Instrument Master Cache ───────────────────────────────────────────

    def _get_instrument_master(self) -> list:
        """Download and cache the AngelOne instrument master JSON.

        Cached for the lifetime of this client (one trading day).
        """
        if self._instrument_master is not None:
            return self._instrument_master

        import json
        from urllib.request import urlopen

        url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
        logger.info("Downloading AngelOne instrument master (~20MB)...")
        with urlopen(url, timeout=60) as resp:
            self._instrument_master = json.loads(resp.read().decode())
        logger.info("Instrument master loaded: %d instruments", len(self._instrument_master))
        return self._instrument_master

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
            self._last_auth = datetime.now(_IST)
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
            or (datetime.now(_IST) - self._last_auth) > timedelta(hours=6)
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
        now = datetime.now(_IST)
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

    def get_nifty_futures_candles(
        self,
        interval: str = "ONE_MINUTE",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[Candle]:
        """Fetch current month NIFTY Futures candle data.

        NIFTY futures have real volume, unlike the spot index (always 0).
        Used to get volume data for VWAP computation and volume analysis.
        """
        token = self._get_nifty_fut_token()
        if not token:
            logger.warning("Could not find NIFTY Futures token — no volume data")
            return []
        return self.get_candle_data(token, self.nfo_exchange, interval, from_date, to_date)

    def _get_nifty_fut_token(self) -> Optional[str]:
        """Find the current month NIFTY futures symbol token.

        Uses the instrument master JSON (same source as expiry discovery)
        to reliably find the nearest NIFTY futures contract token.
        searchScrip is unreliable because it returns too many option matches.
        """
        if hasattr(self, "_nifty_fut_token_cache") and self._nifty_fut_token_cache:
            return self._nifty_fut_token_cache

        try:
            import json
            from urllib.request import urlopen

            url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
            logger.info("Fetching instrument master for NIFTY Futures token...")

            instruments = self._get_instrument_master()

            today = datetime.now(_IST).date()
            best_token = None
            best_symbol = None
            best_expiry = None

            for inst in instruments:
                name = inst.get("name", "")
                exch_seg = inst.get("exch_seg", "")
                symbol = inst.get("symbol", "")
                expiry_str = inst.get("expiry", "")
                inst_type = inst.get("instrumenttype", "")

                # Look for NIFTY futures in NFO segment
                if exch_seg != "NFO" or name != "NIFTY":
                    continue
                # FUTIDX = Index Futures
                if inst_type != "FUTIDX":
                    continue
                if not symbol.endswith("FUT"):
                    continue
                if not expiry_str:
                    continue

                try:
                    exp_date = datetime.strptime(expiry_str, "%d%b%Y").date()
                except ValueError:
                    continue

                # Must not be expired
                if exp_date < today:
                    continue

                # Pick the nearest expiry
                if best_expiry is None or exp_date < best_expiry:
                    best_expiry = exp_date
                    best_token = inst.get("token", "")
                    best_symbol = symbol

            if best_token:
                self._nifty_fut_token_cache = best_token
                logger.info(
                    "NIFTY Futures token from instrument master: %s (%s, expiry=%s)",
                    best_token, best_symbol, best_expiry,
                )
                return self._nifty_fut_token_cache

            logger.warning("No NIFTY Futures contract found in instrument master")
        except Exception:
            logger.exception("Error fetching NIFTY futures token from instrument master")
        return None

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

        Batches token lookups then uses getMarketData(FULL) in batches
        of up to 50 tokens to minimise API calls and get real OI/volume.

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
        strikes = list(range(base - 1000, base + 1050, 50))

        # Phase 1: Search all symbol tokens (uses instrument master — no API calls)
        token_map: dict[str, dict] = {}  # token -> symbol_info
        strike_tokens: dict[int, dict] = {}  # strike -> {ce_token, pe_token}
        for strike in strikes:
            ce_symbol = f"NIFTY{expiry_date}{strike}CE"
            pe_symbol = f"NIFTY{expiry_date}{strike}PE"
            ce_info = self._search_symbol(ce_symbol)
            pe_info = self._search_symbol(pe_symbol)
            entry: dict = {}
            if ce_info:
                t = ce_info.get("symboltoken", "")
                entry["ce_token"] = t
                token_map[t] = ce_info
            if pe_info:
                t = pe_info.get("symboltoken", "")
                entry["pe_token"] = t
                token_map[t] = pe_info
            strike_tokens[strike] = entry

        # Phase 2: Batch getMarketData(FULL) — max 50 tokens per call
        all_tokens = list(token_map.keys())
        quotes: dict[str, dict] = {}  # token -> market data
        batch_size = 50
        for i in range(0, len(all_tokens), batch_size):
            batch = all_tokens[i : i + batch_size]
            try:
                data = self._smart_api.getMarketData(
                    "FULL", {self.nfo_exchange: batch}
                )
                if data and data.get("status") and data.get("data"):
                    for item in data["data"].get("fetched", []):
                        quotes[str(item.get("symbolToken", ""))] = item
            except Exception as e:
                logger.warning("Batch market data failed: %s", e)
            time.sleep(0.3)  # Rate limiting between batches

        # Phase 3: Build rows
        for strike in strikes:
            entry = strike_tokens.get(strike, {})
            row = OptionsChainRow(strike_price=float(strike))
            ce_q = quotes.get(entry.get("ce_token", ""), {})
            if ce_q:
                row.call_ltp = ce_q.get("ltp", 0.0)
                row.call_oi = ce_q.get("opnInterest", 0)
                row.call_volume = ce_q.get("exchTradVol", 0)
            pe_q = quotes.get(entry.get("pe_token", ""), {})
            if pe_q:
                row.put_ltp = pe_q.get("ltp", 0.0)
                row.put_oi = pe_q.get("opnInterest", 0)
                row.put_volume = pe_q.get("exchTradVol", 0)
            rows.append(row)

        return rows

    def _search_symbol(self, trading_symbol: str) -> Optional[dict]:
        """Search for a symbol token — first from instrument master, then API fallback."""
        # Fast path: look up from cached instrument master (no API call)
        try:
            master = self._get_instrument_master()
            for item in master:
                if item.get("symbol") == trading_symbol and item.get("exch_seg") == self.nfo_exchange:
                    return {
                        "tradingsymbol": item.get("symbol", ""),
                        "symboltoken": item.get("token", ""),
                    }
        except Exception:
            logger.debug("Instrument master lookup failed for %s", trading_symbol)

        # Slow fallback: searchScrip API
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
        """Get full market data for a symbol including OI and volume.

        Uses getMarketData(FULL) which returns:
          ltp, open, high, low, close, opnInterest, exchTradVol, totBuyQuan, totSelQuan, etc.
        """
        try:
            token = symbol_info.get("symboltoken", "")
            exchange_tokens = {exchange: [token]}
            data = self._smart_api.getMarketData("FULL", exchange_tokens)
            if data and data.get("status") and data.get("data"):
                fetched = data["data"].get("fetched", [])
                if fetched:
                    return fetched[0]
        except Exception:
            pass
        # Fallback to ltpData if getMarketData fails
        try:
            token = symbol_info.get("symboltoken", "")
            tsym = symbol_info.get("tradingsymbol", "")
            data = self._smart_api.ltpData(exchange, tsym, token)
            if data and data.get("status"):
                return data.get("data", {})
        except Exception:
            pass
        return None

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

            logger.info("Fetching AngelOne instrument master for expiry discovery...")

            instruments = self._get_instrument_master()

            today = datetime.now(_IST).date()
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
