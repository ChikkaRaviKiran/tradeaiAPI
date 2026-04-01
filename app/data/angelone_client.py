"""AngelOne SmartAPI client for market data fetching."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

import pandas as pd
import pyotp
import pytz
from SmartApi import SmartConnect

_IST = pytz.timezone("Asia/Kolkata")

from app.core.config import settings
from app.core.models import Candle, OptionsChainRow

if TYPE_CHECKING:
    from app.data.angelone_ws import AngelOneWebSocket

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
        self._nfo_symbol_index: dict[str, dict] = {}  # symbol -> {tradingsymbol, symboltoken}
        # WebSocket streaming (initialised via start_websocket)
        self.ws: Optional[AngelOneWebSocket] = None
        self._ws_bootstrapped: set[str] = set()  # tokens already bootstrapped

    # ── Instrument Master Cache ─────────────────────────────────────────

    def _get_instrument_master(self) -> list:
        """Download and cache the AngelOne instrument master JSON.

        Cached for the lifetime of this client (one trading day).
        Also builds an NFO symbol index for O(1) lookups.
        Retries up to 3 times with increasing timeout on failure.
        """
        if self._instrument_master is not None:
            return self._instrument_master

        import json
        from urllib.request import urlopen
        from urllib.error import URLError

        url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

        for attempt in range(3):
            timeout = 60 + (attempt * 30)  # 60s, 90s, 120s
            try:
                logger.info(
                    "Downloading AngelOne instrument master (~20MB) attempt %d/3 (timeout=%ds)...",
                    attempt + 1, timeout,
                )
                with urlopen(url, timeout=timeout) as resp:
                    self._instrument_master = json.loads(resp.read().decode())
                logger.info("Instrument master loaded: %d instruments", len(self._instrument_master))
                break
            except (URLError, TimeoutError, OSError) as e:
                logger.warning("Instrument master download attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(5)
        else:
            logger.error("Failed to download instrument master after 3 attempts")
            self._instrument_master = []
            return self._instrument_master

        # Build NFO symbol -> token index for fast option lookups
        for item in self._instrument_master:
            if item.get("exch_seg") == self.nfo_exchange:
                sym = item.get("symbol", "")
                if sym:
                    self._nfo_symbol_index[sym] = {
                        "tradingsymbol": sym,
                        "symboltoken": item.get("token", ""),
                    }
        logger.info("NFO symbol index built: %d entries", len(self._nfo_symbol_index))
        return self._instrument_master

    def get_lot_size(self, symbol_name: str) -> int | None:
        """Look up the NFO lot size for a given instrument from the instrument master.

        Args:
            symbol_name: Instrument name (e.g. "NIFTY", "FINNIFTY", "RELIANCE").

        Returns:
            Lot size as int, or None if not found.
        """
        instruments = self._get_instrument_master()
        for item in instruments:
            if (
                item.get("exch_seg") == self.nfo_exchange
                and item.get("name", "") == symbol_name
            ):
                try:
                    return int(item["lotsize"])
                except (KeyError, ValueError):
                    continue
        return None

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
            # Update WebSocket credentials so reconnects use fresh tokens
            if self.ws:
                self.ws.update_credentials(self._auth_token, self._feed_token)
            return True
        except Exception:
            logger.exception("AngelOne authentication error")
            return False

    def ensure_authenticated(self) -> None:
        """Re-authenticate if token is stale (>2 hours) or missing."""
        if (
            self._smart_api is None
            or self._last_auth is None
            or (datetime.now(_IST) - self._last_auth) > timedelta(hours=2)
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
            # Retry up to 3 times — AngelOne occasionally returns AB1004 or rate limit errors
            for attempt in range(3):
                try:
                    resp = self._smart_api.getCandleData(params)
                except Exception as api_err:
                    err_str = str(api_err)
                    if "exceeding access rate" in err_str.lower() or "rate" in err_str.lower():
                        wait = 3 + (attempt * 3)  # 3s, 6s, 9s backoff
                        logger.warning(
                            "Rate limited on candle fetch attempt %d — waiting %ds: %s",
                            attempt + 1, wait, err_str[:100],
                        )
                        time.sleep(wait)
                        continue
                    raise  # Re-raise non-rate-limit errors
                if resp and resp.get("status") is True:
                    break
                error_code = resp.get("errorcode", "") if resp else ""
                logger.warning(
                    "Candle data attempt %d failed: %s", attempt + 1, resp
                )
                # AB1004 often means session expired — force re-auth on 2nd attempt
                if attempt == 1 and error_code in ("AB1004", "AB1010"):
                    logger.info("Forcing re-authentication after repeated AB errors...")
                    self._last_auth = None  # Force ensure_authenticated to re-auth
                    self.ensure_authenticated()
                time.sleep(1 + attempt)
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

    def get_index_futures_candles(
        self,
        index_name: str = "NIFTY",
        interval: str = "ONE_MINUTE",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[Candle]:
        """Fetch current month index Futures candle data.

        Index futures have real volume, unlike the spot index (always 0).
        Used to get volume data for VWAP computation and volume analysis.
        Works for NIFTY, BANKNIFTY, FINNIFTY, etc.
        """
        token = self._get_index_fut_token(index_name)
        if not token:
            logger.warning("Could not find %s Futures token — no volume data", index_name)
            return []
        return self.get_candle_data(token, self.nfo_exchange, interval, from_date, to_date)

    def get_daily_candles(
        self,
        symbol_token: str,
        exchange: str,
        days: int = 25,
    ) -> list[Candle]:
        """Fetch daily OHLCV candles for Donchian channel computation.

        Args:
            symbol_token: AngelOne symbol token.
            exchange: Exchange (NSE, NFO).
            days: Number of calendar days to look back (default 25 covers ~20 trading days).

        Returns:
            List of daily Candle objects.
        """
        now = datetime.now(_IST)
        from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d 09:15")
        to_date = now.strftime("%Y-%m-%d 15:30")
        return self.get_candle_data(
            symbol_token, exchange, "ONE_DAY", from_date, to_date,
        )

    # Keep backward compat
    def get_nifty_futures_candles(self, **kwargs) -> list[Candle]:
        return self.get_index_futures_candles("NIFTY", **kwargs)

    def _get_index_fut_token(self, index_name: str = "NIFTY") -> Optional[str]:
        """Find the current month futures symbol token for a given index.

        Uses the instrument master JSON to find the nearest futures contract.
        """
        cache_attr = f"_fut_token_cache_{index_name}"
        if hasattr(self, cache_attr) and getattr(self, cache_attr):
            return getattr(self, cache_attr)

        try:
            instruments = self._get_instrument_master()

            today = datetime.now(_IST).date()
            best_token = None
            best_symbol = None
            best_expiry = None

            # Determine instrument type — FUTIDX for indices, FUTSTK for stocks
            fut_type = "FUTIDX"

            for inst in instruments:
                name = inst.get("name", "")
                exch_seg = inst.get("exch_seg", "")
                symbol = inst.get("symbol", "")
                expiry_str = inst.get("expiry", "")
                inst_type = inst.get("instrumenttype", "")

                if exch_seg != "NFO" or name != index_name:
                    continue
                if inst_type != fut_type:
                    continue
                if not symbol.endswith("FUT"):
                    continue
                if not expiry_str:
                    continue

                try:
                    exp_date = datetime.strptime(expiry_str, "%d%b%Y").date()
                except ValueError:
                    continue

                if exp_date < today:
                    continue

                if best_expiry is None or exp_date < best_expiry:
                    best_expiry = exp_date
                    best_token = inst.get("token", "")
                    best_symbol = symbol

            if best_token:
                setattr(self, cache_attr, best_token)
                logger.info(
                    "%s Futures token: %s (%s, expiry=%s)",
                    index_name, best_token, best_symbol, best_expiry,
                )
                return best_token

            logger.warning("No %s Futures contract found in instrument master", index_name)
        except Exception:
            logger.exception("Error fetching %s futures token", index_name)
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

    def get_option_quote(
        self, exchange: str, symbol: str, token: str
    ) -> Optional[dict]:
        """Get full quote with bid/ask/ltp for an option contract.

        Uses getMarketData(FULL) which returns depth data including
        best bid/ask prices. Falls back to ltpData if FULL fails.

        Returns:
            dict with keys: ltp, best_bid, best_ask, spread, spread_pct
            or None if fetch fails.
        """
        self.ensure_authenticated()
        try:
            exchange_tokens = {exchange: [token]}
            data = self._smart_api.getMarketData("FULL", exchange_tokens)
            if data and data.get("status") and data.get("data"):
                fetched = data["data"].get("fetched", [])
                if fetched:
                    item = fetched[0]
                    ltp = float(item.get("ltp", 0))
                    # Extract best bid/ask from depth data
                    depth = item.get("depth", {})
                    buy_depth = depth.get("buy", [])
                    sell_depth = depth.get("sell", [])
                    best_bid = float(buy_depth[0].get("price", 0)) if buy_depth else 0.0
                    best_ask = float(sell_depth[0].get("price", 0)) if sell_depth else 0.0
                    # Compute spread
                    spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
                    mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else ltp
                    spread_pct = (spread / mid * 100) if mid > 0 else 0.0
                    return {
                        "ltp": ltp,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": round(spread, 2),
                        "spread_pct": round(spread_pct, 2),
                    }
        except Exception:
            logger.warning("getMarketData(FULL) failed for %s, falling back to ltpData", symbol)
        # Fallback: ltpData (no bid/ask available)
        try:
            data = self._smart_api.ltpData(exchange, symbol, token)
            if data and data.get("status"):
                ltp = float(data["data"]["ltp"])
                return {"ltp": ltp, "best_bid": 0.0, "best_ask": 0.0, "spread": 0.0, "spread_pct": 0.0}
        except Exception:
            logger.exception("Error fetching quote for %s", symbol)
        return None

    # ── Options Chain ────────────────────────────────────────────────────

    def get_option_chain(
        self,
        expiry_date: str,
        symbol_prefix: str = "NIFTY",
        spot_price: Optional[float] = None,
        strike_interval: float = 50,
    ) -> list[OptionsChainRow]:
        """Fetch options chain for a given instrument and expiry.

        Batches token lookups then uses getMarketData(FULL) in batches
        of up to 50 tokens to minimise API calls and get real OI/volume.

        Args:
            expiry_date: Expiry in DDMMMYY format (e.g., '06MAR26').
            symbol_prefix: Option symbol prefix (e.g., 'NIFTY', 'FINNIFTY', 'RELIANCE').
            spot_price: Current spot price. If None, fetches NIFTY LTP.
            strike_interval: Strike gap (50 for NIFTY/FINNIFTY, 100 for BANKNIFTY, etc.).
        """
        self.ensure_authenticated()
        rows: list[OptionsChainRow] = []

        if spot_price is None:
            spot_price = self.get_ltp(self.exchange, self.nifty_symbol, self.nifty_token)
        if spot_price is None:
            return rows

        # Build strikes around current price (±1000 range)
        step = int(strike_interval)
        base = int(round(spot_price / strike_interval) * strike_interval)
        strikes = list(range(base - 1000, base + 1000 + step, step))

        # Phase 1: Search all symbol tokens (uses instrument master — no API calls)
        token_map: dict[str, dict] = {}  # token -> symbol_info
        strike_tokens: dict[int, dict] = {}  # strike -> {ce_token, pe_token}
        for strike in strikes:
            ce_symbol = f"{symbol_prefix}{expiry_date}{strike}CE"
            pe_symbol = f"{symbol_prefix}{expiry_date}{strike}PE"
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
        """Search for a symbol token using the O(1) NFO index."""
        # Ensure instrument master (and index) is loaded
        self._get_instrument_master()
        result = self._nfo_symbol_index.get(trading_symbol)
        if result:
            return result
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
            expiry: Expiry in DDMMMYY (e.g. '06MAR26').
            strike: Strike price.
            option_type: 'CE' or 'PE'.
        """
        symbol = f"NIFTY{expiry}{int(strike)}{option_type}"
        return self._search_symbol(symbol)

    # ── Expiry Discovery ─────────────────────────────────────────────────

    def get_nearest_weekly_expiry(self, instrument_name: str = "NIFTY") -> Optional[str]:
        """Find the nearest valid weekly expiry for an instrument.

        Uses the OpenAPI instrument master, filters for the given instrument's
        option contracts, and returns the nearest expiry in DDMMMYY format.

        Args:
            instrument_name: Instrument name in the master (e.g. 'NIFTY', 'FINNIFTY', 'BANKNIFTY').
        """
        self.ensure_authenticated()
        try:
            logger.info("Fetching AngelOne instrument master for %s expiry discovery...", instrument_name)

            instruments = self._get_instrument_master()

            today = datetime.now(_IST).date()
            expiry_dates: set[datetime] = set()

            for inst in instruments:
                name = inst.get("name", "")
                exch_seg = inst.get("exch_seg", "")
                symbol = inst.get("symbol", "")
                expiry_str = inst.get("expiry", "")

                if exch_seg != "NFO" or name != instrument_name:
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
                logger.warning("No %s expiry dates found in instrument master", instrument_name)
                return None

            nearest = min(expiry_dates)
            result = nearest.strftime("%d%b%y").upper()
            logger.info("Nearest %s weekly expiry from instrument master: %s", instrument_name, result)
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
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.sort_index(inplace=True)
        # Remove duplicate timestamps (keep first occurrence)
        if df.index.has_duplicates:
            n_dups = df.index.duplicated().sum()
            logger.warning("Dropping %d duplicate timestamp candles", n_dups)
            df = df[~df.index.duplicated(keep="first")]
        return df

    # ── WebSocket Streaming ──────────────────────────────────────────────

    def start_websocket(self) -> bool:
        """Create and connect WebSocket using current auth credentials.

        Call after authenticate() succeeds.  Returns True if connected.
        """
        if not self._auth_token or not self._feed_token:
            logger.error("Cannot start WebSocket — not authenticated")
            return False

        from app.data.angelone_ws import AngelOneWebSocket

        self.ws = AngelOneWebSocket(
            auth_token=self._auth_token,
            api_key=settings.angelone_api_key,
            client_code=settings.angelone_client_id,
            feed_token=self._feed_token,
        )
        ok = self.ws.connect()
        if ok:
            logger.info("WebSocket streaming started")
        else:
            logger.warning("WebSocket connection failed — will use API polling fallback")
            self.ws = None
        return ok

    def stop_websocket(self) -> None:
        if self.ws:
            self.ws.disconnect()
            self.ws = None

    def subscribe_instrument(
        self, spot_token: str, spot_exchange: str, futures_token: Optional[str] = None,
    ) -> None:
        """Subscribe to spot (and optionally futures) tokens for streaming.

        Tokens are subscribed in Quote mode so we get LTP + volume for
        candle aggregation.
        """
        if not self.ws or not self.ws.is_connected:
            return
        from app.data.angelone_ws import AngelOneWebSocket

        self.ws.subscribe(spot_exchange, [spot_token], AngelOneWebSocket.MODE_QUOTE)

        if futures_token:
            self.ws.subscribe(self.nfo_exchange, [futures_token], AngelOneWebSocket.MODE_QUOTE)

    def bootstrap_ws_candles(
        self, token: str, exchange: str, from_date: str, to_date: str,
    ) -> None:
        """Fetch historical candles via API and seed the WebSocket CandleBuilder.

        Called once per token at the start of the trading day so the
        builder already contains previous-day data needed for indicator
        warmup (EMA200, etc.).
        """
        if not self.ws:
            return

        cache_key = f"{exchange}:{token}"
        if cache_key in self._ws_bootstrapped:
            return

        candles = self.get_candle_data(token, exchange, "ONE_MINUTE", from_date, to_date)
        df = self.candles_to_dataframe(candles)
        if not df.empty:
            self.ws.bootstrap_candles(exchange, token, df)
            self._ws_bootstrapped.add(cache_key)

    def get_live_candles(self, token: str, exchange: str) -> pd.DataFrame:
        """Get 1-min candle DataFrame from WebSocket if available.

        Falls through to empty DataFrame when WebSocket is not connected
        or token is not subscribed, allowing the caller to use the API
        fallback.
        """
        if self.ws and self.ws.is_connected:
            df = self.ws.get_candles_df(exchange, token)
            if not df.empty:
                return df
        return pd.DataFrame()

    def get_live_futures_candles(self, index_name: str) -> pd.DataFrame:
        """Get 1-min futures candle DataFrame from WebSocket."""
        if not self.ws or not self.ws.is_connected:
            return pd.DataFrame()

        token = self._get_index_fut_token(index_name)
        if not token:
            return pd.DataFrame()

        return self.ws.get_candles_df(self.nfo_exchange, token)
