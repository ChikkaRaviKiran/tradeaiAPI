"""AngelOne WebSocket 2.0 client for real-time market data streaming.

Replaces repeated getCandleData API polling with a persistent WebSocket
connection that streams tick data.  Ticks are aggregated into 1-minute
OHLCV candles via CandleBuilder, producing DataFrames identical to those
from AngelOneClient.get_candle_data().

Key details (AngelOne WebSocket 2.0):
  - URL: wss://smartapisocket.angelone.in/smart-stream
  - Auth: headers Authorization, x-api-key, x-client-code, x-feed-token
  - Heartbeat: send "ping" text every 25s, expect "pong"
  - Modes: 1=LTP(51 bytes), 2=Quote(≥75 bytes), 3=SnapQuote
  - Response: binary Little-Endian; prices in paise (÷100)
  - Limit: 1000 token+mode subscriptions per session
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import websocket

_IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)


# ── Candle Aggregator ────────────────────────────────────────────────────


class CandleBuilder:
    """Aggregates streaming ticks into 1-minute OHLCV candles.

    Volume is derived from the cumulative 'volume traded for the day'
    field sent by the exchange: per-bar volume = delta between the first
    tick of the minute and the last tick of the previous minute.
    """

    def __init__(self) -> None:
        self._current_minute: Optional[datetime] = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume_at_minute_start: int = 0
        self._last_cumulative_volume: int = 0
        self._history: list[dict] = []  # finalised candle dicts
        self._lock = threading.Lock()

    # -- tick ingestion --------------------------------------------------

    def on_tick(self, ltp: float, cumulative_volume: int, ts: datetime) -> None:
        """Process an inbound tick."""
        minute = ts.replace(second=0, microsecond=0)

        with self._lock:
            if self._current_minute is None or minute > self._current_minute:
                # New minute → finalise the previous bar (if any)
                if self._current_minute is not None:
                    self._finalize_bar()
                self._current_minute = minute
                self._open = ltp
                self._high = ltp
                self._low = ltp
                self._close = ltp
                self._volume_at_minute_start = cumulative_volume
            else:
                # Same minute → update running bar
                self._high = max(self._high, ltp)
                self._low = min(self._low, ltp)
                self._close = ltp

            self._last_cumulative_volume = cumulative_volume

    def _finalize_bar(self) -> None:
        vol = self._last_cumulative_volume - self._volume_at_minute_start
        self._history.append(
            {
                "timestamp": self._current_minute,
                "open": self._open,
                "high": self._high,
                "low": self._low,
                "close": self._close,
                "volume": max(vol, 0),
            }
        )

    # -- accessors -------------------------------------------------------

    def get_current_bar(self) -> Optional[dict]:
        with self._lock:
            if self._current_minute is None:
                return None
            vol = self._last_cumulative_volume - self._volume_at_minute_start
            return {
                "timestamp": self._current_minute,
                "open": self._open,
                "high": self._high,
                "low": self._low,
                "close": self._close,
                "volume": max(vol, 0),
            }

    def get_history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def set_history(self, candles: list[dict]) -> None:
        """Load bootstrap candles (from getCandleData at startup).

        Resets the current-bar state so the next tick starts a fresh
        bar instead of finalising stale data from before bootstrap.
        """
        with self._lock:
            self._history = list(candles)
            self._current_minute = None
            self._open = 0.0
            self._high = 0.0
            self._low = 0.0
            self._close = 0.0
            self._volume_at_minute_start = 0
            self._last_cumulative_volume = 0

    @property
    def last_price(self) -> float:
        with self._lock:
            return self._close


# ── WebSocket Client ─────────────────────────────────────────────────────


class AngelOneWebSocket:
    """Persistent WebSocket 2.0 connection to AngelOne smart-stream."""

    WS_URL = "wss://smartapisocket.angelone.in/smart-stream"
    HEARTBEAT_SEC = 25

    # Exchange type codes
    EXCHANGE_MAP = {"NSE": 1, "NFO": 2, "BSE": 3, "MCX": 5}

    # Subscription modes
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_SNAP = 3

    # If no tick received for this many seconds during market hours, consider stale
    STALE_THRESHOLD_SEC = 90

    def __init__(
        self,
        auth_token: str,
        api_key: str,
        client_code: str,
        feed_token: str,
    ) -> None:
        self._auth_token = auth_token
        self._api_key = api_key
        self._client_code = client_code
        self._feed_token = feed_token

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._running = False

        # candle_builders: (exchange_type, token) → CandleBuilder
        self._candle_builders: dict[tuple[int, str], CandleBuilder] = {}
        # ltp_cache: (exchange_type, token) → (ltp, timestamp)
        self._ltp_cache: dict[tuple[int, str], tuple[float, datetime]] = {}
        self._lock = threading.Lock()

        # track subscriptions for auto-resubscribe on reconnect
        self._subscriptions: set[tuple[int, str, int]] = set()

        # Staleness tracking: time of last received tick
        self._last_tick_time: Optional[datetime] = None
        self._cred_lock = threading.Lock()

    # ── connection lifecycle ───────────────────────────────────────────

    def connect(self) -> bool:
        """Open WebSocket in a daemon thread.  Returns True on success."""
        if self._running:
            return True

        self._running = True
        headers = {
            "Authorization": self._auth_token,
            "x-api-key": self._api_key,
            "x-client-code": self._client_code,
            "x-feed-token": self._feed_token,
        }
        self._ws = websocket.WebSocketApp(
            self.WS_URL,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(
            target=self._run_forever, daemon=True, name="angelone-ws"
        )
        self._ws_thread.start()

        if self._connected.wait(timeout=15):
            logger.info("AngelOne WebSocket connected")
            return True

        logger.error("AngelOne WebSocket connection timed out")
        self._running = False
        return False

    def disconnect(self) -> None:
        self._running = False
        self._connected.clear()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("AngelOne WebSocket disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def update_credentials(
        self, auth_token: str, feed_token: str,
    ) -> None:
        """Update auth credentials so the next reconnect uses fresh tokens."""
        with self._cred_lock:
            self._auth_token = auth_token
            self._feed_token = feed_token
        logger.info("WebSocket credentials updated for next reconnect")

    @property
    def last_tick_time(self) -> Optional[datetime]:
        """Timestamp of last received binary tick (for staleness detection)."""
        with self._lock:
            return self._last_tick_time

    @property
    def is_stale(self) -> bool:
        """True if connected but no ticks received recently."""
        if not self._connected.is_set():
            return False
        with self._lock:
            if self._last_tick_time is None:
                return False
            elapsed = (datetime.now(_IST) - self._last_tick_time).total_seconds()
        return elapsed > self.STALE_THRESHOLD_SEC

    def force_reconnect(self) -> None:
        """Force close the current WebSocket so _run_forever reconnects."""
        logger.warning("Force-reconnecting WebSocket...")
        self._connected.clear()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    # ── internal WS loop + heartbeat ───────────────────────────────────

    def _run_forever(self) -> None:
        reconnect_delay = 5
        max_delay = 60
        while self._running:
            try:
                self._ws.run_forever(
                    ping_interval=0, skip_utf8_validation=True
                )
            except Exception:
                logger.exception("WebSocket run_forever error")

            if self._running:
                self._connected.clear()
                logger.warning(
                    "WebSocket disconnected — reconnecting in %d s", reconnect_delay
                )
                time.sleep(reconnect_delay)
                # Exponential backoff capped at max_delay
                reconnect_delay = min(reconnect_delay * 2, max_delay)

                # Use latest credentials (may have been refreshed by client)
                with self._cred_lock:
                    headers = {
                        "Authorization": self._auth_token,
                        "x-api-key": self._api_key,
                        "x-client-code": self._client_code,
                        "x-feed-token": self._feed_token,
                    }
                self._ws = websocket.WebSocketApp(
                    self.WS_URL,
                    header=headers,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

    def _start_heartbeat(self) -> None:
        def _heartbeat() -> None:
            consecutive_failures = 0
            while self._running and self._connected.is_set():
                try:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send("ping")
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                except Exception:
                    consecutive_failures += 1
                    logger.debug("Heartbeat send failed (consecutive=%d)", consecutive_failures)

                # If heartbeat fails 3 times in a row, force reconnect
                if consecutive_failures >= 3:
                    logger.warning("Heartbeat failed %d times — forcing reconnect", consecutive_failures)
                    self.force_reconnect()
                    break
                time.sleep(self.HEARTBEAT_SEC)

        t = threading.Thread(target=_heartbeat, daemon=True, name="angelone-ws-hb")
        t.start()

    # ── WS callbacks ───────────────────────────────────────────────────

    def _on_open(self, ws) -> None:
        logger.info("WebSocket opened")
        self._connected.set()
        # Reset reconnect backoff on successful connection
        self._start_heartbeat()
        if self._subscriptions:
            self._resubscribe_all()

    def _on_error(self, ws, error) -> None:
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, status_code, msg) -> None:
        logger.warning("WebSocket closed (code=%s msg=%s)", status_code, msg)
        self._connected.clear()

    def _on_message(self, ws, message) -> None:
        if isinstance(message, str):
            if message == "pong":
                return
            try:
                data = json.loads(message)
                if "errorCode" in data:
                    logger.warning("WebSocket error frame: %s", data)
            except json.JSONDecodeError:
                pass
            return

        # binary tick
        try:
            self._parse_binary(message)
        except Exception:
            logger.debug("Tick parse failed (%d bytes)", len(message))

    # ── binary response parsing ────────────────────────────────────────

    def _parse_binary(self, data: bytes) -> None:
        if len(data) < 51:
            return

        mode = struct.unpack_from("<b", data, 0)[0]
        exch_type = struct.unpack_from("<b", data, 1)[0]

        # token: 25 bytes, null-terminated UTF-8
        token_raw = data[2:27]
        token = token_raw.split(b"\x00")[0].decode("utf-8", errors="ignore").strip()

        # exchange timestamp (epoch ms) — may be 0 for indices
        ts_ms = struct.unpack_from("<q", data, 35)[0]
        if ts_ms <= 0:
            ts = datetime.now(_IST)
        else:
            try:
                ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=_IST)
            except (OSError, OverflowError, ValueError):
                ts = datetime.now(_IST)

        # LTP (int64, paise → rupees)
        ltp = struct.unpack_from("<q", data, 43)[0] / 100.0

        # cumulative volume (Quote mode, offset 67)
        volume = 0
        if mode >= self.MODE_QUOTE and len(data) >= 75:
            volume = struct.unpack_from("<q", data, 67)[0]

        key = (exch_type, token)

        with self._lock:
            self._ltp_cache[key] = (ltp, ts)
            self._last_tick_time = datetime.now(_IST)

        builder = self._candle_builders.get(key)
        if builder and mode >= self.MODE_QUOTE:
            builder.on_tick(ltp, volume, ts)

    # ── subscribe / unsubscribe ────────────────────────────────────────

    def subscribe(
        self, exchange: str, tokens: list[str], mode: int = MODE_QUOTE
    ) -> None:
        exch_type = self.EXCHANGE_MAP.get(exchange, 1)

        for t in tokens:
            self._subscriptions.add((exch_type, t, mode))
            if mode >= self.MODE_QUOTE:
                key = (exch_type, t)
                if key not in self._candle_builders:
                    self._candle_builders[key] = CandleBuilder()

        if not self._connected.is_set():
            logger.info("WS not yet connected — subscription queued (%s %s)", exchange, tokens)
            return

        self._send_sub_action(1, exch_type, tokens, mode)
        logger.info("Subscribed: %s tokens=%s mode=%d", exchange, tokens, mode)

    def unsubscribe(self, exchange: str, tokens: list[str], mode: int = MODE_QUOTE) -> None:
        exch_type = self.EXCHANGE_MAP.get(exchange, 1)
        for t in tokens:
            self._subscriptions.discard((exch_type, t, mode))

        if self._connected.is_set():
            self._send_sub_action(0, exch_type, tokens, mode)
            logger.info("Unsubscribed: %s tokens=%s mode=%d", exchange, tokens, mode)

    def _send_sub_action(self, action: int, exch_type: int, tokens: list[str], mode: int) -> None:
        msg = {
            "correlationID": f"{'sub' if action else 'unsub'}_{int(time.time())}",
            "action": action,
            "params": {
                "mode": mode,
                "tokenList": [{"exchangeType": exch_type, "tokens": tokens}],
            },
        }
        try:
            self._ws.send(json.dumps(msg))
        except Exception:
            logger.warning("Failed to send subscription message")

    def _resubscribe_all(self) -> None:
        grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
        for exch_type, token, mode in self._subscriptions:
            grouped[(exch_type, mode)].append(token)

        for (exch_type, mode), tokens in grouped.items():
            self._send_sub_action(1, exch_type, tokens, mode)

        logger.info(
            "Re-subscribed %d token+mode pairs after reconnect",
            len(self._subscriptions),
        )

    # ── data access (thread-safe) ──────────────────────────────────────

    def get_ltp(self, exchange: str, token: str) -> Optional[float]:
        key = (self.EXCHANGE_MAP.get(exchange, 1), token)
        with self._lock:
            cached = self._ltp_cache.get(key)
        return cached[0] if cached else None

    def get_candle_builder(self, exchange: str, token: str) -> Optional[CandleBuilder]:
        key = (self.EXCHANGE_MAP.get(exchange, 1), token)
        return self._candle_builders.get(key)

    def bootstrap_candles(self, exchange: str, token: str, df: pd.DataFrame) -> None:
        """Seed a CandleBuilder with historical candle data (DataFrame)."""
        key = (self.EXCHANGE_MAP.get(exchange, 1), token)
        builder = self._candle_builders.get(key)
        if builder is None:
            builder = CandleBuilder()
            self._candle_builders[key] = builder

        candles = []
        for ts, row in df.iterrows():
            candles.append(
                {
                    "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row.get("volume", 0)),
                }
            )
        builder.set_history(candles)
        logger.info("Bootstrapped %d candles for %s/%s", len(candles), exchange, token)

    def get_candles_df(self, exchange: str, token: str) -> pd.DataFrame:
        """Return a DataFrame of history + current bar (same format as candles_to_dataframe)."""
        builder = self.get_candle_builder(exchange, token)
        if builder is None:
            return pd.DataFrame()

        bars = builder.get_history()
        current = builder.get_current_bar()
        if current:
            bars = bars + [current]

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.sort_index(inplace=True)
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        return df
