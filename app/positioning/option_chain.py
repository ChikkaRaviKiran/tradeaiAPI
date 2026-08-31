"""Live option chain snapshots from Dhan.

This is the first thing in the project that reads the CURRENT chain. Everything
else that touches open interest is either expired-contract history
(app/tools/fetch_rolling.py) or a handful of selected intraday contracts without
implied volatility. A positioning monitor cannot be built on either of those, so
this module exists to fill exactly that gap and nothing more - it fetches,
normalises and hands back. It makes no judgements about what the numbers mean.

    POST /v2/optionchain            entire chain, one expiry, OI + IV + volume
    POST /v2/optionchain/expirylist active expiries for an underlying

Rate limit is one unique request every three seconds and it is enforced here
rather than left to the caller, because the caller that forgets is the one that
gets the token throttled during market hours.

A note on `previous_oi`: the API returns PREVIOUS DAY open interest, not the
previous snapshot. Intraday change in OI - the entire point of a positioning
monitor - therefore cannot be read off a single response. It has to be computed
by differencing snapshots we stored ourselves, which is why every poll is
persisted rather than consumed and discarded.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from app.positioning.env import load_dotenv
from app.positioning.dhan import DhanError, _token, check_profile

CHAIN_URL = "https://api.dhan.co/v2/optionchain"
EXPIRY_URL = "https://api.dhan.co/v2/optionchain/expirylist"

IST = timezone(timedelta(hours=5, minutes=30))

UNDERLYINGS: dict[str, dict[str, object]] = {
    "NIFTY": {"scrip": 13, "segment": "IDX_I", "step": 50.0},
    "BANKNIFTY": {"scrip": 25, "segment": "IDX_I", "step": 100.0},
}

# Dhan's documented floor. One unique request per three seconds.
MIN_INTERVAL_S = 3.0

# How much of the chain to keep. The full response runs to hundreds of strikes,
# almost all of them untraded. Writers who matter sit near the money.
DEFAULT_WINDOW = 15

_RATE_LOCK = threading.Lock()
_last_call = 0.0
_client_id: str | None = None


def _client() -> str:
    """Client id for the header, read once from the profile the token belongs to.

    Deliberately derived rather than configured. A client id that can drift out
    of step with the token is a second thing to keep in sync and a second thing
    to get wrong.
    """
    global _client_id
    if _client_id is None:
        load_dotenv()
        env = os.environ.get("DHAN_CLIENT_ID", "").strip()
        if not env:
            try:
                from app.db.broker_credentials import get_dhan_credentials
                env = (get_dhan_credentials().get("client_id") or "").strip()
            except Exception:
                env = ""
        if not env:
            try:
                from app.core.config import settings
                env = (getattr(settings, "dhan_client_id", "") or "").strip()
            except Exception:
                env = ""
        _client_id = env or str(check_profile().get("dhanClientId") or "")
    if not _client_id:
        raise DhanError("Could not determine Dhan client id for the chain request.")
    return _client_id


def _post(url: str, body: dict, *, tries: int = 4) -> dict:
    global _last_call
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "access-token": _token(), "client-id": _client()}
    payload = json.dumps(body).encode()

    for attempt in range(tries):
        with _RATE_LOCK:
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp) or {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (429, 805) and attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise DhanError(f"optionchain HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == tries - 1:
                raise DhanError(f"optionchain unreachable: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise DhanError("optionchain failed after retries")


def expiries(symbol: str = "NIFTY") -> list[str]:
    meta = UNDERLYINGS[symbol]
    out = _post(EXPIRY_URL, {"UnderlyingScrip": meta["scrip"],
                             "UnderlyingSeg": meta["segment"]})
    return [str(d) for d in (out.get("data") or [])]


def nearest_expiry(symbol: str = "NIFTY", *, today: str | None = None) -> str:
    """The front expiry that has not passed yet.

    Expiry day itself counts as still active - positioning on expiry morning is
    the most informative chain of the week, not the least.
    """
    today = today or datetime.now(IST).date().isoformat()
    upcoming = [d for d in expiries(symbol) if d >= today]
    if not upcoming:
        raise DhanError(f"No active expiry for {symbol} on or after {today}.")
    return min(upcoming)


def floor_to_bucket(when: datetime, minutes: int = 5) -> str:
    """Snap a timestamp onto the 5-minute grid.

    Snapshots have to land on a shared grid or "compare with fifteen minutes
    ago" turns into a nearest-neighbour search that silently compares 09:47 with
    09:31 and calls it fifteen minutes.
    """
    floored = when.replace(second=0, microsecond=0)
    floored = floored.replace(minute=(floored.minute // minutes) * minutes)
    return floored.strftime("%Y-%m-%d %H:%M:00")


def _leg(raw: dict | None) -> dict:
    raw = raw or {}
    return {
        "oi": float(raw.get("oi") or 0.0),
        "prev_oi": float(raw.get("previous_oi") or 0.0),
        "volume": float(raw.get("volume") or 0.0),
        "ltp": float(raw.get("last_price") or 0.0),
        "iv": float(raw.get("implied_volatility") or 0.0),
    }


def fetch_chain(symbol: str = "NIFTY", expiry: str | None = None, *,
                window: int = DEFAULT_WINDOW,
                now: datetime | None = None) -> dict:
    """One normalised snapshot: spot plus a window of strikes either side of ATM.

    Rows with no open interest on either side are dropped. They are strikes
    nobody has taken a position in, and carrying them would let a chain that is
    mostly empty look as rich as one that is not.
    """
    meta = UNDERLYINGS[symbol]
    expiry = expiry or nearest_expiry(symbol)
    out = _post(CHAIN_URL, {"UnderlyingScrip": meta["scrip"],
                            "UnderlyingSeg": meta["segment"],
                            "Expiry": expiry})
    data = out.get("data") or {}
    spot = float(data.get("last_price") or 0.0)
    chain = data.get("oc") or {}
    if not spot or not chain:
        raise DhanError(f"Empty option chain for {symbol} {expiry}.")

    rows = []
    for key, legs in chain.items():
        try:
            strike = float(key)
        except (TypeError, ValueError):
            continue
        ce, pe = _leg(legs.get("ce")), _leg(legs.get("pe"))
        if ce["oi"] <= 0 and pe["oi"] <= 0:
            continue
        rows.append({"strike": strike, "ce": ce, "pe": pe})

    if not rows:
        raise DhanError(f"Option chain for {symbol} {expiry} carried no open interest.")

    rows.sort(key=lambda r: r["strike"])
    atm = min(rows, key=lambda r: abs(r["strike"] - spot))
    i = rows.index(atm)
    rows = rows[max(0, i - window): i + window + 1]

    return {
        "symbol": symbol,
        "expiry": expiry,
        "captured_at": floor_to_bucket(now or datetime.now(IST)),
        "spot": spot,
        "strikes": rows,
    }
