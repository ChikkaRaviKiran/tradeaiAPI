"""DB-backed broker credential store.

Why this exists
---------------
Broker tokens (Dhan client_id/access_token, Kite access_token, ...) used to
live exclusively in the ``.env`` file. Inside Docker the .env file is part
of the image build context and isn't always writable / persisted, so UI
updates appeared to "not take". Storing them in PostgreSQL means:

* Updates from the UI are immediately persisted and survive container
  restarts.
* Reads always reflect the latest value — no need to reload Settings.
* The .env file remains the bootstrap fallback for fresh installs.

The module intentionally uses a synchronous psycopg2 engine so it can be
called from sync code paths (e.g. ``DhanBroker._init_client`` runs at
process startup and from background threads). A short in-process cache
avoids hammering the DB on every order placement.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import settings

logger = logging.getLogger(__name__)

# Short TTL so a UI credential save propagates to live brokers within a
# few seconds even if the broker forgets to invalidate explicitly.
_CACHE_TTL_SECONDS = 5.0
_cache: Dict[str, Dict[str, str]] = {}
_cache_ts: Dict[str, float] = {}
_cache_lock = threading.Lock()
_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def _get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        sync_url = settings.database_url
        sync_url = sync_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
        if sync_url.startswith("postgresql://"):
            sync_url = sync_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        _engine = create_engine(sync_url, pool_pre_ping=True, pool_size=2, max_overflow=2)
    return _engine


def _ensure_table(conn) -> None:
    """Create the table if init_db hasn't run yet (defensive)."""
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS broker_credentials (
                id SERIAL PRIMARY KEY,
                broker VARCHAR(20) NOT NULL,
                key VARCHAR(50) NOT NULL,
                value TEXT,
                updated_at TIMESTAMP
            )
            """
        )
    )
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_broker_credentials_broker_key "
            "ON broker_credentials (broker, key)"
        )
    )


def get_credentials(broker: str, *, fresh: bool = False) -> Dict[str, str]:
    """Return ``{key: value}`` for a broker. Empty dict if none stored."""
    broker = broker.lower()
    now = time.time()
    if not fresh:
        with _cache_lock:
            ts = _cache_ts.get(broker, 0.0)
            if now - ts < _CACHE_TTL_SECONDS and broker in _cache:
                return dict(_cache[broker])
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            _ensure_table(conn)
            rows = conn.execute(
                text("SELECT key, value FROM broker_credentials WHERE broker = :b"),
                {"b": broker},
            ).fetchall()
        result = {r[0]: (r[1] or "") for r in rows}
    except Exception:
        logger.exception("broker_credentials: DB fetch failed for %s", broker)
        result = {}
    with _cache_lock:
        _cache[broker] = dict(result)
        _cache_ts[broker] = now
    return result


def set_credentials(broker: str, updates: Dict[str, str]) -> None:
    """Upsert one or more (key, value) pairs for a broker."""
    broker = broker.lower()
    if not updates:
        return
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        for key, value in updates.items():
            conn.execute(
                text(
                    """
                    INSERT INTO broker_credentials (broker, key, value, updated_at)
                    VALUES (:b, :k, :v, NOW())
                    ON CONFLICT (broker, key) DO UPDATE
                        SET value = EXCLUDED.value,
                            updated_at = NOW()
                    """
                ),
                {"b": broker, "k": key, "v": value},
            )
    invalidate(broker)


def invalidate(broker: Optional[str] = None) -> None:
    """Drop cached values so the next read hits the DB."""
    with _cache_lock:
        if broker is None:
            _cache.clear()
            _cache_ts.clear()
        else:
            _cache.pop(broker.lower(), None)
            _cache_ts.pop(broker.lower(), None)


# ── Convenience wrappers ─────────────────────────────────────────────────

def get_dhan_credentials(*, fresh: bool = False) -> Dict[str, str]:
    """Return ``{"client_id": ..., "access_token": ...}`` falling back to env.

    Pass ``fresh=True`` to bypass the in-process cache (used after a token
    rotation when the broker wants to be 100% certain it has the latest
    value, even within the cache window).
    """
    creds = get_credentials("dhan", fresh=fresh)
    client_id = creds.get("client_id") or settings.dhan_client_id or ""
    access_token = creds.get("access_token") or settings.dhan_access_token or ""
    return {"client_id": client_id, "access_token": access_token}


def set_dhan_credentials(*, client_id: Optional[str] = None,
                         access_token: Optional[str] = None) -> Dict[str, str]:
    """Persist Dhan creds. Only non-empty values are written."""
    updates: Dict[str, str] = {}
    if client_id is not None and client_id.strip():
        updates["client_id"] = client_id.strip()
    if access_token is not None and access_token.strip():
        updates["access_token"] = access_token.strip()
    if updates:
        set_credentials("dhan", updates)
    return get_dhan_credentials()
