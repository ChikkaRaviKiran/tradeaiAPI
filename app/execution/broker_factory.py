"""Build a fully-wired broker adapter from a ``BrokerAccount`` DB row.

This is the bridge between the multi-account UI and the existing
BaseBroker adapters (AngelOneBroker / KiteBroker / DhanBroker).

Usage — synchronous, safe to call from any thread::

    from app.execution.broker_factory import build_broker_from_account_id

    broker = build_broker_from_account_id(42)  # None if not found / bad creds

The returned broker owns a per-account client (its own credentials, its
own token cache) and is completely isolated from the process-wide
singleton adapters that read from environment variables.

Design notes:

- We keep a small in-process cache keyed by ``(account_id, updated_at)``
  so repeated ``build_broker_from_account_id(id)`` calls (e.g. every
  cycle from the registry) reuse the same authenticated client. When a
  user edits credentials, ``updated_at`` changes and the cache entry is
  invalidated on the next lookup.
- Failures never raise — this function is called on hot paths and must
  degrade to ``None`` (which downstream treats as "alert-only" mode).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from app.execution.broker_base import BaseBroker

logger = logging.getLogger(__name__)

# Cache: account_id -> (updated_at_iso, broker)
_cache_lock = threading.RLock()
_cache: dict[int, tuple[str, BaseBroker]] = {}


def build_broker_from_account_id(account_id: int) -> Optional[BaseBroker]:
    """Return a broker adapter wired to the given account, or None."""
    if not account_id:
        return None
    row = _fetch_account_row_sync(int(account_id))
    if row is None:
        logger.warning("[BrokerFactory] account_id=%s not found", account_id)
        return None
    return _build_from_row(row)


def invalidate_account_cache(account_id: Optional[int] = None) -> None:
    """Drop cached brokers. Called after credential UI saves."""
    with _cache_lock:
        if account_id is None:
            _cache.clear()
        else:
            _cache.pop(int(account_id), None)


def _build_from_row(row: dict) -> Optional[BaseBroker]:
    account_id = int(row["id"])
    updated_at = str(row.get("updated_at") or "")
    with _cache_lock:
        cached = _cache.get(account_id)
        if cached and cached[0] == updated_at:
            return cached[1]

    broker_name = (row.get("broker") or "").lower().strip()
    try:
        if broker_name == "angel":
            broker = _build_angel(row)
        elif broker_name == "kite":
            broker = _build_kite(row)
        elif broker_name == "dhan":
            broker = _build_dhan(row)
        else:
            logger.warning("[BrokerFactory] Unknown broker '%s' for account %s", broker_name, account_id)
            return None
    except Exception:
        logger.exception("[BrokerFactory] Failed building broker for account %s", account_id)
        return None

    if broker is None:
        return None

    with _cache_lock:
        _cache[account_id] = (updated_at, broker)
    return broker


def _build_angel(row: dict):
    from app.data.angelone_client import AngelOneClient
    from app.execution.angelone_broker import AngelOneBroker

    creds = {
        "api_key": row.get("api_key") or "",
        "client_id": row.get("client_id") or "",
        "mpin": row.get("mpin") or "",
        "password": row.get("password") or "",
        "totp_secret": row.get("totp_secret") or "",
    }
    if not creds["api_key"] or not creds["client_id"]:
        logger.warning(
            "[BrokerFactory] Angel account %s missing api_key/client_id",
            row.get("id"),
        )
        return None
    client = AngelOneClient(credentials=creds)
    return AngelOneBroker(client=client)


def _build_kite(row: dict):
    from app.data.kite_client import KiteClient
    from app.execution.kite_broker import KiteBroker

    api_key = row.get("api_key") or ""
    access_token = row.get("access_token") or ""
    if not api_key:
        logger.warning("[BrokerFactory] Kite account %s missing api_key", row.get("id"))
        return None
    client = KiteClient(
        api_key=api_key,
        access_token=access_token,
        account_name=row.get("name") or f"kite-{row.get('id')}",
        proxy_url=row.get("proxy_url") or "",
    )
    return KiteBroker(client=client)


def _build_dhan(row: dict):
    from app.data.dhan_client import DhanClient
    from app.execution.dhan_broker import DhanBroker

    client_id = row.get("client_id") or ""
    access_token = row.get("access_token") or ""
    if not client_id or not access_token:
        logger.warning(
            "[BrokerFactory] Dhan account %s missing client_id/access_token",
            row.get("id"),
        )
        return None
    client = DhanClient(client_id=client_id, access_token=access_token)
    return DhanBroker(client=client)


def _fetch_account_row_sync(account_id: int) -> Optional[dict]:
    """Synchronously fetch a BrokerAccount row → plain dict.

    Runs its own asyncio loop in a background thread so it can be called
    from sync contexts (scanner init, registry spawn) without needing an
    event loop.  Returns None on any error.
    """
    import asyncio
    import concurrent.futures

    async def _fetch() -> Optional[dict]:
        try:
            from sqlalchemy import select
            from app.db.account_models import BrokerAccount
            from app.db.models import AsyncSessionLocal

            async with AsyncSessionLocal() as s:
                row = (
                    await s.execute(
                        select(BrokerAccount).where(BrokerAccount.id == account_id)
                    )
                ).scalar_one_or_none()
                if row is None:
                    return None
                return {
                    "id": row.id,
                    "name": row.name,
                    "broker": row.broker,
                    "client_id": row.client_id,
                    "api_key": row.api_key,
                    "api_secret": row.api_secret,
                    "password": row.password,
                    "mpin": row.mpin,
                    "totp_secret": row.totp_secret,
                    "access_token": row.access_token,
                    "refresh_token": row.refresh_token,
                    "proxy_url": row.proxy_url,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else "",
                }
        except Exception:
            logger.debug("_fetch_account_row_sync failed", exc_info=True)
            return None

    def _run_in_thread() -> Optional[dict]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_fetch())
        finally:
            loop.close()

    try:
        # Try synchronously first (no running loop → fine)
        return asyncio.run(_fetch())
    except RuntimeError:
        # A loop is already running in this thread — fall back to a
        # background thread with its own loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_in_thread)
            return fut.result(timeout=15)
