"""Lightweight Redis cache with safe fallback.

Design notes
------------
- All operations are best-effort. Any Redis error is swallowed and treated as
  a cache miss / no-op. The application MUST behave correctly even if Redis
  is unreachable.
- Single shared async client built lazily on first use.
- JSON-only payloads (Pydantic models should be converted to dicts by callers).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as _redis_async  # type: ignore
except Exception:  # pragma: no cover
    _redis_async = None  # type: ignore

_client: Any = None
_init_lock = asyncio.Lock()
_disabled = False


async def _get_client() -> Any:
    """Return a connected Redis client or None if unavailable."""
    global _client, _disabled
    if _disabled or _redis_async is None:
        return None
    if _client is not None:
        return _client
    async with _init_lock:
        if _client is not None:
            return _client
        try:
            client = _redis_async.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            # Ping once to verify connectivity
            await client.ping()
            _client = client
            logger.info("Redis cache connected: %s", settings.redis_url)
        except Exception as e:
            logger.warning("Redis cache unavailable (%s) — running without cache", e)
            _disabled = True
            return None
    return _client


async def get_json(key: str) -> Optional[Any]:
    client = await _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("Cache GET failed for %s: %s", key, e)
        return None


async def set_json(key: str, value: Any, ttl_seconds: int) -> None:
    client = await _get_client()
    if client is None:
        return
    try:
        await client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
    except Exception as e:
        logger.debug("Cache SET failed for %s: %s", key, e)


async def delete(*keys: str) -> None:
    if not keys:
        return
    client = await _get_client()
    if client is None:
        return
    try:
        await client.delete(*keys)
    except Exception as e:
        logger.debug("Cache DELETE failed for %s: %s", keys, e)


async def delete_prefix(prefix: str) -> None:
    """Delete all keys matching a prefix using SCAN (non-blocking)."""
    client = await _get_client()
    if client is None:
        return
    try:
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=f"{prefix}*", count=200)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
    except Exception as e:
        logger.debug("Cache delete_prefix failed for %s: %s", prefix, e)


async def cached(
    key: str,
    ttl_seconds: int,
    loader: Callable[[], Awaitable[Any]],
) -> Any:
    """Get value from cache; on miss, call loader(), store result, return it.

    If Redis is unavailable, simply calls loader() directly.
    """
    cached_val = await get_json(key)
    if cached_val is not None:
        return cached_val
    value = await loader()
    if value is not None:
        await set_json(key, value, ttl_seconds)
    return value
