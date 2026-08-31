"""Dhan authentication helpers used by the option-chain fetcher.

Only the pieces the Market Story needs: the token, and a profile call to prove
it works and to discover the client id the chain endpoint wants in a header.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from app.positioning.env import load_dotenv

PROFILE_URL = "https://api.dhan.co/v2/profile"


class DhanError(RuntimeError):
    pass


def _token() -> str:
    load_dotenv()
    token = os.environ.get("DHAN_ACCESS_TOKEN", "").strip()
    if not token:
        # Resolve through the platform's own credential chain (broker_accounts
        # table -> legacy broker_credentials -> settings) so a token rotated in
        # the Settings UI is picked up here too, instead of going stale.
        try:
            from app.db.broker_credentials import get_dhan_credentials
            token = (get_dhan_credentials().get("access_token") or "").strip()
        except Exception:
            token = ""
    if not token:
        try:
            from app.core.config import settings
            token = (getattr(settings, "dhan_access_token", "") or "").strip()
        except Exception:
            token = ""
    if not token:
        raise DhanError(
            "DHAN_ACCESS_TOKEN is not set.\n"
            "Set it one of two ways - never paste it into a source file:\n"
            "  1. Put DHAN_ACCESS_TOKEN=<jwt> in the .env file at the project root\n"
            '  2. $env:DHAN_ACCESS_TOKEN = "<jwt>"  for one terminal session'
        )
    return token


def check_profile(token: str | None = None) -> dict:
    """Preflight: confirm the token works and the Data API plan is active.

    Without an active plan the fetch fails in ways that look like a code bug,
    so check it explicitly first.
    """
    req = urllib.request.Request(
        PROFILE_URL,
        headers={"Accept": "application/json", "access-token": token or _token()},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 - fixed https host
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        if exc.code in (401, 403):
            raise DhanError(
                f"Token rejected ({exc.code}). DhanHQ tokens expire after 24h - "
                f"generate a fresh one. Detail: {detail}"
            ) from exc
        raise DhanError(f"Profile check failed, HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DhanError(f"Could not reach Dhan: {exc.reason}") from exc
