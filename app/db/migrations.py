"""One-shot bootstrap migrations for the multi-account/multi-strategy schema.

Called from ``app.db.models.init_db`` after ``Base.metadata.create_all``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.db.account_models import BrokerAccount, StrategyInstance
from app.db.models import AsyncSessionLocal

logger = logging.getLogger(__name__)


def _atl_settings_path() -> str:
    backend_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    return os.path.join(backend_root, "data", "atl_straddle_settings.json")


def _load_atl_settings_snapshot() -> dict[str, Any] | None:
    path = _atl_settings_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.exception("Failed to read atl_straddle_settings.json for migration")
    return None


async def _ensure_broker_accounts(session) -> BrokerAccount | None:
    """Create one row per currently-configured broker (from settings/env)
    if the ``broker_accounts`` table is empty.  Returns the primary account.
    """
    existing = (await session.execute(select(BrokerAccount))).scalars().first()
    if existing is not None:
        # Table already has data — leave it alone.
        primary = (
            await session.execute(
                select(BrokerAccount).where(BrokerAccount.is_primary.is_(True))
            )
        ).scalars().first()
        return primary or existing

    primary_pref = (settings.trading_account or "angel").lower()
    seeds: list[BrokerAccount] = []

    if settings.angelone_client_id:
        seeds.append(
            BrokerAccount(
                name="AngelOne",
                broker="angel",
                client_id=settings.angelone_client_id,
                api_key=settings.angelone_api_key or "",
                password=settings.angelone_password or "",
                mpin=settings.angelone_mpin or "",
                totp_secret=settings.angelone_totp_secret or "",
                login_method="manual",
                is_active=True,
                paper_trading=False,
                is_primary=(primary_pref == "angel"),
            )
        )

    kite_api_key = getattr(settings, "kite_api_key", "") or ""
    kite_client_id = getattr(settings, "kite_client_id", "") or ""
    if kite_api_key or kite_client_id:
        seeds.append(
            BrokerAccount(
                name="Kite",
                broker="kite",
                client_id=kite_client_id or "KITE",
                api_key=kite_api_key,
                api_secret=getattr(settings, "kite_api_secret", "") or "",
                access_token=getattr(settings, "kite_access_token", "") or "",
                proxy_url=getattr(settings, "kite_proxy_url", "") or "",
                login_method="oauth",
                is_active=True,
                paper_trading=False,
                is_primary=(primary_pref == "kite"),
            )
        )

    dhan_client_id = getattr(settings, "dhan_client_id", "") or ""
    dhan_access_token = getattr(settings, "dhan_access_token", "") or ""
    if dhan_client_id or dhan_access_token:
        seeds.append(
            BrokerAccount(
                name="Dhan",
                broker="dhan",
                client_id=dhan_client_id or "DHAN",
                access_token=dhan_access_token,
                login_method="manual",
                is_active=True,
                paper_trading=False,
                is_primary=(primary_pref == "dhan"),
            )
        )

    if not seeds:
        # No env-configured broker → drop a paper-trading placeholder so
        # the UI has *something* to show and the strategy row can bind.
        seeds.append(
            BrokerAccount(
                name="Paper",
                broker="angel",
                client_id="PAPER",
                login_method="manual",
                is_active=True,
                paper_trading=True,
                is_primary=True,
            )
        )

    # Guarantee exactly one primary.
    if not any(a.is_primary for a in seeds):
        seeds[0].is_primary = True

    session.add_all(seeds)
    await session.flush()

    primary = next((a for a in seeds if a.is_primary), seeds[0])
    logger.info(
        "Seeded %d broker_accounts rows (primary=%s/%s)",
        len(seeds),
        primary.broker,
        primary.name,
    )
    return primary


async def _ensure_strategy_instances(session, primary: BrokerAccount | None) -> None:
    """Create one seed StrategyInstance row from atl_straddle_settings.json
    if the table is empty.
    """
    existing = (
        await session.execute(select(StrategyInstance))
    ).scalars().first()
    if existing is not None:
        return

    snap = _load_atl_settings_snapshot() or {}
    strike_mode = str(snap.get("strike_mode", "ATM")).upper()
    if strike_mode == "MAXPAIN":
        strategy_type = "MAXPAIN_ROLL"
    elif strike_mode == "STRANGLE":
        strategy_type = "OTM_STRANGLE"
    else:
        strategy_type = "ATM_STRADDLE"
    hedge_mode = str(snap.get("hedge_mode", "none")).lower()
    if hedge_mode not in {"none", "premium", "otm_points"}:
        hedge_mode = "none"

    row = StrategyInstance(
        strategy_type=strategy_type,
        account_id=primary.id if primary else None,
        index=str(snap.get("index", "NIFTY")).upper(),
        trading_day=str(snap.get("trading_day", "Daily")).title(),
        entry_time=str(snap.get("entry_time", "09:20")),
        exit_time=str(snap.get("exit_time", "15:15")),
        lots=int(snap.get("lots", 1) or 1),
        strike_interval=int(snap.get("strike_interval", 50) or 50),
        strike_mode=strike_mode if strike_mode in {"ATM", "STRANGLE", "ITM", "MAXPAIN"} else "ATM",
        otm_strikes=int(snap.get("otm_strikes", 0) or 0),
        static_legs=bool(snap.get("static_legs", False)),
        adjustment_points=int(snap.get("adjustment_points", 1) or 1),
        rolling_points=int(snap.get("rolling_points", 300) or 300),
        sl_type=str(snap.get("sl_type", "none")).lower(),
        sl_lower=float(snap.get("sl_lower", 0) or 0),
        sl_upper=float(snap.get("sl_upper", 0) or 0),
        first_straddle_sl_pct=int(snap.get("first_straddle_sl_pct", 100) or 100),
        reform_straddle_sl_pct=int(snap.get("reform_straddle_sl_pct", 60) or 60),
        hedge_mode=hedge_mode,
        hedge_premium=float(snap.get("hedge_premium", 3) or 3),
        hedge_otm_points=int(snap.get("hedge_otm_points", 500) or 500),
        hedge_lots=int(snap.get("hedge_lots", 0) or 0),
        is_active=bool(snap.get("enabled", False)),
        live_execution=False,
        display_name="Migrated from atl_straddle_settings.json",
    )
    session.add(row)
    await session.flush()
    logger.info(
        "Seeded strategy_instances row id=%s (%s / %s / account_id=%s)",
        row.id, row.strategy_type, row.index, row.account_id,
    )


async def seed_multi_account_defaults() -> None:
    """Idempotent bootstrap — safe to call on every startup."""
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                primary = await _ensure_broker_accounts(session)
                await _ensure_strategy_instances(session, primary)
    except Exception:
        # Never block app startup on migration hiccups.
        logger.exception("seed_multi_account_defaults failed (non-fatal)")
