"""Multi-instance registry for ATM Straddle / OTM Strangle scanners.

Owns the lifecycle of N ``ATLStraddleScanner`` instances, one per
active ``StrategyInstance`` row in the DB.  Each scanner has:

- Its own DB-backed settings loader (bound to a specific instance_id).
- Its own on-disk state file (namespaced ``atl_straddle_state_<id>.json``).
- Its own broker client, resolved from the ``BrokerAccount`` bound to
  that instance (falls back to the global broker when the account has
  no broker override).

The registry is used by the orchestrator to:

1. Build the initial fleet at startup.
2. Reconcile every N cycles: spawn scanners for newly-created rows,
   drop scanners for rows that were deleted or de-activated.
3. Dispatch ``run_cycle`` to every scanner whose configured index
   matches the current instrument tick.

Backwards compatibility: when the DB has zero active rows, the registry
returns an empty fleet and the orchestrator's legacy singleton
``self.atl_straddle_scanner`` continues to serve the JSON-file
strategy path.  When the DB has ≥1 rows, ``self.atl_straddle_scanner``
is aliased to the first registry scanner so all existing runtime API
endpoints (``/api/atm/runtime``, ``/api/atm/force-close`` …) keep
working against the primary instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from app.engine.atl_settings import make_instance_settings_loader
from app.engine.atl_straddle_scanner import ATLStraddleScanner
from app.execution.broker_factory import build_broker_from_account_id

logger = logging.getLogger(__name__)


class StrategyInstanceRegistry:
    def __init__(
        self,
        *,
        client,
        alert_manager,
        broker_factory: Callable[[Optional[str]], Any],
        expiry_provider: Optional[Callable[[str], tuple[str, Optional[Any]]]] = None,
    ):
        self._client = client
        self._alert_manager = alert_manager
        # ``broker_factory(broker_name)`` is the legacy env-based path used
        # as a fallback when a StrategyInstance has no account_id bound.
        self._broker_factory = broker_factory
        self._expiry_provider = expiry_provider
        # instance_id -> scanner
        self._scanners: dict[int, ATLStraddleScanner] = {}
        # instance_id -> (account_id, broker_name) — cached to detect changes.
        self._binding: dict[int, tuple[Optional[int], str]] = {}

    @property
    def scanners(self) -> dict[int, ATLStraddleScanner]:
        return self._scanners

    def get_primary(self) -> Optional[ATLStraddleScanner]:
        """First scanner (deterministic by instance_id) or None."""
        if not self._scanners:
            return None
        first_id = min(self._scanners.keys())
        return self._scanners[first_id]

    async def sync_from_db(self) -> tuple[int, int, int]:
        """Reconcile the in-memory scanner fleet against the DB.

        Returns ``(created, updated, dropped)`` counts for diagnostics.
        Best-effort — logs and continues on any per-row failure.
        """
        rows = await _fetch_active_instances()
        wanted_ids = {r["id"] for r in rows}

        created = 0
        updated = 0
        dropped = 0

        # 1. Drop scanners for rows that no longer exist / are inactive.
        stale_ids = set(self._scanners.keys()) - wanted_ids
        for sid in stale_ids:
            try:
                self._drop(sid)
                dropped += 1
            except Exception:
                logger.exception("[Registry] Failed dropping scanner %s", sid)

        # 2. Create / re-bind scanners for wanted rows.
        for r in rows:
            iid = r["id"]
            broker_name = r["broker_name"]
            account_id = r["account_id"]
            new_binding = (account_id, broker_name or "")

            existing = self._scanners.get(iid)
            if existing is None:
                try:
                    self._spawn(iid, account_id, broker_name)
                    self._binding[iid] = new_binding
                    created += 1
                except Exception:
                    logger.exception("[Registry] Failed spawning scanner for instance %s", iid)
                continue

            # Rebind broker if the row's account was swapped OR the
            # account's credentials/broker changed.
            if self._binding.get(iid) != new_binding:
                try:
                    existing.broker = self._resolve_broker(account_id, broker_name)
                    self._binding[iid] = new_binding
                    updated += 1
                    logger.info(
                        "[Registry] Rebound scanner %s to account_id=%s broker=%s",
                        iid, account_id, broker_name,
                    )
                except Exception:
                    logger.exception("[Registry] Failed rebinding scanner %s", iid)

        if created or updated or dropped:
            logger.info(
                "[Registry] sync_from_db: created=%d updated=%d dropped=%d "
                "(active scanners: %d)",
                created, updated, dropped, len(self._scanners),
            )
        return created, updated, dropped

    def _spawn(self, instance_id: int, account_id: Optional[int], broker_name: Optional[str]) -> None:
        broker = self._resolve_broker(account_id, broker_name)
        loader = make_instance_settings_loader(instance_id)
        scanner = ATLStraddleScanner(
            client=self._client,
            alert_manager=self._alert_manager,
            broker=broker,
            expiry_provider=self._expiry_provider,
            instance_id=instance_id,
            settings_loader=loader,
        )
        self._scanners[instance_id] = scanner
        logger.info(
            "[Registry] Spawned scanner instance_id=%s account_id=%s broker=%s",
            instance_id, account_id,
            getattr(broker, "name", broker.__class__.__name__ if broker else "none"),
        )

    def _resolve_broker(self, account_id: Optional[int], broker_name: Optional[str]) -> Any:
        """Return the broker for an instance.

        When ``account_id`` is bound: MUST use the per-account broker built
        from that row's own credentials + proxy. If the factory can't build
        one (missing access_token, expired creds, unknown broker), we
        return ``None`` — the scanner then records an ``order_error`` and
        skips placement instead of silently routing through the shared env
        credentials, which was the root cause of "orders never placed"
        when a Dhan account was created without an access_token.

        When ``account_id`` is unbound: fall back to the legacy env-based
        factory (single-account deployments).
        """
        if account_id:
            broker = build_broker_from_account_id(account_id)
            if broker is not None:
                return broker
            logger.error(
                "[Registry] Cannot build account-scoped broker for "
                "account_id=%s broker=%s — check that access_token / "
                "credentials are set on the account row. Refusing to fall "
                "back to env-based broker (would use wrong creds and skip "
                "the account's proxy).",
                account_id, broker_name,
            )
            return None
        return self._broker_factory(broker_name)

    def _drop(self, instance_id: int) -> None:
        scanner = self._scanners.pop(instance_id, None)
        self._binding.pop(instance_id, None)
        if scanner is None:
            return
        # Best-effort cleanup: mark the state file as retired by removing
        # it. If the user re-adds an instance with the same id later the
        # scanner will start fresh, which is safer than resurrecting stale
        # legs from disk.
        try:
            path = _state_path_for_instance(instance_id)
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.debug("[Registry] Could not remove stale state file for %s", instance_id, exc_info=True)
        logger.info("[Registry] Dropped scanner instance_id=%s", instance_id)


def _state_path_for_instance(instance_id: int) -> str:
    from app.engine.atl_straddle_scanner import _atl_state_path
    return _atl_state_path(instance_id)


async def _fetch_active_instances() -> list[dict]:
    """Return active ATM_STRADDLE / OTM_STRANGLE instances with their
    bound broker name (if any).  Never raises — returns [] on any error.
    """
    try:
        from sqlalchemy import select
        from app.db.account_models import BrokerAccount, StrategyInstance
        from app.db.models import AsyncSessionLocal

        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(StrategyInstance)
                    .where(StrategyInstance.is_active.is_(True))
                    .where(StrategyInstance.strategy_type.in_(
                        ("ATM_STRADDLE", "OTM_STRANGLE")
                    ))
                    .order_by(StrategyInstance.id.asc())
                )
            ).scalars().all()
            if not rows:
                return []

            acct_ids = [r.account_id for r in rows if r.account_id is not None]
            broker_by_id: dict[int, str] = {}
            if acct_ids:
                accts = (
                    await s.execute(
                        select(BrokerAccount).where(BrokerAccount.id.in_(acct_ids))
                    )
                ).scalars().all()
                broker_by_id = {a.id: a.broker for a in accts}

            return [
                {
                    "id": r.id,
                    "account_id": r.account_id,
                    "broker_name": broker_by_id.get(r.account_id) if r.account_id is not None else None,
                }
                for r in rows
            ]
    except Exception:
        logger.debug("_fetch_active_instances failed", exc_info=True)
        return []
