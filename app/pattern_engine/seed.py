"""Insert / refresh seed pattern definitions in the DB.

Idempotent. Refreshes trigger_json/exit_rule_json on each startup UNLESS the
operator has marked the pattern as '[locked]' in notes (set via UI when
hand-tuning). Status / size_multiplier / notes are always preserved.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from app.pattern_engine.db_models import PEPattern
from app.pattern_engine.patterns import SEED_PATTERNS

logger = logging.getLogger(__name__)


def _is_locked(notes: str | None) -> bool:
    return bool(notes) and "[locked]" in notes


async def upsert_seed_patterns(session) -> int:
    """Insert any seed patterns missing from the DB. Returns count inserted."""
    existing_rows = (await session.execute(select(PEPattern))).scalars().all()
    existing_by_id = {r.pattern_id: r for r in existing_rows}

    inserted = 0
    refreshed = 0
    skipped_locked = 0
    for sp in SEED_PATTERNS:
        existing = existing_by_id.get(sp.pattern_id)
        if existing is not None:
            if _is_locked(existing.notes):
                skipped_locked += 1
                continue
            existing.name = sp.name
            existing.description = sp.description
            existing.direction = sp.direction
            existing.trigger_json = sp.trigger_json
            existing.exit_rule_json = sp.exit_rule_json
            existing.tier = sp.tier
            refreshed += 1
            continue
        session.add(PEPattern(
            pattern_id=sp.pattern_id,
            name=sp.name,
            tier=sp.tier,
            description=sp.description,
            direction=sp.direction,
            trigger_json=sp.trigger_json,
            exit_rule_json=sp.exit_rule_json,
            status="shadow",
            size_multiplier=1.0,
        ))
        inserted += 1
    await session.commit()
    logger.info(
        "pattern_engine: seed upsert ok (inserted=%d refreshed=%d locked=%d)",
        inserted, refreshed, skipped_locked,
    )
    return inserted
