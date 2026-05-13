"""Insert / refresh seed pattern definitions in the DB.

Idempotent — safe to run multiple times. Does not overwrite status,
size_multiplier, or notes (those are user-managed).
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from app.pattern_engine.db_models import PEPattern
from app.pattern_engine.patterns import SEED_PATTERNS

logger = logging.getLogger(__name__)


async def upsert_seed_patterns(session) -> int:
    """Insert any seed patterns missing from the DB. Returns count inserted."""
    existing = (await session.execute(select(PEPattern.pattern_id))).scalars().all()
    existing_set = set(existing)

    inserted = 0
    for sp in SEED_PATTERNS:
        if sp.pattern_id in existing_set:
            # Refresh trigger/exit JSON + name/description but preserve user-managed fields
            stmt = select(PEPattern).where(PEPattern.pattern_id == sp.pattern_id)
            row = (await session.execute(stmt)).scalar_one()
            row.name = sp.name
            row.description = sp.description
            row.direction = sp.direction
            row.trigger_json = sp.trigger_json
            row.exit_rule_json = sp.exit_rule_json
            row.tier = sp.tier
            continue
        row = PEPattern(
            pattern_id=sp.pattern_id,
            name=sp.name,
            tier=sp.tier,
            description=sp.description,
            direction=sp.direction,
            trigger_json=sp.trigger_json,
            exit_rule_json=sp.exit_rule_json,
            status="shadow",
            size_multiplier=1.0,
        )
        session.add(row)
        inserted += 1
    await session.commit()
    logger.info("pattern_engine: upserted %d seed patterns (existing=%d)", inserted, len(existing_set))
    return inserted
