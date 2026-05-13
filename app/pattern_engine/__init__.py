"""Pattern Engine — Continuous Live Pattern Re-Evaluation Trading System.

Self-contained subsystem. Does NOT modify or depend on any existing
strategy / scanner / sniper / orchestrator logic. Safe to enable / disable
independently via the PATTERN_ENGINE_ENABLED env flag.

Components:
- db_models      : new SQLAlchemy tables (registered to the existing Base)
- features       : point-in-time feature snapshot computation
- patterns       : seed Tier-1 pattern definitions
- stats          : stats refresh + auto-tiering rules
- backfill       : one-time historical backfill CLI
- matcher        : live matcher + mini-sim gate
- routes         : FastAPI endpoints for the UI
- seed           : inserts seed pattern definitions into DB

Storage tables (created automatically by init_db()):
- pe_market_snapshots
- pe_patterns
- pe_pattern_stats
- pe_pattern_occurrences
- pe_live_probes
"""

from app.pattern_engine import db_models  # noqa: F401  (registers tables with Base)
