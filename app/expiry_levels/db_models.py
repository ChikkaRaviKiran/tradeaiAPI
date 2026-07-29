"""DB model for weekly/monthly expiry-planning pivot level snapshots."""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db.models import Base, _now_ist


class ExpiryLevelSnapshot(Base):
    """Classic-pivot (3 resistance + 3 support) weekly/monthly S/R snapshot
    for one index, computed on a schedule (weekly: every Wednesday morning -
    NIFTY/SENSEX's Wed->Tue expiry cycle; monthly: first trading days of a
    new month) — see `app/expiry_levels/scheduler.py`.

    INFORMATIONAL ONLY — no strategy or order-placement is attached.
    """

    __tablename__ = "expiry_level_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)          # NIFTY, SENSEX
    timeframe = Column(String(10), nullable=False, index=True)       # "weekly" | "monthly"
    period_start = Column(String(10), nullable=False)                # YYYY-MM-DD
    period_end = Column(String(10), nullable=False)                  # YYYY-MM-DD
    computed_at = Column(DateTime, default=_now_ist)

    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)

    r1 = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    r3 = Column(Float, nullable=True)
    s1 = Column(Float, nullable=True)
    s2 = Column(Float, nullable=True)
    s3 = Column(Float, nullable=True)

    status = Column(String(20), nullable=False, default="ok")        # ok / no_data / error
    notes = Column(Text, nullable=True)
