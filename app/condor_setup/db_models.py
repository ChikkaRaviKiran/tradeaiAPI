"""DB model for daily pre-market condor setups — registered with the shared Base."""
from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.db.models import Base, _now_ist


class CondorDailySetup(Base):
    """Pre-market computed confluence-based iron condor setup for one index/date.

    INFORMATIONAL ONLY — the operator manually places the trade using these
    levels via the Strategy page / their own broker terminal. Nothing in this
    table is ever read by the live order-execution path.
    """

    __tablename__ = "condor_daily_setups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)          # YYYY-MM-DD
    index = Column(String(20), nullable=False, index=True)         # NIFTY, SENSEX
    computed_at = Column(DateTime, default=_now_ist)

    # Whether this index is "today's scheduled pick" per the validated
    # Mon/Tue/Fri=NIFTY, Wed/Thu=SENSEX weekday rule.
    is_recommended_today = Column(Boolean, nullable=False, default=False)

    spot = Column(Float, nullable=True)
    atm_strike = Column(Float, nullable=True)
    strike_interval = Column(Float, nullable=False)
    lot_size = Column(Integer, nullable=False)
    expiry_weekday = Column(Integer, nullable=False)

    resistance_price = Column(Float, nullable=True)
    resistance_source = Column(String(120), nullable=True)
    resistance_confidence = Column(Integer, nullable=True)

    support_price = Column(Float, nullable=True)
    support_source = Column(String(120), nullable=True)
    support_confidence = Column(Integer, nullable=True)

    short_ce_strike = Column(Float, nullable=True)
    short_pe_strike = Column(Float, nullable=True)
    long_ce_strike = Column(Float, nullable=True)
    long_pe_strike = Column(Float, nullable=True)
    wing_width_points = Column(Integer, nullable=False, default=250)

    status = Column(String(20), nullable=False, default="ok")      # ok / no_data / error
    notes = Column(Text, nullable=True)
    levels_json = Column(Text, nullable=True)                      # all confluence candidates, for transparency
