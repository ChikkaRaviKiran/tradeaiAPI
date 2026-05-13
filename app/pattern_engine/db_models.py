"""Pattern Engine DB models — registered with the existing Base."""

from __future__ import annotations

from datetime import datetime

import pytz
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from app.db.models import Base

_IST = pytz.timezone("Asia/Kolkata")


def _now_ist():
    return datetime.now(_IST).replace(tzinfo=None)


class PEMarketSnapshot(Base):
    """Point-in-time market feature snapshot, one row every 5 minutes per symbol.

    Every feature must be computable using ONLY data with timestamp <= ts.
    """

    __tablename__ = "pe_market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True, default="NIFTY")
    spot = Column(Float, nullable=False)

    # Trend / structure
    vwap = Column(Float, nullable=True)
    vwap_dist_pct = Column(Float, nullable=True)        # (spot - vwap) / spot * 100
    ema9 = Column(Float, nullable=True)
    ema20 = Column(Float, nullable=True)
    ema50 = Column(Float, nullable=True)
    ema20_slope_5 = Column(Float, nullable=True)        # slope over last 5 candles

    # Volatility / range
    atr_14 = Column(Float, nullable=True)
    range_pct_15m = Column(Float, nullable=True)        # (h-l)/c over last 15m
    candle_body_pct = Column(Float, nullable=True)      # current candle body / range

    # Levels
    pdh = Column(Float, nullable=True)                  # prior day high
    pdl = Column(Float, nullable=True)
    pdc = Column(Float, nullable=True)
    dist_to_pdh_pct = Column(Float, nullable=True)
    dist_to_pdl_pct = Column(Float, nullable=True)
    day_high = Column(Float, nullable=True)
    day_low = Column(Float, nullable=True)

    # Open / gap
    day_open = Column(Float, nullable=True)
    gap_pct = Column(Float, nullable=True)              # vs prior close
    gap_class = Column(String(10), nullable=True)       # flat/up_small/up_large/down_small/down_large

    # Time / regime
    time_bucket = Column(String(15), nullable=True)     # 0915-1000, 1000-1100, etc.
    minute_of_day = Column(Integer, nullable=True)
    dow = Column(Integer, nullable=True)                # 0=Mon
    dte = Column(Integer, nullable=True)                # days to nearest weekly expiry
    regime = Column(String(20), nullable=True)          # trend_up/trend_down/range/volatile/unknown

    # Options context (nullable — not always available historically)
    pcr = Column(Float, nullable=True)
    pcr_delta_30m = Column(Float, nullable=True)
    atm_ce_oi_change_pct = Column(Float, nullable=True)
    atm_pe_oi_change_pct = Column(Float, nullable=True)
    iv_atm = Column(Float, nullable=True)

    # ORB
    orb_high = Column(Float, nullable=True)             # 09:15-09:30
    orb_low = Column(Float, nullable=True)
    orb_broken = Column(String(10), nullable=True)      # none/up/down

    created_at = Column(DateTime, default=_now_ist)


class PEPattern(Base):
    """Pattern definition (immutable trigger; mutable status & params)."""

    __tablename__ = "pe_patterns"

    pattern_id = Column(String(60), primary_key=True)
    name = Column(String(120), nullable=False)
    tier = Column(Integer, nullable=False, default=1)   # 1=structural, 2=discovered
    description = Column(Text, nullable=True)
    direction = Column(String(4), nullable=False)       # CE / PE
    trigger_json = Column(JSON, nullable=False)         # predicate definition
    exit_rule_json = Column(JSON, nullable=False)       # sl/target/time_stop/etc
    status = Column(String(15), nullable=False, default="shadow")  # research/shadow/live/paused/retired
    size_multiplier = Column(Float, nullable=False, default=1.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now_ist)
    updated_at = Column(DateTime, default=_now_ist, onupdate=_now_ist)


class PEPatternStats(Base):
    """Versioned stats per pattern per rolling-window. Refreshed nightly."""

    __tablename__ = "pe_pattern_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(60), nullable=False, index=True)
    computed_at = Column(DateTime, nullable=False, default=_now_ist, index=True)
    window = Column(String(15), nullable=False, index=True)  # all/180d/90d/30d
    n_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    expectancy_pct = Column(Float, default=0.0)
    avg_win_pct = Column(Float, default=0.0)
    avg_loss_pct = Column(Float, default=0.0)
    avg_hold_min = Column(Float, default=0.0)
    avg_mae_pct = Column(Float, default=0.0)
    avg_mfe_pct = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    monthly_pnl_json = Column(JSON, nullable=True)           # {"2026-04": 12.4, ...}
    months_profitable = Column(Integer, default=0)
    months_total = Column(Integer, default=0)
    suggested_tier = Column(String(10), nullable=True)        # S/A/B/REJECT


class PEPatternOccurrence(Base):
    """Every time a pattern would have triggered, with simulated outcome."""

    __tablename__ = "pe_pattern_occurrences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(60), nullable=False, index=True)
    ts = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="NIFTY")
    direction = Column(String(4), nullable=False)
    spot_at_entry = Column(Float, nullable=True)
    strike = Column(Float, nullable=True)
    entry_premium = Column(Float, nullable=True)
    exit_premium = Column(Float, nullable=True)
    outcome_pnl_pct = Column(Float, nullable=True)           # premium % move
    outcome_spot_pts = Column(Float, nullable=True)
    hold_minutes = Column(Integer, nullable=True)
    exit_reason = Column(String(30), nullable=True)
    mae_pct = Column(Float, nullable=True)
    mfe_pct = Column(Float, nullable=True)
    regime_at_entry = Column(String(20), nullable=True)
    source = Column(String(15), nullable=False, default="backfill")  # backfill/shadow/live
    features_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_now_ist)


class PELiveProbe(Base):
    """Every live match decision (taken or skipped) — feeds learning."""

    __tablename__ = "pe_live_probes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, nullable=False, index=True)
    pattern_id = Column(String(60), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="NIFTY")
    edge_score = Column(Float, nullable=True)
    minisim_n = Column(Integer, nullable=True)
    minisim_wr = Column(Float, nullable=True)
    minisim_pf = Column(Float, nullable=True)
    decision = Column(String(20), nullable=False)            # taken/skipped_threshold/skipped_minisim/skipped_regime/skipped_other
    skip_reason = Column(Text, nullable=True)
    occurrence_id = Column(Integer, nullable=True)           # link to outcome if taken
    created_at = Column(DateTime, default=_now_ist)


__all__ = [
    "PEMarketSnapshot",
    "PEPattern",
    "PEPatternStats",
    "PEPatternOccurrence",
    "PELiveProbe",
]
