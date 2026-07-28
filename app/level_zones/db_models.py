"""DB model for the Level Zones breakout PAPER-TRADE alert tracker.

PAPER TRADING ONLY. Nothing in this module ever places a real order — it
just records a hypothetical option-buying trade whenever the alert engine
detects a confirmed breakout of a weekly/monthly confluence zone, and
tracks it through to target / stop-loss / EOD square-off so a Telegram
alert can be sent at entry and at exit.
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db.models import Base, _now_ist


class LevelZonePaperTrade(Base):
    __tablename__ = "level_zone_paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)          # YYYY-MM-DD
    symbol = Column(String(20), nullable=False, index=True)        # NIFTY, SENSEX
    direction = Column(String(2), nullable=False)                  # CE or PE

    zone_price = Column(Float, nullable=False)                     # the S/R zone that broke
    zone_confidence = Column(Integer, nullable=False)
    zone_sources = Column(Text, nullable=True)                     # JSON list of contributing pivot/swing sources

    strike = Column(Float, nullable=False)
    expiry = Column(String(10), nullable=False)                    # DDMMMYY

    entry_price = Column(Float, nullable=False)                    # option premium at signal (best-ask if available)
    sl_price = Column(Float, nullable=False)                       # option premium stop-loss
    target_price = Column(Float, nullable=False)                   # option premium target
    entry_time = Column(DateTime, nullable=False)
    spot_at_entry = Column(Float, nullable=True)                   # underlying spot when the trade was opened

    status = Column(String(20), nullable=False, default="open")    # open / target_hit / sl_hit / eod_close
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    spot_at_exit = Column(Float, nullable=True)                    # underlying spot when the trade was closed
    pnl_points = Column(Float, nullable=True)                      # exit_price - entry_price (per unit, x lot_size = rupees)

    created_at = Column(DateTime, default=_now_ist)
