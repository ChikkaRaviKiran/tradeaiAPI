"""Database models and session management."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    """Persistent trade log in PostgreSQL."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(20), unique=True, nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)
    time = Column(String(8), nullable=False)
    symbol = Column(String(60), nullable=False)
    strike = Column(Float, nullable=False)
    option_type = Column(String(2), nullable=False)
    strategy = Column(String(30), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stoploss = Column(Float, nullable=False)
    target1 = Column(Float, nullable=True)
    target2 = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String(10), nullable=False, default="open")
    lot_size = Column(Integer, default=50)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyReport(Base):
    """Daily summary record."""

    __tablename__ = "daily_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), unique=True, nullable=False, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketSnapshotRecord(Base):
    """Stores each analysis cycle's market snapshot."""

    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    time = Column(String(8), nullable=False)
    nifty_price = Column(Float, nullable=False)
    vwap = Column(Float, default=0.0)
    regime = Column(String(20), nullable=False)
    global_bias = Column(String(10), nullable=False)
    # Technical indicators
    ema9 = Column(Float, default=0.0)
    ema20 = Column(Float, default=0.0)
    ema50 = Column(Float, default=0.0)
    rsi = Column(Float, default=50.0)
    macd = Column(Float, default=0.0)
    macd_signal = Column(Float, default=0.0)
    macd_hist = Column(Float, default=0.0)
    atr = Column(Float, default=0.0)
    adx = Column(Float, default=0.0)
    bollinger_upper = Column(Float, default=0.0)
    bollinger_middle = Column(Float, default=0.0)
    bollinger_lower = Column(Float, default=0.0)
    # Options metrics
    pcr = Column(Float, default=1.0)
    max_pain = Column(Float, default=0.0)
    call_oi_cluster = Column(Float, default=0.0)
    put_oi_cluster = Column(Float, default=0.0)
    oi_change = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AlertRecord(Base):
    """Persisted alerts for historical review."""

    __tablename__ = "alert_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    alert_type = Column(String(10), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=True)
    trade_id = Column(String(20), nullable=True)
    strategy = Column(String(30), nullable=True)
    pnl = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Async engine
async_engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def create_new_async_session_factory():
    """Create a NEW async engine + session factory — use from separate threads."""
    engine = create_async_engine(settings.database_url, echo=False)
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False), engine


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
