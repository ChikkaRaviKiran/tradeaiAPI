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
    text,
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
    instrument = Column(String(20), nullable=False, default="NIFTY", index=True)
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
    instrument = Column(String(20), nullable=False, default="NIFTY", index=True)
    date = Column(String(10), nullable=False, index=True)
    time = Column(String(8), nullable=False)
    nifty_price = Column(Float, nullable=False)
    price = Column(Float, nullable=True)  # Generic price column
    vwap = Column(Float, nullable=True)
    regime = Column(String(20), nullable=False)
    global_bias = Column(String(15), nullable=False)
    # Technical indicators
    ema9 = Column(Float, nullable=True)
    ema20 = Column(Float, nullable=True)
    ema50 = Column(Float, nullable=True)
    ema200 = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    bollinger_upper = Column(Float, nullable=True)
    bollinger_middle = Column(Float, nullable=True)
    bollinger_lower = Column(Float, nullable=True)
    # Options metrics
    pcr = Column(Float, nullable=True)
    max_pain = Column(Float, nullable=True)
    call_oi_cluster = Column(Float, nullable=True)
    put_oi_cluster = Column(Float, nullable=True)
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


class StockRankingRecord(Base):
    """AI-ranked stocks — SRS Module 3.3."""

    __tablename__ = "stock_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    trend_strength_score = Column(Float, default=0.0)
    institutional_score = Column(Float, default=0.0)
    volume_breakout_score = Column(Float, default=0.0)
    earnings_growth_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    composite_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionRecord(Base):
    """ML model prediction history — SRS Module 3.3."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    instrument = Column(String(20), nullable=False, index=True)
    bias = Column(String(10), nullable=False)
    confidence = Column(Float, default=0.0)
    model = Column(String(30), nullable=False)
    actual_outcome = Column(String(10), nullable=True)  # Filled post-market for accuracy tracking
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalRecord(Base):
    """Persisted strategy signals for audit trail — SRS Module 3.5."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    time = Column(String(8), nullable=False)
    instrument = Column(String(20), nullable=False, index=True)
    strategy = Column(String(30), nullable=False)
    option_type = Column(String(2), nullable=False)
    score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    strike_price = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    ai_decision = Column(String(10), nullable=True)  # "accepted" / "rejected"
    ai_confidence = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class StrategyEvalRecord(Base):
    """Strategy evaluation results — ranked recommendations for next-day trading."""

    __tablename__ = "strategy_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_date = Column(String(10), nullable=False, index=True)
    instrument = Column(String(20), nullable=False, index=True)
    strategy = Column(String(30), nullable=False)
    rank = Column(Integer, nullable=False)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    avg_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    composite_score = Column(Float, default=0.0)
    current_regime = Column(String(20), nullable=True)
    signal_frequency = Column(Float, default=0.0)
    eval_days = Column(Integer, default=0)
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
    """Create all tables and run pending column migrations."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add columns that may be missing on existing tables (safe migrations)
        migrations = [
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS ema200 FLOAT",
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS instrument VARCHAR(20) DEFAULT 'NIFTY'",
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS price FLOAT",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS instrument VARCHAR(20) DEFAULT 'NIFTY'",
        ]
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass  # Column may already exist or DB may not support IF NOT EXISTS
