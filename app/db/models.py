"""Database models and session management."""

from __future__ import annotations

from datetime import datetime

import pytz

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

_IST = pytz.timezone("Asia/Kolkata")


def _now_ist():
    return datetime.now(_IST).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    """Persistent trade log in PostgreSQL."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(20), unique=True, nullable=False, index=True)
    instrument = Column(String(20), nullable=False, default="NIFTY", index=True)
    engine = Column(String(4), nullable=False, default="v1", index=True)  # "v1" or "v2"
    date = Column(String(10), nullable=False, index=True)
    time = Column(String(8), nullable=False)
    exit_time = Column(String(8), nullable=True)
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
    exit_type = Column(String(20), nullable=True)  # V2: stoploss/target/time_exit/thesis_break/trailing/eod
    day_type = Column(String(10), nullable=True)  # V2: trend/range/volatile/unclear
    created_at = Column(DateTime, default=_now_ist)
    updated_at = Column(DateTime, default=_now_ist, onupdate=_now_ist)


class DailyReport(Base):
    """Daily summary record."""

    __tablename__ = "daily_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    engine = Column(String(4), nullable=False, default="v1", index=True)  # "v1" or "v2"
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    day_type = Column(String(10), nullable=True)  # V2: day classification
    created_at = Column(DateTime, default=_now_ist)


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
    htf_trend = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=_now_ist)


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
    engine = Column(String(4), nullable=True, default="v1")  # v1 or v2
    created_at = Column(DateTime, default=_now_ist)


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
    created_at = Column(DateTime, default=_now_ist)


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
    created_at = Column(DateTime, default=_now_ist)


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
    created_at = Column(DateTime, default=_now_ist)


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
    created_at = Column(DateTime, default=_now_ist)


class TelegramNewsRecord(Base):
    """Scraped news from Telegram channel with GPT Vision extraction."""

    __tablename__ = "telegram_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    message_id = Column(Integer, nullable=True)
    image_url = Column(Text, nullable=True)
    extracted_text = Column(Text, nullable=True)
    symbols = Column(Text, nullable=True)  # comma-separated stock symbols found
    sentiment = Column(String(20), nullable=True)  # bullish/bearish/neutral
    sentiment_score = Column(Float, default=0.0)  # -1.0 to 1.0
    source = Column(String(50), default="daytradertelugu")
    created_at = Column(DateTime, default=_now_ist)


class DailyAIInsight(Base):
    """AI-generated daily market insight from pre-market analysis."""

    __tablename__ = "daily_ai_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    insight_type = Column(String(30), nullable=False)  # pre_market / mid_day / post_market
    market_bias = Column(String(20), nullable=True)  # bullish/bearish/neutral
    confidence = Column(Float, default=0.0)
    fii_dii_signal = Column(String(20), nullable=True)
    fii_net = Column(Float, nullable=True)
    dii_net = Column(Float, nullable=True)
    breadth_signal = Column(String(20), nullable=True)
    advance_decline_ratio = Column(Float, nullable=True)
    news_sentiment = Column(Float, default=0.0)  # avg of news sentiment scores
    strong_sectors = Column(Text, nullable=True)  # comma-separated
    weak_sectors = Column(Text, nullable=True)  # comma-separated
    key_levels = Column(Text, nullable=True)  # JSON string with support/resistance
    ai_summary = Column(Text, nullable=True)  # GPT-generated summary
    trading_plan = Column(Text, nullable=True)  # GPT-generated plan
    raw_data = Column(Text, nullable=True)  # JSON dump of all input data
    created_at = Column(DateTime, default=_now_ist)


class OptionCandle(Base):
    """1-minute OHLCV candle for NIFTY/BANKNIFTY options — cached for backtesting."""

    __tablename__ = "option_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(20), nullable=False, index=True)   # NIFTY, BANKNIFTY
    expiry = Column(String(10), nullable=False, index=True)       # DDMMMYY e.g. 10APR26
    strike = Column(Float, nullable=False)
    option_type = Column(String(2), nullable=False)               # CE or PE
    trading_symbol = Column(String(60), nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)         # YYYY-MM-DD
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=_now_ist)


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
    # Use a fresh engine bound to the current event loop to avoid
    # 'attached to a different loop' errors on container startup.
    global async_engine, AsyncSessionLocal
    async_engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add columns that may be missing on existing tables (safe migrations)
        migrations = [
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS ema200 FLOAT",
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS instrument VARCHAR(20) DEFAULT 'NIFTY'",
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS price FLOAT",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS instrument VARCHAR(20) DEFAULT 'NIFTY'",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_time VARCHAR(8)",
            "ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS htf_trend VARCHAR(10)",
            "CREATE INDEX IF NOT EXISTS ix_telegram_news_date ON telegram_news (date)",
            "CREATE INDEX IF NOT EXISTS ix_daily_ai_insights_date ON daily_ai_insights (date)",
            # V2 dual-engine columns
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS engine VARCHAR(4) DEFAULT 'v1'",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_type VARCHAR(20)",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS day_type VARCHAR(10)",
            "CREATE INDEX IF NOT EXISTS ix_trades_engine ON trades (engine)",
            "ALTER TABLE daily_reports ADD COLUMN IF NOT EXISTS engine VARCHAR(4) DEFAULT 'v1'",
            "ALTER TABLE daily_reports ADD COLUMN IF NOT EXISTS day_type VARCHAR(10)",
            "CREATE INDEX IF NOT EXISTS ix_daily_reports_engine ON daily_reports (engine)",
            # V2 alert engine column
            "ALTER TABLE alert_records ADD COLUMN IF NOT EXISTS engine VARCHAR(4) DEFAULT 'v1'",
            # Option candle cache indexes
            "CREATE INDEX IF NOT EXISTS ix_option_candles_lookup ON option_candles (instrument, date, strike, option_type)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_option_candles_unique ON option_candles (trading_symbol, timestamp)",
        ]
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass  # Column may already exist or DB may not support IF NOT EXISTS
