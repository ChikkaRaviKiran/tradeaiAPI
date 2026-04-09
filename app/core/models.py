"""Data models used across the system."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class OptionType(str, enum.Enum):
    CALL = "CE"
    PUT = "PE"


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"


class MarketRegime(str, enum.Enum):
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    INSUFFICIENT_DATA = "insufficient_data"


class GlobalBias(str, enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNAVAILABLE = "unavailable"


class DayType(str, enum.Enum):
    """Day classification for V2 engine."""
    TREND = "trend"
    RANGE = "range"
    VOLATILE = "volatile"
    UNCLEAR = "unclear"
    PENDING = "pending"  # Not yet classified (before 10:00)


class StrategyName(str, enum.Enum):
    ORB = "ORB"
    VWAP_RECLAIM = "VWAP_RECLAIM"
    TREND_PULLBACK = "TREND_PULLBACK"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    RANGE_BREAKOUT = "RANGE_BREAKOUT"
    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"
    EMA_BREAKOUT = "EMA_BREAKOUT"
    BREAKOUT_20D = "BREAKOUT_20D"          # SRS Strategy 2: 20-day high breakout
    OPTIONS_INCOME = "OPTIONS_INCOME"      # SRS Strategy 3: Iron Condor / range
    # V2 strategies
    VWAP_PULLBACK = "VWAP_PULLBACK"        # V2: VWAP pullback on trend days
    GEX_BOUNCE = "GEX_BOUNCE"              # V2: GEX level bounce on range days
    RSI_EXTREME = "RSI_EXTREME"            # V2: RSI extreme reversal on volatile days
    # Adaptive market structure strategy
    ADAPTIVE = "ADAPTIVE"                  # Context-aware adaptive entry


# ── Market Data Models ─────────────────────────────────────────────────────────

class Candle(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class OptionsChainRow(BaseModel):
    strike_price: float
    call_ltp: Optional[float] = None
    put_ltp: Optional[float] = None
    call_oi: int = 0
    put_oi: int = 0
    call_volume: int = 0
    put_volume: int = 0
    change_in_oi: int = 0
    implied_volatility: Optional[float] = None


class GlobalIndex(BaseModel):
    symbol: str
    change_pct: float
    last_price: float


# ── Technical Indicators ───────────────────────────────────────────────────────

class TechnicalIndicators(BaseModel):
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    vwap: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    atr: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    adx: Optional[float] = None
    trend_strength: Optional[int] = None  # 0-3: EMA alignment score (3=strong uptrend)
    vwap_is_volume_weighted: bool = False  # True only when real volume data was used


class OptionsMetrics(BaseModel):
    pcr: Optional[float] = None  # None = not fetched yet, 1.0+ = real data
    max_pain: Optional[float] = None  # None = no chain data, 0.0 impossible
    call_oi_cluster: Optional[float] = None
    put_oi_cluster: Optional[float] = None
    oi_change: int = 0
    total_call_volume: int = 0  # Summed option volume from chain
    total_put_volume: int = 0
    atm_option_volume: int = 0  # ATM strikes volume (participation proxy)


# ── Strategy Signal ───────────────────────────────────────────────────────────

class StrategySignal(BaseModel):
    strategy: StrategyName
    instrument: str = "NIFTY"  # Instrument symbol this signal is for
    option_type: OptionType
    score: float = 0.0
    confidence: float = 0.0  # Strategy-level confidence (0-100)
    entry_price: float = 0.0
    strike_price: float = 0.0
    stoploss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    details: dict = Field(default_factory=dict)


class SignalScore(BaseModel):
    """LOCKED v1.0: 5-factor scoring model (max 100)."""
    strategy_strength: float = 0.0      # 30 pts max
    market_alignment: float = 0.0       # 25 pts max
    volume_confirmation: float = 0.0    # 20 pts max
    options_oi_signal: float = 0.0      # 15 pts max
    volatility_context: float = 0.0     # 10 pts max
    total: float = 0.0


# ── AI Decision ───────────────────────────────────────────────────────────────

class AIDecision(BaseModel):
    trade_decision: bool = False
    confidence_score: float = 0.0
    entry_price: float = 0.0
    stoploss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    reason: str = ""


# ── Trade ─────────────────────────────────────────────────────────────────────

class Trade(BaseModel):
    trade_id: Optional[str] = None
    instrument: str = "NIFTY"  # Underlying instrument
    engine: str = "v1"  # "v1" (current) or "v2" (new system)
    date: str = ""
    time: str = ""
    exit_time: Optional[str] = None
    symbol: str = ""
    strike: float = 0.0
    option_type: OptionType = OptionType.CALL
    strategy: StrategyName = StrategyName.ORB
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    stoploss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    confidence: float = 0.0
    pnl: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    lot_size: int = 50
    reason: str = ""
    breakout_level: Optional[float] = None  # Spot level that triggered breakout entry
    # V2 exit tracking
    entry_datetime: Optional[datetime] = None  # For time-based exits
    max_hold_minutes: int = 0  # 0 = no time limit (V1), >0 = V2 time exit
    exit_type: Optional[str] = None  # stoploss/target/time_exit/thesis_break/trailing/eod
    day_type: Optional[str] = None  # Day classification when trade was taken


# ── Market Snapshot ───────────────────────────────────────────────────────────

class MarketSnapshot(BaseModel):
    instrument: str = "NIFTY"  # Which instrument this snapshot is for
    price: float = 0.0         # Current price (generic, replaces nifty_price)
    nifty_price: float = 0.0   # Backward compat — alias for price when instrument is NIFTY
    vwap: Optional[float] = None
    regime: MarketRegime = MarketRegime.INSUFFICIENT_DATA
    global_bias: GlobalBias = GlobalBias.UNAVAILABLE
    indicators: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    options_metrics: OptionsMetrics = Field(default_factory=OptionsMetrics)
    timestamp: datetime = Field(default_factory=datetime.now)
    prev_day_high: Optional[float] = None   # Previous day high — key resistance
    prev_day_low: Optional[float] = None    # Previous day low — key support
    prev_day_close: Optional[float] = None  # Previous day close — reference level
    day_open: Optional[float] = None          # Today's opening price — for gap analysis
    is_expiry_day: bool = False             # True if today is this instrument's options expiry
    htf_trend: Optional[str] = None         # Higher-timeframe (5-min) trend: "bullish", "bearish", "neutral"


# ── Alert Item ────────────────────────────────────────────────────────────────

class AlertItem(BaseModel):
    id: str = ""
    alert_type: str = "signal"  # signal | exit | info | report
    title: str = ""
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    trade_id: Optional[str] = None
    strategy: Optional[str] = None
    pnl: Optional[float] = None
    engine: str = "v1"  # "v1" or "v2"


# ── Sentiment & Stock Ranking (SRS Modules 3.3 / 3.4) ────────────────────────

class SentimentScore(BaseModel):
    """News / social sentiment for a symbol."""
    symbol: str = ""
    score: float = 0.0        # -1.0 (bearish) to +1.0 (bullish)
    source: str = ""          # e.g. "news", "twitter", "earnings"
    headline: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class StockRanking(BaseModel):
    """AI-ranked stock with factor scores (SRS 3.3)."""
    symbol: str = ""
    rank: int = 0
    trend_strength_score: float = 0.0    # 25%
    institutional_score: float = 0.0     # 20%
    volume_breakout_score: float = 0.0   # 20%
    earnings_growth_score: float = 0.0   # 20%
    sentiment_score: float = 0.0         # 15%
    composite_score: float = 0.0         # Weighted total
    timestamp: datetime = Field(default_factory=datetime.now)


class MarketPrediction(BaseModel):
    """AI prediction output (SRS 3.3)."""
    instrument: str = ""
    bias: GlobalBias = GlobalBias.NEUTRAL  # Bullish / Bearish / Neutral
    confidence: float = 0.0               # 0-100%
    model: str = ""                        # Which model produced this
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Performance Metrics ───────────────────────────────────────────────────────

class PerformanceMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_pnl_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
