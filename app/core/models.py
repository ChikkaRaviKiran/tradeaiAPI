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


class StrategyName(str, enum.Enum):
    ORB = "ORB"
    VWAP_RECLAIM = "VWAP_RECLAIM"
    TREND_PULLBACK = "TREND_PULLBACK"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    RANGE_BREAKOUT = "RANGE_BREAKOUT"


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
    option_type: OptionType
    score: float = 0.0
    entry_price: float = 0.0
    strike_price: float = 0.0
    stoploss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    details: dict = Field(default_factory=dict)


class SignalScore(BaseModel):
    strategy_trigger: float = 0.0
    volume_confirmation: float = 0.0
    vwap_alignment: float = 0.0
    options_oi_signal: float = 0.0
    global_bias_score: float = 0.0
    historical_pattern: float = 0.0
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
    date: str = ""
    time: str = ""
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


# ── Market Snapshot ───────────────────────────────────────────────────────────

class MarketSnapshot(BaseModel):
    nifty_price: float = 0.0
    vwap: Optional[float] = None
    regime: MarketRegime = MarketRegime.INSUFFICIENT_DATA
    global_bias: GlobalBias = GlobalBias.UNAVAILABLE
    indicators: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    options_metrics: OptionsMetrics = Field(default_factory=OptionsMetrics)
    timestamp: datetime = Field(default_factory=datetime.now)


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
