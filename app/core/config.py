"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # AngelOne SmartAPI
    angelone_api_key: str = Field(default="", alias="ANGELONE_API_KEY")
    angelone_client_id: str = Field(default="", alias="ANGELONE_CLIENT_ID")
    angelone_password: str = Field(default="", alias="ANGELONE_PASSWORD")
    angelone_mpin: str = Field(default="", alias="ANGELONE_MPIN")
    angelone_totp_secret: str = Field(default="", alias="ANGELONE_TOTP_SECRET")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/tradeai",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")

    # DhanHQ (expired options data only)
    dhan_access_token: str = Field(default="", alias="DHAN_ACCESS_TOKEN")

    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    telegram_news_channel: str = Field(default="", alias="TELEGRAM_NEWS_CHANNEL")

    # Email
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    alert_email_to: str = Field(default="", alias="ALERT_EMAIL_TO")

    # Trading
    initial_capital: float = Field(default=100000, alias="INITIAL_CAPITAL")
    max_trades_per_day: int = Field(default=3, alias="MAX_TRADES_PER_DAY")  # LOCKED v1.0
    max_daily_loss_pct: float = Field(default=3.0, alias="MAX_DAILY_LOSS_PCT")
    risk_per_trade_pct: float = Field(default=1.0, alias="RISK_PER_TRADE_PCT")
    max_concurrent_positions: int = Field(default=2, alias="MAX_CONCURRENT_POSITIONS")  # Total across all instruments
    max_concurrent_per_instrument: int = Field(default=1, alias="MAX_CONCURRENT_PER_INSTRUMENT")  # Per-index limit
    consecutive_loss_limit: int = Field(default=3, alias="CONSECUTIVE_LOSS_LIMIT")
    nifty_lot_size: int = Field(default=65, alias="NIFTY_LOT_SIZE")

    # Instruments
    # If auto_select_instruments is True, the system evaluates ALL registered
    # instruments and picks the best ones automatically each day.
    # Set to False + ACTIVE_INSTRUMENTS to manually override.
    auto_select_instruments: bool = Field(default=True, alias="AUTO_SELECT_INSTRUMENTS")
    max_active_instruments: int = Field(default=3, alias="MAX_ACTIVE_INSTRUMENTS")  # NIFTY, BANKNIFTY, FINNIFTY
    min_composite_score: float = Field(default=35.0, alias="MIN_COMPOSITE_SCORE")
    active_instruments: str = Field(default="", alias="ACTIVE_INSTRUMENTS")

    # Data sources
    yahoo_finance_enabled: bool = Field(default=True, alias="YAHOO_FINANCE_ENABLED")
    fii_dii_enabled: bool = Field(default=True, alias="FII_DII_ENABLED")
    news_sentiment_enabled: bool = Field(default=False, alias="NEWS_SENTIMENT_ENABLED")

    # System
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    min_margin_required: float = Field(default=5000, alias="MIN_MARGIN_REQUIRED")

    # Dual-Engine Control
    v1_enabled: bool = Field(default=True, alias="V1_ENABLED")
    v2_enabled: bool = Field(default=False, alias="V2_ENABLED")
    v2_live_orders: bool = Field(default=False, alias="V2_LIVE_ORDERS")  # V2 always paper until proven

    # V2 Trading Parameters
    v2_max_trades_per_day: int = Field(default=1, alias="V2_MAX_TRADES_PER_DAY")
    v2_max_concurrent_positions: int = Field(default=1, alias="V2_MAX_CONCURRENT_POSITIONS")
    v2_risk_per_trade_pct: float = Field(default=1.5, alias="V2_RISK_PER_TRADE_PCT")
    v2_consecutive_loss_limit: int = Field(default=2, alias="V2_CONSECUTIVE_LOSS_LIMIT")
    v2_max_hold_minutes: int = Field(default=120, alias="V2_MAX_HOLD_MINUTES")  # LOCKED v1.0
    v2_quick_target_pct: float = Field(default=25.0, alias="V2_QUICK_TARGET_PCT")
    v2_stoploss_pct: float = Field(default=15.0, alias="V2_STOPLOSS_PCT")  # Tightened from 20% to 15%
    v2_breakeven_trigger_pct: float = Field(default=8.0, alias="V2_BREAKEVEN_TRIGGER_PCT")
    v2_skip_unclear_days: bool = Field(default=True, alias="V2_SKIP_UNCLEAR_DAYS")
    v2_openai_model: str = Field(default="gpt-4o-mini", alias="V2_OPENAI_MODEL")

    # MOB (Momentum Option Buying) Backtest Parameters
    mob_max_trades_per_day: int = Field(default=2, alias="MOB_MAX_TRADES_PER_DAY")
    mob_consecutive_loss_stop: int = Field(default=2, alias="MOB_CONSECUTIVE_LOSS_STOP")
    mob_slippage_pct: float = Field(default=1.0, alias="MOB_SLIPPAGE_PCT")
    mob_sl_pct: float = Field(default=0.15, alias="MOB_SL_PCT")  # Tightened from 20% to 15% in v2
    mob_brokerage_per_lot: float = Field(default=40.0, alias="MOB_BROKERAGE_PER_LOT")
    mob_eod_exit_hour: int = Field(default=15, alias="MOB_EOD_EXIT_HOUR")
    mob_eod_exit_minute: int = Field(default=10, alias="MOB_EOD_EXIT_MINUTE")
    mob_starting_capital: float = Field(default=100000.0, alias="MOB_STARTING_CAPITAL")
    mob_high_score_risk_pct: float = Field(default=1.0, alias="MOB_HIGH_SCORE_RISK_PCT")
    mob_low_score_risk_pct: float = Field(default=0.5, alias="MOB_LOW_SCORE_RISK_PCT")
    mob_high_score_threshold: int = Field(default=4, alias="MOB_HIGH_SCORE_THRESHOLD")  # v2: score 4+ = full risk (was 3)

    # MOB Range-Bound Filter
    mob_range_filter_enabled: bool = Field(default=True, alias="MOB_RANGE_FILTER_ENABLED")
    mob_range_adx_threshold: float = Field(default=18.0, alias="MOB_RANGE_ADX_THRESHOLD")
    mob_range_opening_pct: float = Field(default=0.4, alias="MOB_RANGE_OPENING_PCT")  # Opening range as % of spot
    mob_range_vwap_crosses: int = Field(default=4, alias="MOB_RANGE_VWAP_CROSSES")  # Min VWAP crossings to flag range
    mob_range_min_signals: int = Field(default=2, alias="MOB_RANGE_MIN_SIGNALS")  # Need N of 3 signals to skip
    mob_range_check_bars: int = Field(default=30, alias="MOB_RANGE_CHECK_BARS")  # Bars for opening session check (~9:45)

    # MOB Risk Management
    mob_daily_sl_stop: bool = Field(default=True, alias="MOB_DAILY_SL_STOP")  # Stop after first SL hit of the day
    mob_cooldown_loss_days: int = Field(default=3, alias="MOB_COOLDOWN_LOSS_DAYS")  # Consecutive loss days before cooldown
    mob_cooldown_skip_days: int = Field(default=0, alias="MOB_COOLDOWN_SKIP_DAYS")  # Trading days to skip during cooldown (0=disabled)
    mob_min_premium: float = Field(default=50.0, alias="MOB_MIN_PREMIUM")  # Min option premium to enter

    # MOB v2 Enhanced Parameters
    mob_afternoon_enabled: bool = Field(default=False, alias="MOB_AFTERNOON_ENABLED")  # Afternoon window (default off for higher edge)
    mob_max_hold_bars: int = Field(default=45, alias="MOB_MAX_HOLD_BARS")  # Exit if no T1 within N bars
    mob_exit_slippage_pct: float = Field(default=0.5, alias="MOB_EXIT_SLIPPAGE_PCT")  # Exit slippage %
    mob_t1_partial_pct: float = Field(default=0.50, alias="MOB_T1_PARTIAL_PCT")  # % of lots to exit at T1
    mob_direction_alignment: bool = Field(default=True, alias="MOB_DIRECTION_ALIGNMENT")  # Require NIFTY/SENSEX same direction

    # ORB+VWAP Backtest Parameters
    orb_sl_type: str = Field(default="structural", alias="ORB_SL_TYPE")  # structural (ORB opposite end) or pct
    orb_sl_pct: float = Field(default=0.15, alias="ORB_SL_PCT")  # Fallback SL if structural too wide
    orb_rr_ratio: float = Field(default=2.5, alias="ORB_RR_RATIO")  # Reward:Risk ratio for target
    orb_max_trades_per_day: int = Field(default=1, alias="ORB_MAX_TRADES_PER_DAY")  # 1 trade per day per instrument
    orb_slippage_pct: float = Field(default=1.0, alias="ORB_SLIPPAGE_PCT")  # Entry slippage %
    orb_exit_slippage_pct: float = Field(default=0.5, alias="ORB_EXIT_SLIPPAGE_PCT")  # Exit slippage %
    orb_brokerage_per_lot: float = Field(default=40.0, alias="ORB_BROKERAGE_PER_LOT")
    orb_starting_capital: float = Field(default=100000.0, alias="ORB_STARTING_CAPITAL")
    orb_risk_pct: float = Field(default=1.0, alias="ORB_RISK_PCT")  # Risk % per trade
    orb_min_range_pct: float = Field(default=0.3, alias="ORB_MIN_RANGE_PCT")  # Min ORB range as % of spot
    orb_max_range_atr_mult: float = Field(default=15.0, alias="ORB_MAX_RANGE_ATR_MULT")  # Max ORB range as 1-min ATR multiple
    orb_entry_window_end_hour: int = Field(default=11, alias="ORB_ENTRY_WINDOW_END_HOUR")  # No entries after
    orb_entry_window_end_min: int = Field(default=30, alias="ORB_ENTRY_WINDOW_END_MIN")
    orb_eod_exit_hour: int = Field(default=15, alias="ORB_EOD_EXIT_HOUR")
    orb_eod_exit_minute: int = Field(default=10, alias="ORB_EOD_EXIT_MINUTE")
    orb_vwap_exit_enabled: bool = Field(default=False, alias="ORB_VWAP_EXIT_ENABLED")  # VWAP cross exit disabled  # Exit on VWAP cross
    orb_direction_lock: bool = Field(default=True, alias="ORB_DIRECTION_LOCK")  # First break locks direction
    orb_trail_after_target: bool = Field(default=True, alias="ORB_TRAIL_AFTER_TARGET")  # Trail SL after target hit
    orb_min_premium: float = Field(default=50.0, alias="ORB_MIN_PREMIUM")  # Min option premium

    # Bid-Ask / Liquidity
    max_spread_pct: float = Field(default=3.0, alias="MAX_SPREAD_PCT")  # Skip if bid-ask spread > this %

    # DhanHQ (expired options data)
    dhan_access_token: str = Field(default="", alias="DHAN_ACCESS_TOKEN")
    dhan_client_id: str = Field(default="", alias="DHAN_CLIENT_ID")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_active_instrument_list(self) -> list[str]:
        """Parse ACTIVE_INSTRUMENTS into a list of symbol names.

        Returns empty list when auto-select is on and nothing manually set.
        """
        return [s.strip().upper() for s in self.active_instruments.split(",") if s.strip()]


settings = Settings()
