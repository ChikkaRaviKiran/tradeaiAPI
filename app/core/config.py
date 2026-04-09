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
    max_concurrent_positions: int = Field(default=1, alias="MAX_CONCURRENT_POSITIONS")  # LOCKED v1.0 — 1 at a time on same underlying
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

    # Bid-Ask / Liquidity
    max_spread_pct: float = Field(default=3.0, alias="MAX_SPREAD_PCT")  # Skip if bid-ask spread > this %

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_active_instrument_list(self) -> list[str]:
        """Parse ACTIVE_INSTRUMENTS into a list of symbol names.

        Returns empty list when auto-select is on and nothing manually set.
        """
        return [s.strip().upper() for s in self.active_instruments.split(",") if s.strip()]


settings = Settings()
