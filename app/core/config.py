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

    # Email
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    alert_email_to: str = Field(default="", alias="ALERT_EMAIL_TO")

    # Trading
    initial_capital: float = Field(default=100000, alias="INITIAL_CAPITAL")
    max_trades_per_day: int = Field(default=2, alias="MAX_TRADES_PER_DAY")
    max_daily_loss_pct: float = Field(default=2.0, alias="MAX_DAILY_LOSS_PCT")
    consecutive_loss_limit: int = Field(default=3, alias="CONSECUTIVE_LOSS_LIMIT")
    nifty_lot_size: int = Field(default=50, alias="NIFTY_LOT_SIZE")

    # System
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
