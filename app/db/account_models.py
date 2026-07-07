"""Multi-account + multi-strategy DB models.

Modeled on the OptionSelling schema so the same UI patterns port over.
Adds two tables:

- ``broker_accounts``:  one row per broker login (AngelOne / Kite / Dhan).
  Multiple accounts of the same broker are allowed.
- ``strategy_instances``: one row per (strategy_type, account) pairing.
  Each row is an independent, individually-toggled strategy configuration.

The existing single-strategy JSON (``atl_straddle_settings.json``) is
migrated into one seed ``strategy_instances`` row on first boot — see
``app.db.migrations.seed_multi_account_defaults``.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.db.models import Base, _now_ist


# Public constants — imported by API + UI validation paths.
SUPPORTED_BROKERS = ("angel", "kite", "dhan")
SUPPORTED_STRATEGY_TYPES = ("ATM_STRADDLE", "OTM_STRANGLE")


class BrokerAccount(Base):
    """A single broker login/session.

    Credentials are optional at the row level because Kite/Dhan re-auth
    daily via OAuth or token refresh; only ``client_id`` + ``broker`` +
    ``name`` are required to create a row.
    """

    __tablename__ = "broker_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)                 # Display name
    broker = Column(String(20), nullable=False, default="angel")  # angel | kite | dhan
    client_id = Column(String(50), nullable=False)             # Broker client id (AngelOne login, Kite user_id, Dhan client_id)
    api_key = Column(String(200), nullable=True, default="")
    api_secret = Column(String(200), nullable=True, default="")
    password = Column(String(200), nullable=True, default="")
    mpin = Column(String(50), nullable=True, default="")
    totp_secret = Column(String(200), nullable=True, default="")
    access_token = Column(Text, nullable=True)                 # Kite / Dhan access token
    refresh_token = Column(Text, nullable=True)
    auth_token = Column(Text, nullable=True)                   # AngelOne JWT
    feed_token = Column(Text, nullable=True)                   # AngelOne feed token
    login_method = Column(String(20), nullable=False, default="manual")  # manual | oauth
    paper_trading = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_data_feed = Column(Boolean, nullable=False, default=False)      # Preferred source for market data
    is_primary = Column(Boolean, nullable=False, default=False)        # Single default account for global fallbacks
    proxy_url = Column(String(300), nullable=True, default="")         # socks5://user:pass@ip:port
    proxy_ip = Column(String(50), nullable=True, default="")
    proxy_instance_name = Column(String(100), nullable=True, default="")
    available_funds = Column(Float, nullable=True, default=0.0)
    used_funds = Column(Float, nullable=True, default=0.0)
    # Per-account daily-loss kill switch. When ``kill_switch_enabled`` is
    # True and the account's realised+unrealised PnL drops below
    # ``-daily_loss_limit``, the watchdog force-closes only this account's
    # open positions and blocks new-entry orders routed through it until
    # midnight IST (or an explicit reset via the UI).
    kill_switch_enabled = Column(Boolean, nullable=False, default=True)
    daily_loss_limit = Column(Float, nullable=False, default=6000.0)
    last_connection_status = Column(String(30), nullable=True, default="unknown")  # connected | disconnected | error
    last_connection_error = Column(Text, nullable=True)
    last_connected_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now_ist)
    updated_at = Column(DateTime, default=_now_ist, onupdate=_now_ist)


class StrategyInstance(Base):
    """One configured strategy row.

    Multiple rows may target the same account. Each row is scheduled and
    toggled independently.  The scanner runtime looks up the matching
    ``BrokerAccount`` by ``account_id`` and routes orders there.

    ``params_json`` stores any strategy-specific overrides that don't fit
    the flat column schema (smart-mode thresholds, indicator tuning etc.).
    """

    __tablename__ = "strategy_instances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_type = Column(String(30), nullable=False, default="ATM_STRADDLE")  # See SUPPORTED_STRATEGY_TYPES
    account_id = Column(Integer, nullable=True)                # FK-ish → broker_accounts.id (nullable = paper)
    index = Column(String(20), nullable=False, default="NIFTY")   # NIFTY | BANKNIFTY | SENSEX
    trading_day = Column(String(20), nullable=False, default="Daily")  # Daily / Monday .. Friday
    entry_time = Column(String(10), nullable=False, default="09:20")   # HH:MM
    exit_time = Column(String(10), nullable=False, default="15:15")    # HH:MM
    lots = Column(Integer, nullable=False, default=1)
    strike_interval = Column(Integer, nullable=False, default=50)
    strike_mode = Column(String(20), nullable=False, default="ATM")    # ATM | STRANGLE | ITM
    otm_strikes = Column(Integer, nullable=False, default=0)           # OTM offset in strike-steps
    static_legs = Column(Boolean, nullable=False, default=False)
    adjustment_points = Column(Integer, nullable=False, default=1)
    rolling_points = Column(Integer, nullable=False, default=300)
    sl_type = Column(String(20), nullable=False, default="none")       # none | premium_pct | spot | amount
    sl_lower = Column(Float, nullable=True, default=0)
    sl_upper = Column(Float, nullable=True, default=0)
    sl_amount = Column(Float, nullable=True, default=0)
    first_straddle_sl_pct = Column(Integer, nullable=False, default=100)
    reform_straddle_sl_pct = Column(Integer, nullable=False, default=60)
    hedge_mode = Column(String(20), nullable=False, default="none")    # none | premium | otm_points
    hedge_premium = Column(Float, nullable=True, default=3)
    hedge_otm_points = Column(Integer, nullable=True, default=500)
    hedge_lots = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    live_execution = Column(Boolean, nullable=False, default=False)    # False = paper, True = live orders
    display_name = Column(String(120), nullable=True, default="")      # Optional friendly label
    params_json = Column(Text, nullable=True)                          # Strategy-specific overrides
    created_at = Column(DateTime, default=_now_ist)
    updated_at = Column(DateTime, default=_now_ist, onupdate=_now_ist)
