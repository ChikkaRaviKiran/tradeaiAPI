"""Instrument registry — defines all tradeable instruments and their properties.

Each instrument knows its symbol, exchange, lot size, strike interval, and
how to build option symbol names for broker APIs.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class Exchange(str, enum.Enum):
    NSE = "NSE"
    NFO = "NFO"
    BSE = "BSE"
    BFO = "BFO"
    MCX = "MCX"


class InstrumentType(str, enum.Enum):
    INDEX = "INDEX"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for a single tradeable instrument."""

    symbol: str               # e.g. "NIFTY", "BANKNIFTY", "TCS"
    display_name: str         # e.g. "NIFTY 50", "Bank Nifty", "TCS Ltd"
    exchange: Exchange        # Primary exchange
    instrument_type: InstrumentType
    lot_size: int             # Contract lot size (1 for equity)
    strike_interval: float    # Option strike gap (50 for NIFTY, 100 for BANKNIFTY)
    token: str                # AngelOne symbol token for spot/index data
    futures_symbol_prefix: str = ""   # e.g. "NIFTY" for NIFTY FUTIDX
    option_symbol_prefix: str = ""    # e.g. "NIFTY" for NIFTY option chain
    is_index: bool = False    # True for indices (no direct volume)
    enabled: bool = True      # Whether this instrument is active

    def nearest_strike(self, price: float, option_type: str = "CE") -> float:
        """Round price to nearest valid strike for this instrument."""
        base = round(price / self.strike_interval) * self.strike_interval
        # For CE: go slightly OTM (up), for PE: go slightly OTM (down)
        if option_type == "CE" and base < price:
            base += self.strike_interval
        elif option_type == "PE" and base > price:
            base -= self.strike_interval
        return base

    def build_option_symbol(self, expiry: str, strike: float, option_type: str) -> str:
        """Build NFO trading symbol. e.g. NIFTY17MAR202622500CE"""
        prefix = self.option_symbol_prefix or self.symbol
        return f"{prefix}{expiry}{int(strike)}{option_type}"

    def build_futures_symbol(self, expiry: str) -> str:
        """Build futures trading symbol. e.g. NIFTY17MAR2026FUT"""
        prefix = self.futures_symbol_prefix or self.symbol
        return f"{prefix}{expiry}FUT"


# ── Pre-configured Instruments ────────────────────────────────────────────────

NIFTY = InstrumentConfig(
    symbol="NIFTY",
    display_name="NIFTY 50",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.INDEX,
    lot_size=65,
    strike_interval=50,
    token="99926000",
    futures_symbol_prefix="NIFTY",
    option_symbol_prefix="NIFTY",
    is_index=True,
)

BANKNIFTY = InstrumentConfig(
    symbol="BANKNIFTY",
    display_name="Bank Nifty",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.INDEX,
    lot_size=30,
    strike_interval=100,
    token="99926009",
    futures_symbol_prefix="BANKNIFTY",
    option_symbol_prefix="BANKNIFTY",
    is_index=True,
    enabled=False,  # Disabled — only NIFTY active
)

FINNIFTY = InstrumentConfig(
    symbol="FINNIFTY",
    display_name="Fin Nifty",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.INDEX,
    lot_size=60,
    strike_interval=50,
    token="99926037",
    futures_symbol_prefix="FINNIFTY",
    option_symbol_prefix="FINNIFTY",
    is_index=True,
    enabled=False,  # Disabled — only NIFTY active
)

MIDCPNIFTY = InstrumentConfig(
    symbol="MIDCPNIFTY",
    display_name="Midcap Nifty",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.INDEX,
    lot_size=120,
    strike_interval=25,
    token="99926074",
    futures_symbol_prefix="MIDCPNIFTY",
    option_symbol_prefix="MIDCPNIFTY",
    is_index=True,
    enabled=False,  # Disabled — focusing on NIFTY, BANKNIFTY, FINNIFTY
)

# ── Stock definitions (equity options on NSE) ────────────────────────────

def _equity(symbol: str, name: str, token: str, lot_size: int, strike_int: float, enabled: bool = False) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol,
        display_name=name,
        exchange=Exchange.NSE,
        instrument_type=InstrumentType.EQUITY,
        lot_size=lot_size,
        strike_interval=strike_int,
        token=token,
        option_symbol_prefix=symbol,
        is_index=False,
        enabled=enabled,
    )


# Top traded F&O stocks — tokens from AngelOne instrument master
# Tokens will be resolved dynamically if empty
RELIANCE = _equity("RELIANCE", "Reliance Industries", "2885", 500, 20)
TCS = _equity("TCS", "Tata Consultancy", "11536", 175, 50)
HDFCBANK = _equity("HDFCBANK", "HDFC Bank", "1333", 550, 20)
INFY = _equity("INFY", "Infosys", "1594", 400, 25)
ICICIBANK = _equity("ICICIBANK", "ICICI Bank", "4963", 700, 20)
SBIN = _equity("SBIN", "State Bank of India", "3045", 750, 10)
BHARTIARTL = _equity("BHARTIARTL", "Bharti Airtel", "10604", 475, 20)
ITC = _equity("ITC", "ITC Ltd", "1660", 1600, 5)
TATAMOTORS = _equity("TATAMOTORS", "Tata Motors", "3456", 575, 10)
LT = _equity("LT", "Larsen & Toubro", "11483", 175, 50)


# ── Registry ──────────────────────────────────────────────────────────────────

_ALL_INSTRUMENTS: dict[str, InstrumentConfig] = {
    "NIFTY": NIFTY,
    "BANKNIFTY": BANKNIFTY,
    "FINNIFTY": FINNIFTY,
    "MIDCPNIFTY": MIDCPNIFTY,
    "RELIANCE": RELIANCE,
    "TCS": TCS,
    "HDFCBANK": HDFCBANK,
    "INFY": INFY,
    "ICICIBANK": ICICIBANK,
    "SBIN": SBIN,
    "BHARTIARTL": BHARTIARTL,
    "ITC": ITC,
    "TATAMOTORS": TATAMOTORS,
    "LT": LT,
}


def get_instrument(symbol: str) -> Optional[InstrumentConfig]:
    """Look up an instrument by symbol name."""
    return _ALL_INSTRUMENTS.get(symbol.upper())


def get_all_instruments() -> list[InstrumentConfig]:
    """Return all registered instruments."""
    return list(_ALL_INSTRUMENTS.values())


def get_enabled_instruments() -> list[InstrumentConfig]:
    """Return only instruments marked as enabled."""
    return [i for i in _ALL_INSTRUMENTS.values() if i.enabled]


def get_indices() -> list[InstrumentConfig]:
    """Return all index instruments."""
    return [i for i in _ALL_INSTRUMENTS.values() if i.is_index]


def get_equities() -> list[InstrumentConfig]:
    """Return all equity instruments."""
    return [i for i in _ALL_INSTRUMENTS.values() if i.instrument_type == InstrumentType.EQUITY]


def register_instrument(config: InstrumentConfig) -> None:
    """Register a new instrument at runtime."""
    _ALL_INSTRUMENTS[config.symbol.upper()] = config


def sync_lot_sizes_from_broker(client) -> None:
    """Update lot sizes for all registered instruments from SmartAPI instrument master.

    Called once at startup after AngelOne authentication. Replaces hardcoded
    lot sizes with live values from the exchange.

    Args:
        client: AngelOneClient instance (must have instrument master loaded).
    """
    from dataclasses import replace as dc_replace

    updated = 0
    for symbol, inst in list(_ALL_INSTRUMENTS.items()):
        live_lot = client.get_lot_size(symbol)
        if live_lot and live_lot != inst.lot_size:
            logger.info(
                "Lot size updated: %s %d → %d (from SmartAPI)",
                symbol, inst.lot_size, live_lot,
            )
            _ALL_INSTRUMENTS[symbol] = dc_replace(inst, lot_size=live_lot)
            updated += 1
        elif live_lot:
            logger.debug("Lot size confirmed: %s = %d", symbol, live_lot)

    if updated:
        logger.info("Synced %d lot sizes from SmartAPI instrument master", updated)
    else:
        logger.info("All lot sizes match SmartAPI — no updates needed")
