"""Classic + Camarilla pivots, confluence clustering, and condor leg building.

Ported (near-verbatim) from AgenticTrade's `levels/pivots.py`,
`levels/confluence.py` and `strategy/selling.py::build_condor` — this is the
same logic validated in the AgenticTrade backtester, adapted to operate on
plain floats / pandas DataFrames sourced from TradeAI's own `IndexCandle`
table rather than the Dhan historical API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


def classic_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict:
    p = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * p - prev_low
    s1 = 2 * p - prev_high
    r2 = p + (prev_high - prev_low)
    s2 = p - (prev_high - prev_low)
    r3 = prev_high + 2 * (p - prev_low)
    s3 = prev_low - 2 * (prev_high - p)
    return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def camarilla_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict:
    rng = prev_high - prev_low
    c = prev_close
    return {
        "R1": c + rng * 1.1 / 12,
        "R2": c + rng * 1.1 / 6,
        "R3": c + rng * 1.1 / 4,
        "R4": c + rng * 1.1 / 2,
        "S1": c - rng * 1.1 / 12,
        "S2": c - rng * 1.1 / 6,
        "S3": c - rng * 1.1 / 4,
        "S4": c - rng * 1.1 / 2,
    }


def opening_range(intraday_df: pd.DataFrame, minutes: int = 15) -> Optional[tuple]:
    """intraday_df: candles for the target day only, indexed by timestamp,
    already filtered to <= "now" by the caller. Returns (high, low) of the
    first `minutes` of the session, or None if not enough data yet."""
    if intraday_df.empty:
        return None
    session_start = intraday_df.index[0]
    window = intraday_df[intraday_df.index <= session_start + pd.Timedelta(minutes=minutes)]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())


def round_levels(spot: float, step: int = 100, span: int = 2) -> list:
    base = round(spot / step) * step
    return [base + i * step for i in range(-span, span + 1)]


@dataclass
class ConfluenceLevel:
    price: float
    sources: list = field(default_factory=list)

    @property
    def confidence(self) -> int:
        return len(self.sources)


def build_confluence_levels(level_sources: dict, tolerance_pct: float = 0.0015) -> list:
    """level_sources: {"pivot_R1": 24120, ...} (name -> price). Clusters nearby
    levels (within tolerance_pct of each other) into ConfluenceLevel groups,
    sorted ascending by price."""
    items = sorted(level_sources.items(), key=lambda kv: kv[1])
    clusters: list = []
    for name, price in items:
        placed = False
        for cluster in clusters:
            if abs(price - cluster.price) / cluster.price <= tolerance_pct:
                cluster.sources.append(name)
                cluster.price = sum(level_sources[s] for s in cluster.sources) / len(cluster.sources)
                placed = True
                break
        if not placed:
            clusters.append(ConfluenceLevel(price=price, sources=[name]))
    return sorted(clusters, key=lambda c: c.price)


def nearest_resistance(levels: list, spot: float) -> Optional[ConfluenceLevel]:
    above = [l for l in levels if l.price > spot]
    return min(above, key=lambda l: l.price - spot) if above else None


def nearest_support(levels: list, spot: float) -> Optional[ConfluenceLevel]:
    below = [l for l in levels if l.price < spot]
    return max(below, key=lambda l: l.price) if below else None


@dataclass
class CondorLegs:
    short_ce_strike: float
    short_pe_strike: float
    long_ce_strike: float
    long_pe_strike: float


def build_condor(resistance_strike: float, support_strike: float, wing_width: int, strike_step: int) -> CondorLegs:
    """Short strikes placed at the confluence resistance/support levels
    (rounded to the nearest tradable strike); long wings further out for
    defined risk."""
    def round_to_step(x: float) -> float:
        return round(x / strike_step) * strike_step

    short_ce = round_to_step(resistance_strike)
    short_pe = round_to_step(support_strike)
    long_ce = short_ce + wing_width
    long_pe = short_pe - wing_width
    return CondorLegs(short_ce_strike=short_ce, short_pe_strike=short_pe, long_ce_strike=long_ce, long_pe_strike=long_pe)
