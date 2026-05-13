"""Seed Tier-1 pattern definitions.

Each pattern is a pure-Python predicate on a FeatureSnapshot. Definitions
are inserted into pe_patterns table by `seed.upsert_seed_patterns()`.

Predicates are evaluated by `matches(snap)`. Trigger JSON in DB is for
documentation / UI display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from app.pattern_engine.features import FeatureSnapshot


@dataclass
class SeedPattern:
    pattern_id: str
    name: str
    direction: str               # CE / PE
    description: str
    trigger_json: dict           # human-readable definition
    exit_rule_json: dict         # sl_pct / target_pct / time_stop_min / etc
    matches: Callable[[FeatureSnapshot], bool]
    tier: int = 1


def _has(snap, *attrs) -> bool:
    return all(getattr(snap, a, None) is not None for a in attrs)


# ─────────────────────────────────────────────────────────────────────────
# 1. Opening Range Break — Bullish
# ─────────────────────────────────────────────────────────────────────────
def _orb_bullish(s: FeatureSnapshot) -> bool:
    if not _has(s, "orb_high", "ema20", "vwap"):
        return False
    return (
        s.orb_broken == "up"
        and s.spot > s.vwap
        and s.spot > s.ema20
        and (s.candle_body_pct or 0) >= 50
        and s.minute_of_day is not None
        and 570 <= s.minute_of_day <= 720   # 09:30 - 12:00
    )


orb_bullish = SeedPattern(
    pattern_id="orb_bullish",
    name="Opening Range Break — Bullish",
    direction="CE",
    description="Spot breaks above 09:15-09:30 range with body strength, above VWAP & EMA20.",
    trigger_json={
        "orb_broken": "up",
        "spot_above_vwap": True,
        "spot_above_ema20": True,
        "candle_body_pct_min": 50,
        "time_window": "09:30-12:00",
    },
    exit_rule_json={
        "sl_pct": 25, "target_pct": 50, "time_stop_min": 60,
        "trail_after_pct": 25, "trail_giveback_pct": 35,
    },
    matches=_orb_bullish,
)


# ─────────────────────────────────────────────────────────────────────────
# 2. Opening Range Break — Bearish
# ─────────────────────────────────────────────────────────────────────────
def _orb_bearish(s: FeatureSnapshot) -> bool:
    if not _has(s, "orb_low", "ema20", "vwap"):
        return False
    return (
        s.orb_broken == "down"
        and s.spot < s.vwap
        and s.spot < s.ema20
        and (s.candle_body_pct or 0) >= 50
        and s.minute_of_day is not None
        and 570 <= s.minute_of_day <= 720
    )


orb_bearish = SeedPattern(
    pattern_id="orb_bearish",
    name="Opening Range Break — Bearish",
    direction="PE",
    description="Spot breaks below 09:15-09:30 range with body strength, below VWAP & EMA20.",
    trigger_json={
        "orb_broken": "down",
        "spot_below_vwap": True,
        "spot_below_ema20": True,
        "candle_body_pct_min": 50,
        "time_window": "09:30-12:00",
    },
    exit_rule_json={
        "sl_pct": 25, "target_pct": 50, "time_stop_min": 60,
        "trail_after_pct": 25, "trail_giveback_pct": 35,
    },
    matches=_orb_bearish,
)


# ─────────────────────────────────────────────────────────────────────────
# 3. VWAP Reclaim Bullish
# ─────────────────────────────────────────────────────────────────────────
def _vwap_reclaim_bull(s: FeatureSnapshot) -> bool:
    if not _has(s, "vwap", "ema20", "vwap_dist_pct", "ema20_slope_5"):
        return False
    return (
        -0.15 <= s.vwap_dist_pct <= 0.10        # near or just above VWAP
        and s.spot >= s.vwap
        and s.ema20_slope_5 > 0
        and s.minute_of_day is not None
        and 585 <= s.minute_of_day <= 750       # 09:45 - 12:30
        and s.regime in ("trend_up", "range")
    )


vwap_reclaim_bull = SeedPattern(
    pattern_id="vwap_reclaim_bull",
    name="VWAP Reclaim — Bullish",
    direction="CE",
    description="Price reclaims VWAP with EMA20 turning up; not in volatile/trend-down regime.",
    trigger_json={
        "vwap_dist_pct_range": [-0.15, 0.10],
        "spot_at_or_above_vwap": True,
        "ema20_slope_positive": True,
        "time_window": "09:45-12:30",
        "regime_in": ["trend_up", "range"],
    },
    exit_rule_json={
        "sl_pct": 22, "target_pct": 40, "time_stop_min": 45,
        "deterioration": ["vwap_break", "ema20_break"],
    },
    matches=_vwap_reclaim_bull,
)


# ─────────────────────────────────────────────────────────────────────────
# 4. VWAP Reclaim Bearish
# ─────────────────────────────────────────────────────────────────────────
def _vwap_reclaim_bear(s: FeatureSnapshot) -> bool:
    if not _has(s, "vwap", "ema20", "vwap_dist_pct", "ema20_slope_5"):
        return False
    return (
        -0.10 <= s.vwap_dist_pct <= 0.15
        and s.spot <= s.vwap
        and s.ema20_slope_5 < 0
        and s.minute_of_day is not None
        and 585 <= s.minute_of_day <= 750
        and s.regime in ("trend_down", "range")
    )


vwap_reclaim_bear = SeedPattern(
    pattern_id="vwap_reclaim_bear",
    name="VWAP Reject — Bearish",
    direction="PE",
    description="Price rejects VWAP from below with EMA20 turning down.",
    trigger_json={
        "vwap_dist_pct_range": [-0.10, 0.15],
        "spot_at_or_below_vwap": True,
        "ema20_slope_negative": True,
        "time_window": "09:45-12:30",
        "regime_in": ["trend_down", "range"],
    },
    exit_rule_json={
        "sl_pct": 22, "target_pct": 40, "time_stop_min": 45,
        "deterioration": ["vwap_reclaim", "ema20_reclaim"],
    },
    matches=_vwap_reclaim_bear,
)


# ─────────────────────────────────────────────────────────────────────────
# 5. PDH Rejection — Bearish (price tests prior day high and rolls)
# ─────────────────────────────────────────────────────────────────────────
def _pdh_rejection(s: FeatureSnapshot) -> bool:
    if not _has(s, "pdh", "vwap"):
        return False
    near_pdh = abs(s.spot - s.pdh) / s.spot * 100 < 0.10
    return (
        near_pdh
        and s.spot < s.pdh
        and s.spot < s.vwap
        and s.minute_of_day is not None
        and 585 <= s.minute_of_day <= 810       # 09:45 - 13:30
    )


pdh_rejection = SeedPattern(
    pattern_id="pdh_rejection_bear",
    name="PDH Rejection — Bearish",
    direction="PE",
    description="Spot tests prior-day-high (within 0.1%), fails, drops below VWAP.",
    trigger_json={
        "near_pdh_pct": 0.10,
        "spot_below_pdh": True,
        "spot_below_vwap": True,
        "time_window": "09:45-13:30",
    },
    exit_rule_json={
        "sl_pct": 25, "target_pct": 45, "time_stop_min": 50,
        "deterioration": ["spot_above_pdh", "vwap_reclaim"],
    },
    matches=_pdh_rejection,
)


# ─────────────────────────────────────────────────────────────────────────
# 6. PDL Reaction — Bullish (price tests prior day low and bounces)
# ─────────────────────────────────────────────────────────────────────────
def _pdl_reaction(s: FeatureSnapshot) -> bool:
    if not _has(s, "pdl", "vwap"):
        return False
    near_pdl = abs(s.spot - s.pdl) / s.spot * 100 < 0.10
    return (
        near_pdl
        and s.spot > s.pdl
        and (s.candle_body_pct or 0) >= 40
        and s.minute_of_day is not None
        and 585 <= s.minute_of_day <= 810
    )


pdl_reaction = SeedPattern(
    pattern_id="pdl_reaction_bull",
    name="PDL Reaction — Bullish",
    direction="CE",
    description="Spot tests prior-day-low (within 0.1%) and prints a strong reclaim candle.",
    trigger_json={
        "near_pdl_pct": 0.10,
        "spot_above_pdl": True,
        "candle_body_pct_min": 40,
        "time_window": "09:45-13:30",
    },
    exit_rule_json={
        "sl_pct": 25, "target_pct": 45, "time_stop_min": 50,
        "deterioration": ["spot_below_pdl"],
    },
    matches=_pdl_reaction,
)


# ─────────────────────────────────────────────────────────────────────────
# 7. Trending Day Hold — late-morning continuation
# ─────────────────────────────────────────────────────────────────────────
def _trend_continuation_bull(s: FeatureSnapshot) -> bool:
    if not _has(s, "ema20", "ema50", "vwap"):
        return False
    return (
        s.regime == "trend_up"
        and s.spot > s.ema20 > s.ema50
        and s.spot > s.vwap
        and s.minute_of_day is not None
        and 615 <= s.minute_of_day <= 780      # 10:15 - 13:00
        and (s.range_pct_15m or 0) < 0.40       # not currently exploding (entry on pullback-ish)
    )


trend_continuation_bull = SeedPattern(
    pattern_id="trend_continuation_bull",
    name="Trend Continuation — Bullish",
    direction="CE",
    description="Mid-morning hold in confirmed up-trend on a calm 15-min window.",
    trigger_json={
        "regime": "trend_up",
        "ema_stack": "ema9>ema20>ema50",
        "spot_above_vwap": True,
        "time_window": "10:15-13:00",
        "range_pct_15m_max": 0.40,
    },
    exit_rule_json={
        "sl_pct": 30, "target_pct": 60, "time_stop_min": 75,
        "trail_after_pct": 30, "trail_giveback_pct": 40,
    },
    matches=_trend_continuation_bull,
)


# Registry
SEED_PATTERNS: list[SeedPattern] = [
    orb_bullish,
    orb_bearish,
    vwap_reclaim_bull,
    vwap_reclaim_bear,
    pdh_rejection,
    pdl_reaction,
    trend_continuation_bull,
]


def get_pattern(pattern_id: str) -> Optional[SeedPattern]:
    for p in SEED_PATTERNS:
        if p.pattern_id == pattern_id:
            return p
    return None


def evaluate_all(snap: FeatureSnapshot) -> list[SeedPattern]:
    """Return all seed patterns whose predicates match this snapshot."""
    return [p for p in SEED_PATTERNS if p.matches(snap)]
