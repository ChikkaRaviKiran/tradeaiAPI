"""Seed Tier-1 pattern definitions in JSON DSL format.

Definitions are written into pe_patterns at startup by `seed.upsert_seed_patterns`.
On subsequent restarts the seeder REFRESHES trigger/exit JSON for built-in
patterns UNLESS the pattern's notes contain '[locked]' — that marker tells the
seeder the operator has hand-tuned this pattern via the UI and it must not be
overwritten on container restart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeedPattern:
    pattern_id: str
    name: str
    direction: str
    description: str
    trigger_json: dict
    exit_rule_json: dict
    tier: int = 1


orb_bullish = SeedPattern(
    pattern_id="orb_bullish",
    name="Opening Range Break — Bullish",
    direction="CE",
    description="Spot breaks above 09:15-09:30 range with body strength, above VWAP & EMA20.",
    trigger_json={
        "all": [
            {"f": "orb_broken", "op": "eq", "v": "up"},
            {"f": "spot", "op": "gte", "v_field": "vwap"},
            {"f": "spot", "op": "gte", "v_field": "ema20"},
            {"f": "candle_body_pct", "op": "gte", "v": 50},
            {"f": "minute_of_day", "op": "between", "v": [570, 720]},
        ]
    },
    exit_rule_json={"sl_pct": 25, "target_pct": 50, "time_stop_min": 60, "leverage": 8.0},
)

orb_bearish = SeedPattern(
    pattern_id="orb_bearish",
    name="Opening Range Break — Bearish",
    direction="PE",
    description="Spot breaks below 09:15-09:30 range with body strength, below VWAP & EMA20.",
    trigger_json={
        "all": [
            {"f": "orb_broken", "op": "eq", "v": "down"},
            {"f": "spot", "op": "lte", "v_field": "vwap"},
            {"f": "spot", "op": "lte", "v_field": "ema20"},
            {"f": "candle_body_pct", "op": "gte", "v": 50},
            {"f": "minute_of_day", "op": "between", "v": [570, 720]},
        ]
    },
    exit_rule_json={"sl_pct": 25, "target_pct": 50, "time_stop_min": 60, "leverage": 8.0},
)

vwap_reclaim_bull = SeedPattern(
    pattern_id="vwap_reclaim_bull",
    name="VWAP Reclaim — Bullish",
    direction="CE",
    description="Price reclaims VWAP with EMA20 turning up.",
    trigger_json={
        "all": [
            {"f": "vwap_dist_pct", "op": "between", "v": [-0.20, 0.15]},
            {"f": "spot", "op": "gte", "v_field": "vwap"},
            {"f": "ema20_slope_5", "op": "gt", "v": 0},
            {"f": "minute_of_day", "op": "between", "v": [585, 750]},
        ]
    },
    exit_rule_json={"sl_pct": 22, "target_pct": 40, "time_stop_min": 45, "leverage": 8.0},
)

vwap_reclaim_bear = SeedPattern(
    pattern_id="vwap_reclaim_bear",
    name="VWAP Reject — Bearish",
    direction="PE",
    description="Price rejects VWAP from below with EMA20 turning down.",
    trigger_json={
        "all": [
            {"f": "vwap_dist_pct", "op": "between", "v": [-0.15, 0.20]},
            {"f": "spot", "op": "lte", "v_field": "vwap"},
            {"f": "ema20_slope_5", "op": "lt", "v": 0},
            {"f": "minute_of_day", "op": "between", "v": [585, 750]},
        ]
    },
    exit_rule_json={"sl_pct": 22, "target_pct": 40, "time_stop_min": 45, "leverage": 8.0},
)

pdh_rejection = SeedPattern(
    pattern_id="pdh_rejection_bear",
    name="PDH Rejection — Bearish",
    direction="PE",
    description="Spot tests prior-day-high (within 0.10%), fails, drops below VWAP.",
    trigger_json={
        "all": [
            {"f": "dist_to_pdh_pct", "op": "abs_lt", "v": 0.10},
            {"f": "spot", "op": "lte", "v_field": "pdh"},
            {"f": "spot", "op": "lte", "v_field": "vwap"},
            {"f": "minute_of_day", "op": "between", "v": [585, 810]},
        ]
    },
    exit_rule_json={"sl_pct": 25, "target_pct": 45, "time_stop_min": 50, "leverage": 8.0},
)

pdl_reaction = SeedPattern(
    pattern_id="pdl_reaction_bull",
    name="PDL Reaction — Bullish",
    direction="CE",
    description="Spot tests prior-day-low (within 0.10%) and prints a strong reclaim candle.",
    trigger_json={
        "all": [
            {"f": "dist_to_pdl_pct", "op": "abs_lt", "v": 0.10},
            {"f": "spot", "op": "gte", "v_field": "pdl"},
            {"f": "candle_body_pct", "op": "gte", "v": 40},
            {"f": "minute_of_day", "op": "between", "v": [585, 810]},
        ]
    },
    exit_rule_json={"sl_pct": 25, "target_pct": 45, "time_stop_min": 50, "leverage": 8.0},
)

trend_continuation_bull = SeedPattern(
    pattern_id="trend_continuation_bull",
    name="Trend Continuation — Bullish",
    direction="CE",
    description="Mid-morning hold above EMA20 + VWAP on a calm 15-min window.",
    trigger_json={
        "all": [
            {"f": "spot", "op": "gte", "v_field": "ema20"},
            {"f": "spot", "op": "gte", "v_field": "vwap"},
            {"f": "ema20_slope_5", "op": "gt", "v": 0},
            {"f": "minute_of_day", "op": "between", "v": [615, 780]},
            {"f": "range_pct_15m", "op": "lt", "v": 0.40},
        ]
    },
    exit_rule_json={"sl_pct": 30, "target_pct": 60, "time_stop_min": 75, "leverage": 8.0},
)


SEED_PATTERNS: list[SeedPattern] = [
    orb_bullish,
    orb_bearish,
    vwap_reclaim_bull,
    vwap_reclaim_bear,
    pdh_rejection,
    pdl_reaction,
    trend_continuation_bull,
]
