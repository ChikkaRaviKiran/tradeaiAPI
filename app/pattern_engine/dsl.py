"""Small JSON DSL for pattern triggers + exit-rule normalization.

Trigger format (stored in pe_patterns.trigger_json):
{
  "all": [                       # AND of conditions; "any" = OR
    {"f": "vwap_dist_pct", "op": "between", "v": [-0.15, 0.10]},
    {"f": "spot",         "op": "gte",     "v_field": "vwap"},
    {"f": "ema20_slope_5","op": "gt",      "v": 0},
    {"f": "minute_of_day","op": "between", "v": [585, 750]},
    {"f": "regime",       "op": "in",      "v": ["trend_up", "range"]},
    {"f": "orb_broken",   "op": "eq",      "v": "up"}
  ]
}

Operators: eq, ne, gt, gte, lt, lte, between (v=[lo,hi]), in (v=[...]), not_in,
           is_null, not_null, abs_lt, abs_gt
Value source: 'v' (literal) OR 'v_field' (compare against another snapshot field).
Top-level keys 'all' (AND) and 'any' (OR) may be nested.

Exit rule format (stored in pe_patterns.exit_rule_json):
{
  "sl_pct": 25, "target_pct": 50, "time_stop_min": 60,
  "leverage": 8.0,                   # spot% -> premium% multiplier (synthetic)
  "trail_after_pct": 25, "trail_giveback_pct": 35
}
"""

from __future__ import annotations

from typing import Any


VALID_OPS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "between", "in", "not_in",
    "is_null", "not_null",
    "abs_lt", "abs_gt",
}


def _resolve(side: dict, snap_dict: dict) -> Any:
    if "v_field" in side:
        return snap_dict.get(side["v_field"])
    return side.get("v")


def _eval_cond(cond: dict, snap_dict: dict) -> bool:
    f = cond.get("f")
    op = cond.get("op")
    if f is None or op not in VALID_OPS:
        return False
    lhs = snap_dict.get(f)

    if op == "is_null":
        return lhs is None
    if op == "not_null":
        return lhs is not None

    if lhs is None:
        return False  # null fails any comparison

    rhs = _resolve(cond, snap_dict)

    try:
        if op == "eq":  return lhs == rhs
        if op == "ne":  return lhs != rhs
        if op == "gt":  return lhs > rhs
        if op == "gte": return lhs >= rhs
        if op == "lt":  return lhs < rhs
        if op == "lte": return lhs <= rhs
        if op == "between":
            lo, hi = rhs[0], rhs[1]
            return lo <= lhs <= hi
        if op == "in":     return lhs in rhs
        if op == "not_in": return lhs not in rhs
        if op == "abs_lt": return abs(lhs) < rhs
        if op == "abs_gt": return abs(lhs) > rhs
    except Exception:
        return False
    return False


def evaluate_trigger(trigger_json: dict | None, snap_dict: dict) -> bool:
    """Evaluate a JSON-DSL trigger against a snapshot dict.

    Empty / missing trigger never matches (defensive).
    """
    if not trigger_json or not isinstance(trigger_json, dict):
        return False
    return _eval_node(trigger_json, snap_dict)


def _eval_node(node: dict, snap_dict: dict) -> bool:
    if "all" in node:
        return all(_eval_child(c, snap_dict) for c in node["all"])
    if "any" in node:
        return any(_eval_child(c, snap_dict) for c in node["any"])
    # bare condition
    return _eval_cond(node, snap_dict)


def _eval_child(c: dict, snap_dict: dict) -> bool:
    if isinstance(c, dict) and ("all" in c or "any" in c):
        return _eval_node(c, snap_dict)
    return _eval_cond(c, snap_dict)


# Default exit rule + sane bounds
DEFAULT_EXIT = {
    "sl_pct": 25.0,
    "target_pct": 50.0,
    "time_stop_min": 60,
    "leverage": 8.0,
}


def normalize_exit(exit_json: dict | None) -> dict:
    out = dict(DEFAULT_EXIT)
    if isinstance(exit_json, dict):
        for k, v in exit_json.items():
            out[k] = v
    return out
