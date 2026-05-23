"""ATL straddle settings persistence shared by API and runtime engine."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ATL_SETTINGS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "strategy_type": "ATM_STRADDLE",
    "index": "NIFTY",
    "trading_day": "Daily",
    "entry_time": "09:20",
    "exit_time": "15:15",
    "strike_mode": "ATM",
    "otm_strikes": 0,
    # When True: enter the two legs once at entry_time and HOLD them
    # unchanged until exit_time. No strangle-to-straddle conversion on
    # touch, no rolling, no adjustment/reform. This matches the backtest
    # simulator exactly (Phase-3a / ULTIMATE sweep results were produced
    # under this rule).
    "static_legs": False,
    "lots": 1,
    "strike_interval": 50,
    "offset_points": 500,
    "rolling_points": 300,
    "sl_type": "premium_pct",
    "sl_lower": 0,
    "sl_upper": 0,
    "first_straddle_sl_pct": 100,
    "reform_straddle_sl_pct": 60,
    "hedge_mode": "none",
    "hedge_enabled": False,
    "hedge_premium": 3,
    "hedge_otm_points": 500,
    "hedge_lots": 0,
    "execution_account": "Primary",
    # ── Smart-condition reform mode ─────────────────────────────────
    # Activated ONLY when `adjustment_points == 0`. When >0, the legacy
    # fixed-distance reform rule runs and these keys are ignored. This
    # preserves the exact current behavior for every existing user.
    #
    # `adjustment_points` is intentionally NOT in defaults — its absence
    # means "fall back to rolling_points" (legacy path). Setting it to 0
    # via the API/UI is the explicit opt-in for smart mode.
    "smart_ratio_trigger": 2.0,             # CE/PE premium imbalance to allow a reform
    "smart_ratio_trigger_expiry": 1.6,      # tighter on expiry day (gamma is brutal)
    "smart_use_5min_close": True,           # require trend confirm (5-min close avg vs LTP)
    "smart_reform_cooldown_min": 20,        # min minutes between consecutive reforms
    "smart_reform_cooldown_min_expiry": 10, # shorter cooldown on expiry day
    "smart_max_reforms_per_day": 4,         # hard cap on intraday reforms
    "smart_no_reform_after": "14:45",       # last clock time a reform may fire
    "smart_no_reform_after_expiry": "14:30",
    # Per-instrument minimum drift floor (in spot points) before any
    # smart reform is considered. 0 = use built-in default for that index.
    "smart_min_drift_nifty": 0,             # built-in default: 50
    "smart_min_drift_banknifty": 0,         # built-in default: 150
    "smart_min_drift_sensex": 0,            # built-in default: 200
    # Emergency override: if drift exceeds this multiplier × min_drift,
    # reform regardless of premium ratio (handles vol-crush edge cases
    # where ratio stays low despite a large directional move).
    "smart_force_reform_drift_mult": 3.0,
}


def atl_settings_path() -> str:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(backend_root, "data", "atl_straddle_settings.json")


def load_atl_settings() -> dict[str, Any]:
    path = atl_settings_path()
    if not os.path.exists(path):
        return dict(ATL_SETTINGS_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(ATL_SETTINGS_DEFAULTS)
        out = dict(ATL_SETTINGS_DEFAULTS)
        out.update(data)
        return normalize_atl_settings(out)
    except Exception:
        logger.exception("Failed to load ATL settings; using defaults")
        return dict(ATL_SETTINGS_DEFAULTS)


def normalize_atl_settings(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(ATL_SETTINGS_DEFAULTS)
    out.update(payload or {})

    out["enabled"] = bool(out.get("enabled", False))
    out["strategy_type"] = "ATM_STRADDLE"
    out["index"] = str(out.get("index", "NIFTY")).upper()
    if out["index"] not in {"NIFTY", "BANKNIFTY", "SENSEX"}:
        out["index"] = "NIFTY"

    day = str(out.get("trading_day", "Daily")).strip().title()
    if day not in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Daily"}:
        day = "Daily"
    out["trading_day"] = day

    out["entry_time"] = str(out.get("entry_time", "09:20"))
    out["exit_time"] = str(out.get("exit_time", "15:15"))
    mode = str(out.get("strike_mode", "ATM")).upper()
    if mode not in {"ATM", "ITM", "STRANGLE"}:
        mode = "ATM"
    out["strike_mode"] = mode
    out["lots"] = max(1, int(out.get("lots", 1)))
    out["strike_interval"] = max(1, int(out.get("strike_interval", 50)))
    # otm_strikes is the OTM offset expressed in strike-steps (0/1/2/3...).
    # When strike_mode == STRANGLE, this drives offset_points so users can
    # pick "+2 OTM" without hand-computing points.
    out["otm_strikes"] = max(0, int(out.get("otm_strikes", 0)))
    out["static_legs"] = bool(out.get("static_legs", False))
    if mode == "STRANGLE" and out["otm_strikes"] > 0:
        out["offset_points"] = out["otm_strikes"] * out["strike_interval"]
    elif mode == "ATM":
        out["offset_points"] = 0
    else:
        out["offset_points"] = max(0, int(out.get("offset_points", 500)))
    out["rolling_points"] = max(1, int(out.get("rolling_points", 300)))

    sl_type = str(out.get("sl_type", "premium_pct")).strip().lower()
    if sl_type not in {"none", "premium_pct", "spot"}:
        sl_type = "premium_pct"
    out["sl_type"] = sl_type
    out["sl_lower"] = max(0, int(out.get("sl_lower", 0)))
    out["sl_upper"] = max(0, int(out.get("sl_upper", 0)))

    out["first_straddle_sl_pct"] = max(1, int(out.get("first_straddle_sl_pct", 100)))
    out["reform_straddle_sl_pct"] = max(1, int(out.get("reform_straddle_sl_pct", 60)))

    hedge_mode = str(out.get("hedge_mode", "none")).strip().lower()
    if hedge_mode not in {"none", "premium", "otm_points"}:
        # Backward compatibility with previous boolean-only flag.
        hedge_mode = "premium" if bool(out.get("hedge_enabled", False)) else "none"
    out["hedge_mode"] = hedge_mode
    out["hedge_enabled"] = hedge_mode != "none"
    out["hedge_premium"] = max(1, int(out.get("hedge_premium", 3)))
    out["hedge_otm_points"] = max(1, int(out.get("hedge_otm_points", out.get("offset_points", 500))))
    out["hedge_lots"] = max(0, int(out.get("hedge_lots", 0)))
    out["execution_account"] = str(out.get("execution_account", "Primary"))

    if sl_type == "none":
        out["sl_lower"] = 0
        out["sl_upper"] = 0

    if sl_type == "spot":
        out["first_straddle_sl_pct"] = 100

    if hedge_mode == "none":
        out["hedge_lots"] = 0

    # ── Smart-mode normalization ─────────────────────────────────────
    # `adjustment_points` is optional. If the user provided it, accept it
    # as a non-negative int (0 == smart mode). If absent, leave it absent
    # so the scanner's legacy fallback (`rolling_points`) kicks in.
    if "adjustment_points" in (payload or {}):
        try:
            adj = int(payload.get("adjustment_points"))
        except Exception:
            adj = 0
        out["adjustment_points"] = max(0, adj)

    def _clamp_float(key: str, lo: float, hi: float, default: float) -> None:
        try:
            v = float(out.get(key, default))
        except Exception:
            v = default
        out[key] = max(lo, min(hi, v))

    def _clamp_int(key: str, lo: int, hi: int, default: int) -> None:
        try:
            v = int(out.get(key, default))
        except Exception:
            v = default
        out[key] = max(lo, min(hi, v))

    def _clamp_time(key: str, default: str) -> None:
        raw = str(out.get(key, default))
        try:
            h, m = [int(x) for x in raw.split(":")]
            if 0 <= h <= 23 and 0 <= m <= 59:
                out[key] = f"{h:02d}:{m:02d}"
                return
        except Exception:
            pass
        out[key] = default

    _clamp_float("smart_ratio_trigger", 1.05, 10.0, 2.0)
    _clamp_float("smart_ratio_trigger_expiry", 1.05, 10.0, 1.6)
    out["smart_use_5min_close"] = bool(out.get("smart_use_5min_close", True))
    _clamp_int("smart_reform_cooldown_min", 0, 240, 20)
    _clamp_int("smart_reform_cooldown_min_expiry", 0, 240, 10)
    _clamp_int("smart_max_reforms_per_day", 0, 50, 4)
    _clamp_time("smart_no_reform_after", "14:45")
    _clamp_time("smart_no_reform_after_expiry", "14:30")
    _clamp_int("smart_min_drift_nifty", 0, 5000, 0)
    _clamp_int("smart_min_drift_banknifty", 0, 5000, 0)
    _clamp_int("smart_min_drift_sensex", 0, 10000, 0)
    _clamp_float("smart_force_reform_drift_mult", 1.0, 20.0, 3.0)

    return out


def save_atl_settings(payload: dict[str, Any]) -> dict[str, Any]:
    out = normalize_atl_settings(payload)
    path = atl_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
