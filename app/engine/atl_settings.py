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
    out["strike_mode"] = "ITM" if mode == "ITM" else "ATM"
    out["lots"] = max(1, int(out.get("lots", 1)))
    out["strike_interval"] = max(1, int(out.get("strike_interval", 50)))
    out["offset_points"] = max(1, int(out.get("offset_points", 500)))
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

    return out


def save_atl_settings(payload: dict[str, Any]) -> dict[str, Any]:
    out = normalize_atl_settings(payload)
    path = atl_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
