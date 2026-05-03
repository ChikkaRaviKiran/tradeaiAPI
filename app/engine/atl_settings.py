"""ATL straddle settings persistence shared by API and runtime engine."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ATL_SETTINGS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "index": "NIFTY",
    "entry_time": "09:20",
    "exit_time": "15:15",
    "lots": 1,
    "strike_interval": 50,
    "offset_points": 500,
    "rolling_points": 300,
    "first_straddle_sl_pct": 100,
    "reform_straddle_sl_pct": 60,
    "hedge_enabled": False,
    "hedge_premium": 3,
    "hedge_lots": 0,
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
    out["index"] = str(out.get("index", "NIFTY")).upper()
    if out["index"] not in {"NIFTY", "BANKNIFTY", "SENSEX"}:
        out["index"] = "NIFTY"

    out["entry_time"] = str(out.get("entry_time", "09:20"))
    out["exit_time"] = str(out.get("exit_time", "15:15"))
    out["lots"] = max(1, int(out.get("lots", 1)))
    out["strike_interval"] = max(1, int(out.get("strike_interval", 50)))
    out["offset_points"] = max(1, int(out.get("offset_points", 500)))
    out["rolling_points"] = max(1, int(out.get("rolling_points", 300)))
    out["first_straddle_sl_pct"] = max(1, int(out.get("first_straddle_sl_pct", 100)))
    out["reform_straddle_sl_pct"] = max(1, int(out.get("reform_straddle_sl_pct", 60)))
    out["hedge_enabled"] = bool(out.get("hedge_enabled", False))
    out["hedge_premium"] = max(1, int(out.get("hedge_premium", 3)))
    out["hedge_lots"] = max(0, int(out.get("hedge_lots", 0)))
    return out


def save_atl_settings(payload: dict[str, Any]) -> dict[str, Any]:
    out = normalize_atl_settings(payload)
    path = atl_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
