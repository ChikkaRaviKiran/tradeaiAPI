"""Global priority-handoff setting.

When enabled (default), `_pre_signal_handoff` force-closes any open
ATM/MoveDet/PDH-PDL position before a new entry from the three
priority scanners (MoveDet, MoveDet-Bull, PDH/PDL) is placed.

When disabled, the new signal is BLOCKED (entry skipped) whenever
another priority-scanner trade or the ATM straddle is already live —
i.e. the user opts out of position rollover entirely.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
}


def _path() -> str:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(backend_root, "data", "priority_handoff_settings.json")


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update(payload or {})
    out["enabled"] = bool(out.get("enabled", True))
    return out


def load() -> dict[str, Any]:
    path = _path()
    if not os.path.exists(path):
        return dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULTS)
        return normalize(data)
    except Exception:
        logger.exception("Failed to load priority_handoff_settings; using defaults")
        return dict(DEFAULTS)


def save(payload: dict[str, Any]) -> dict[str, Any]:
    out = normalize(payload)
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
