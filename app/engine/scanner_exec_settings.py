"""Per-scanner execution settings (lots mode + funds) for MoveDet & PDH/PDL.

Each scanner has its own JSON file under backend/data/.

Schema:
    enabled        : bool   — master scanner switch (false = scanner idle)
    live_execution : bool   — place real orders when paper_trading=False
    lots_mode      : str    — "auto" (size by available funds) or "manual"
    manual_lots    : int    — used when lots_mode == "manual"
    max_funds      : float  — capital ceiling (₹) when lots_mode == "auto"
    buffer_pct     : float  — safety margin (% of usable funds reserved)
    max_lots       : int    — hard ceiling regardless of mode
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "live_execution": False,
    "lots_mode": "auto",        # "auto" | "manual"
    "manual_lots": 1,
    "max_funds": 150000.0,
    "buffer_pct": 5.0,
    "max_lots": 20,
}

_FILES = {
    "move_det": "move_det_exec_settings.json",
    "pdh_pdl": "pdh_pdl_exec_settings.json",
}


def _path(scanner: str) -> str:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fname = _FILES.get(scanner)
    if not fname:
        raise ValueError(f"Unknown scanner: {scanner}")
    return os.path.join(backend_root, "data", fname)


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update(payload or {})

    out["enabled"] = bool(out.get("enabled", True))
    out["live_execution"] = bool(out.get("live_execution", False))

    mode = str(out.get("lots_mode", "auto")).strip().lower()
    if mode not in {"auto", "manual"}:
        mode = "auto"
    out["lots_mode"] = mode

    try:
        out["manual_lots"] = max(1, int(out.get("manual_lots", 1)))
    except (TypeError, ValueError):
        out["manual_lots"] = 1
    try:
        out["max_funds"] = max(0.0, float(out.get("max_funds", 150000.0)))
    except (TypeError, ValueError):
        out["max_funds"] = 150000.0
    try:
        out["buffer_pct"] = max(0.0, float(out.get("buffer_pct", 5.0)))
    except (TypeError, ValueError):
        out["buffer_pct"] = 5.0
    try:
        out["max_lots"] = max(1, int(out.get("max_lots", 20)))
    except (TypeError, ValueError):
        out["max_lots"] = 20

    return out


def load(scanner: str) -> dict[str, Any]:
    path = _path(scanner)
    if not os.path.exists(path):
        return dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULTS)
        return normalize(data)
    except Exception:
        logger.exception("Failed to load exec settings for %s; using defaults", scanner)
        return dict(DEFAULTS)


def save(scanner: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = normalize(payload)
    path = _path(scanner)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
