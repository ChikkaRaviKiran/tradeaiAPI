"""Settings persistence for the Research Multi-Index ATM Strategy.

This is the configuration backend for the four research-mode plans that
were validated in `backend/strategy_indicator_search.py`:

    mode = "single_time"        single-index, fixed-time entries
    mode = "single_indicator"   single-index, indicator-gated entries
    mode = "multi_time"         NIFTY + SENSEX same day, fixed-time
    mode = "multi_indicator"    NIFTY + SENSEX same day, indicator-gated  (recommended)

The schema is intentionally INDEPENDENT of `atl_straddle_settings.json`
so the existing battle-tested single-index live engine keeps working
unchanged. The new scanner reads ONLY `atl_research_settings.json`.

Schedule shape (per weekday × per index):
    {
      "Mon": {
        "NIFTY":  { "strat": "straddle",   "entry": "T0", "entry_time": "09:20",
                    "exit_time": "14:30",  "enabled": true },
        "SENSEX": { "strat": "strangle_2", "entry": "T1", "entry_time": "09:20",
                    "exit_time": "15:15",  "enabled": true }
      },
      ...
    }

strat:        "straddle" | "strangle_1" | "strangle_2"
              (strangle_N = legs N strike-steps OTM on each side)
entry:        "T0" .. "T5"
              T0=fixed time, T1=iv_crush, T2=vwap_align, T3=rsi_revert,
              T4=bb_squeeze, T5=ema_cross. See research_straddle_scanner._evaluate_entry().
entry_time:   anchor minute for T0 AND start-of-search for T1..T5.
exit_time:    hard EOD exit.
enabled:      include this (weekday, index) cell at all.

Defaults pre-populate the recommended schedule from the latest backtest
(`data/strategy_indicator_search.log`) so the user can flip a single
switch and go live.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

VALID_MODES = {"single_time", "single_indicator", "multi_time", "multi_indicator"}
VALID_STRATS = {"straddle", "strangle_1", "strangle_2"}
VALID_ENTRIES = {"T0", "T1", "T2", "T3", "T4", "T5"}
VALID_INDICES = {"NIFTY", "SENSEX"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ── Backtest-derived defaults (₹6k SL, no TP, 4+4 lots) ──────────────
# These ARE the winning schedules from the indicator search. See
# `data/strategy_indicator_search.log` (Jun 2026 run, +₹3.80L/week).

DEFAULT_SCHEDULE_MULTI_INDICATOR: dict[str, dict[str, dict[str, Any]]] = {
    "Mon": {
        "NIFTY":  {"strat": "straddle",   "entry": "T0", "entry_time": "09:20", "exit_time": "14:30", "enabled": True},
        "SENSEX": {"strat": "strangle_2", "entry": "T1", "entry_time": "09:20", "exit_time": "15:15", "enabled": True},
    },
    "Tue": {
        "NIFTY":  {"strat": "straddle",   "entry": "T4", "entry_time": "09:30", "exit_time": "14:30", "enabled": True},
        "SENSEX": {"strat": "strangle_2", "entry": "T4", "entry_time": "09:30", "exit_time": "15:15", "enabled": True},
    },
    "Wed": {
        "NIFTY":  {"strat": "strangle_1", "entry": "T2", "entry_time": "09:45", "exit_time": "15:15", "enabled": True},
        "SENSEX": {"strat": "straddle",   "entry": "T2", "entry_time": "09:45", "exit_time": "15:15", "enabled": True},
    },
    "Thu": {
        "NIFTY":  {"strat": "straddle",   "entry": "T3", "entry_time": "09:30", "exit_time": "15:15", "enabled": True},
        "SENSEX": {"strat": "strangle_1", "entry": "T1", "entry_time": "09:30", "exit_time": "15:15", "enabled": True},
    },
    "Fri": {
        "NIFTY":  {"strat": "straddle",   "entry": "T3", "entry_time": "11:00", "exit_time": "15:15", "enabled": True},
        "SENSEX": {"strat": "strangle_2", "entry": "T4", "entry_time": "11:00", "exit_time": "15:15", "enabled": True},
    },
}

DEFAULT_SCHEDULE_MULTI_TIME: dict[str, dict[str, dict[str, Any]]] = {
    # All cells T0/X0 straddle — fixed entry/exit times only. No indicators.
    "Mon": {
        "NIFTY":  {"strat": "straddle", "entry": "T0", "entry_time": "09:20", "exit_time": "14:30", "enabled": True},
        "SENSEX": {"strat": "straddle", "entry": "T0", "entry_time": "09:20", "exit_time": "15:15", "enabled": True},
    },
    "Tue": {
        "NIFTY":  {"strat": "straddle", "entry": "T0", "entry_time": "09:30", "exit_time": "14:30", "enabled": True},
        "SENSEX": {"strat": "straddle", "entry": "T0", "entry_time": "09:30", "exit_time": "15:15", "enabled": True},
    },
    "Wed": {
        "NIFTY":  {"strat": "straddle", "entry": "T0", "entry_time": "09:45", "exit_time": "15:15", "enabled": True},
        "SENSEX": {"strat": "straddle", "entry": "T0", "entry_time": "09:45", "exit_time": "15:15", "enabled": True},
    },
    "Thu": {
        "NIFTY":  {"strat": "straddle", "entry": "T0", "entry_time": "09:30", "exit_time": "15:15", "enabled": True},
        "SENSEX": {"strat": "straddle", "entry": "T0", "entry_time": "09:30", "exit_time": "15:15", "enabled": True},
    },
    "Fri": {
        "NIFTY":  {"strat": "straddle", "entry": "T0", "entry_time": "11:00", "exit_time": "15:15", "enabled": True},
        "SENSEX": {"strat": "straddle", "entry": "T0", "entry_time": "11:00", "exit_time": "15:15", "enabled": True},
    },
}


def _single_index_view(schedule: dict, index: str) -> dict:
    """Return a copy of `schedule` with only `index` cells kept; other side disabled."""
    out: dict = {}
    for day in WEEKDAYS:
        cells = schedule.get(day, {}) or {}
        kept_other = dict(cells.get("NIFTY" if index == "SENSEX" else "SENSEX", {}) or {})
        kept_other["enabled"] = False
        out[day] = {
            index: dict(cells.get(index, {}) or {}),
            ("NIFTY" if index == "SENSEX" else "SENSEX"): kept_other,
        }
    return out


RESEARCH_SETTINGS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mode": "multi_indicator",          # one of VALID_MODES
    "primary_index": "NIFTY",           # used when mode startswith "single_"
    "execution_account": "Primary",
    "lots_nifty": 4,
    "lots_sensex": 4,
    "sl_rs": 6000,                      # hard ₹ stop per (CE+PE) pair
    "strike_step_nifty": 50,
    "strike_step_sensex": 100,
    "schedule": DEFAULT_SCHEDULE_MULTI_INDICATOR,
}


def research_settings_path() -> str:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(backend_root, "data", "atl_research_settings.json")


def load_research_settings() -> dict[str, Any]:
    path = research_settings_path()
    if not os.path.exists(path):
        return _deep_copy_defaults()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _deep_copy_defaults()
        return normalize_research_settings(data)
    except Exception:
        logger.exception("Failed to load research settings; returning defaults")
        return _deep_copy_defaults()


def _deep_copy_defaults() -> dict[str, Any]:
    return json.loads(json.dumps(RESEARCH_SETTINGS_DEFAULTS))


def _normalize_cell(cell: Any) -> dict[str, Any] | None:
    if not isinstance(cell, dict):
        return None
    out: dict[str, Any] = {}
    out["enabled"] = bool(cell.get("enabled", False))
    strat = str(cell.get("strat", "straddle")).lower()
    out["strat"] = strat if strat in VALID_STRATS else "straddle"
    entry = str(cell.get("entry", "T0")).upper()
    out["entry"] = entry if entry in VALID_ENTRIES else "T0"
    out["entry_time"] = _clamp_time(cell.get("entry_time", "09:20"), "09:20")
    out["exit_time"] = _clamp_time(cell.get("exit_time", "15:15"), "15:15")
    return out


def _clamp_time(raw: Any, default: str) -> str:
    try:
        h, m = [int(x) for x in str(raw).split(":")]
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    except Exception:
        pass
    return default


def normalize_research_settings(payload: dict[str, Any]) -> dict[str, Any]:
    base = _deep_copy_defaults()
    out = dict(base)
    out.update(payload or {})

    out["enabled"] = bool(out.get("enabled", False))

    mode = str(out.get("mode", "multi_indicator")).lower()
    out["mode"] = mode if mode in VALID_MODES else "multi_indicator"

    pidx = str(out.get("primary_index", "NIFTY")).upper()
    out["primary_index"] = pidx if pidx in VALID_INDICES else "NIFTY"

    out["execution_account"] = str(out.get("execution_account", "Primary"))

    out["lots_nifty"] = max(1, int(out.get("lots_nifty", 4) or 1))
    out["lots_sensex"] = max(1, int(out.get("lots_sensex", 4) or 1))
    out["sl_rs"] = max(500, int(out.get("sl_rs", 6000) or 6000))
    out["strike_step_nifty"] = max(1, int(out.get("strike_step_nifty", 50) or 50))
    out["strike_step_sensex"] = max(1, int(out.get("strike_step_sensex", 100) or 100))

    sched = out.get("schedule") or base["schedule"]
    if not isinstance(sched, dict):
        sched = base["schedule"]
    norm_sched: dict[str, dict[str, dict[str, Any]]] = {}
    for day in WEEKDAYS:
        day_cells = sched.get(day) or {}
        norm_sched[day] = {}
        for idx in VALID_INDICES:
            cell = _normalize_cell(day_cells.get(idx))
            if cell is None:
                # Fall back to default for missing/invalid cells
                cell = _normalize_cell(base["schedule"][day][idx]) or {}
                # Default-derived cells are disabled unless user enabled them
                cell["enabled"] = False
            norm_sched[day][idx] = cell
    out["schedule"] = norm_sched
    return out


def save_research_settings(payload: dict[str, Any]) -> dict[str, Any]:
    out = normalize_research_settings(payload)
    path = research_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def effective_schedule(settings_dict: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return schedule with only the cells that the active mode will actually use.

    - single_time/single_indicator: only `primary_index` cells; other index forced off.
    - multi_*: full schedule honored as user set it.
    Cells with `entry`!="T0" are downgraded to T0 in the *_time modes.
    """
    mode = settings_dict.get("mode", "multi_indicator")
    sched = settings_dict.get("schedule") or {}
    pidx = settings_dict.get("primary_index", "NIFTY")

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for day in WEEKDAYS:
        out[day] = {}
        for idx in VALID_INDICES:
            cell = dict((sched.get(day, {}) or {}).get(idx, {}) or {})
            if not cell:
                cell = {"enabled": False, "strat": "straddle", "entry": "T0",
                        "entry_time": "09:20", "exit_time": "15:15"}
            # Mode-driven restrictions
            if mode in {"single_time", "single_indicator"} and idx != pidx:
                cell = dict(cell, enabled=False)
            if mode in {"single_time", "multi_time"}:
                cell = dict(cell, entry="T0")
            out[day][idx] = cell
    return out
