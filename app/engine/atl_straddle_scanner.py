"""ATL Straddle scanner.

Implements OptionSelling-style at-strike logic in paper mode:
- Enter short strangle at configured entry time.
- At-strike touch converts to straddle.
- Straddle SL in points = (CE premium + PE premium) * configured %.
- On SL breach reform straddle at new ATM with reform SL %.
- Force close at configured exit time.

This scanner is alert-first and does not place broker orders directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Optional

import pandas as pd
import pytz

from app.alerts.alert_manager import AlertManager
from app.core.config import settings
from app.core.instruments import InstrumentConfig
from app.data.angelone_client import AngelOneClient
from app.engine.atl_settings import load_atl_settings
from app.execution.broker_base import (
    BaseBroker,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")
ORDER_POLL_RETRIES = 8
ORDER_POLL_DELAY_SECONDS = 1.0


def _atl_state_path() -> str:
    """On-disk location for persisted ATL runtime state + events.

    Lives next to atl_straddle_settings.json so restarts within the
    same trading day can recover ref_spot, current legs, hedge legs,
    roll/reform counters, and the event timeline. Without this the
    scanner forgets where it entered the moment the backend restarts
    and cannot compute roll/reform triggers correctly.
    """
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(backend_root, "data", "atl_straddle_state.json")


@dataclass
class ATLLeg:
    option_type: str
    strike: float
    symbol: str = ""
    symboltoken: str = ""
    exchange: str = "NFO"
    premium: float = 0.0


@dataclass
class ATLState:
    phase: str = "IDLE"  # IDLE | STRANGLE | STRADDLE
    ref_spot: float = 0.0
    straddle_strike: float = 0.0
    straddle_sl_points: float = 0.0
    is_first_straddle: bool = True
    roll_count: int = 0
    reform_count: int = 0
    ce: Optional[ATLLeg] = None
    pe: Optional[ATLLeg] = None
    hedge_ce: Optional[ATLLeg] = None
    hedge_pe: Optional[ATLLeg] = None
    entered: bool = False
    entry_in_progress: bool = False
    done_for_day: bool = False
    halted: bool = False           # circuit-breaker tripped
    halt_reason: str = ""          # human-readable reason for the halt


class ATLStraddleScanner:
    def __init__(
        self,
        client: AngelOneClient,
        alert_manager: AlertManager,
        broker: Optional[BaseBroker] = None,
        expiry_provider: Optional["Callable[[str], tuple[str, Optional[date]]]"] = None,
    ):
        self.client = client
        self.alert_manager = alert_manager
        self.broker = broker
        # Optional callback (index_symbol) -> (expiry_str, expiry_date). When
        # provided, the scanner consults it every cycle so changing the index
        # in the UI mid-session immediately picks up the new index's expiry
        # instead of being stuck with whatever set_expiry() pushed at startup.
        self._expiry_provider = expiry_provider
        self._settings = load_atl_settings()
        self._today_str = ""
        self._expiry = ""
        self._expiry_date: Optional[date] = None
        self._state = ATLState()
        self._events: list[dict] = []
        self._diag_last: dict[str, datetime] = {}
        self._force_entry: bool = False
        self._last_done_signature: str = ""
        # Consecutive scanner ticks where the broker reported BOTH SELL legs
        # missing. We only declare a manual exit after >=2 consecutive misses
        # to avoid false positives from Dhan's positions API being briefly
        # stale right after a fresh entry. Transient — not persisted.
        self._manual_exit_misses: int = 0
        # Wall-clock timestamp of the most recent SELL-leg placement (initial
        # entry OR reform/switch). Used to suppress the manual-exit detector
        # for a grace window so a slow broker positions endpoint can't
        # falsely trip it during/after a strike rollover. Transient.
        self._last_legs_changed_at: Optional[datetime] = None
        # Recover prior intraday state from disk if it's from today.
        self._load_state_from_disk()

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._settings = load_atl_settings()
        self._state = ATLState()
        self._events = []
        self._diag_last = {}
        self._force_entry = False
        self._last_done_signature = ""
        self._manual_exit_misses = 0
        self._last_legs_changed_at = None
        self._save_state_to_disk()

    # ── State persistence ────────────────────────────────────────────
    def _leg_to_dict(self, leg: Optional[ATLLeg]) -> Optional[dict]:
        if leg is None:
            return None
        return {
            "option_type": leg.option_type,
            "strike": leg.strike,
            "symbol": leg.symbol,
            "symboltoken": leg.symboltoken,
            "exchange": leg.exchange,
            "premium": leg.premium,
        }

    def _dict_to_leg(self, d: Optional[dict]) -> Optional[ATLLeg]:
        if not isinstance(d, dict):
            return None
        try:
            return ATLLeg(
                option_type=str(d.get("option_type", "")),
                strike=float(d.get("strike", 0) or 0),
                symbol=str(d.get("symbol", "")),
                symboltoken=str(d.get("symboltoken", "")),
                exchange=str(d.get("exchange", "NFO")),
                premium=float(d.get("premium", 0) or 0),
            )
        except Exception:
            return None

    def _save_state_to_disk(self) -> None:
        path = _atl_state_path()
        try:
            payload = {
                "today": self._today_str,
                "state": {
                    "phase": self._state.phase,
                    "ref_spot": self._state.ref_spot,
                    "straddle_strike": self._state.straddle_strike,
                    "straddle_sl_points": self._state.straddle_sl_points,
                    "is_first_straddle": self._state.is_first_straddle,
                    "roll_count": self._state.roll_count,
                    "reform_count": self._state.reform_count,
                    "ce": self._leg_to_dict(self._state.ce),
                    "pe": self._leg_to_dict(self._state.pe),
                    "hedge_ce": self._leg_to_dict(self._state.hedge_ce),
                    "hedge_pe": self._leg_to_dict(self._state.hedge_pe),
                    "entered": self._state.entered,
                    "entry_in_progress": False,  # never resume mid-placement
                    "done_for_day": self._state.done_for_day,
                    "halted": self._state.halted,
                    "halt_reason": self._state.halt_reason,
                },
                "events": self._events[-200:],
                "last_done_signature": self._last_done_signature,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            os.replace(tmp, path)
        except Exception:
            logger.exception("[ATL] Failed to persist state to %s", path)

    def _load_state_from_disk(self) -> None:
        path = _atl_state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("[ATL] Failed to load persisted state from %s", path)
            return
        today = datetime.now(IST).strftime("%Y-%m-%d")
        saved_today = str(payload.get("today") or "")
        if saved_today != today:
            # Stale (different trading day) — ignore, reset_daily will
            # clear the file on next cycle.
            logger.info("[ATL] Persisted state is from %s (today=%s); ignoring", saved_today, today)
            return
        st = payload.get("state") or {}
        try:
            self._today_str = saved_today
            self._state = ATLState(
                phase=str(st.get("phase", "IDLE")),
                ref_spot=float(st.get("ref_spot", 0) or 0),
                straddle_strike=float(st.get("straddle_strike", 0) or 0),
                straddle_sl_points=float(st.get("straddle_sl_points", 0) or 0),
                is_first_straddle=bool(st.get("is_first_straddle", True)),
                roll_count=int(st.get("roll_count", 0) or 0),
                reform_count=int(st.get("reform_count", 0) or 0),
                ce=self._dict_to_leg(st.get("ce")),
                pe=self._dict_to_leg(st.get("pe")),
                hedge_ce=self._dict_to_leg(st.get("hedge_ce")),
                hedge_pe=self._dict_to_leg(st.get("hedge_pe")),
                entered=bool(st.get("entered", False)),
                entry_in_progress=False,
                done_for_day=bool(st.get("done_for_day", False)),
                halted=bool(st.get("halted", False)),
                halt_reason=str(st.get("halt_reason", "")),
            )
            self._events = list(payload.get("events") or [])
            self._last_done_signature = str(payload.get("last_done_signature") or "")
            logger.info(
                "[ATL] Recovered state for %s: phase=%s entered=%s ref_spot=%.2f straddle_strike=%.0f events=%d",
                today, self._state.phase, self._state.entered,
                self._state.ref_spot, self._state.straddle_strike, len(self._events),
            )
        except Exception:
            logger.exception("[ATL] Failed to apply persisted state; starting fresh")
            self._state = ATLState()
            self._events = []

    def _settings_signature(self) -> str:
        """Minimal signature used to detect user intent to re-arm intraday."""
        return "|".join(
            [
                str(self._settings.get("enabled", False)),
                str(self._settings.get("index", "NIFTY")),
                str(self._settings.get("entry_time", "09:20")),
                str(self._settings.get("exit_time", "15:15")),
                str(self._settings.get("execution_account", "Primary")),
            ]
        )

    def _complete_for_day(self, reason: str) -> None:
        self._state.done_for_day = True
        self._state.entry_in_progress = False
        self._state.halted = False
        self._state.halt_reason = ""
        self._state.phase = "IDLE"
        self._state.entered = False
        self._last_done_signature = self._settings_signature()
        self._record_event("complete", reason)

    def _is_live(self) -> bool:
        """True only when global mode is LIVE AND the strategy's execution
        account is set to a real broker (anything other than 'Paper').

        Per-strategy 'Paper' account simulates even when global is LIVE.
        Global 'paper_trading=True' acts as a kill-switch — every strategy
        runs in simulation regardless of its execution_account.
        """
        if bool(getattr(settings, "paper_trading", True)):
            return False
        account = str(self._settings.get("execution_account", "Primary")).strip().lower()
        return account not in ("", "paper")

    def _mode_label(self) -> str:
        return "LIVE" if self._is_live() else "PAPER"

    def _resolve_broker(self):
        """Pick the broker for this cycle based on the strategy's
        execution_account setting. Allows per-strategy override of the
        global TRADING_ACCOUNT.

        Values:
          'Paper' / ''        → simulated, no broker call
          'Primary'           → use the orchestrator-injected default broker
          'Kite' / 'Live (Kite)' / case-insensitive variants → Kite
          'Dhan' / 'Live (Dhan)'                              → Dhan
          'Angel' / 'AngelOne' / 'Live (Angel)'              → AngelOne
        Falls back to self.broker on any failure.
        """
        raw = str(self._settings.get("execution_account", "Primary")).strip().lower()
        if raw in ("", "paper"):
            return None
        if raw in ("primary", "live (primary)"):
            return self.broker
        # Extract the broker token from labels like "live (kite)"
        token = raw
        if "(" in raw and ")" in raw:
            token = raw[raw.find("(") + 1: raw.find(")")].strip()
        try:
            if token in ("kite", "zerodha"):
                from app.execution.kite_broker import KiteBroker
                if not isinstance(self.broker, KiteBroker):
                    return KiteBroker()
                return self.broker
            if token == "dhan":
                from app.execution.dhan_broker import DhanBroker
                if not isinstance(self.broker, DhanBroker):
                    return DhanBroker()
                return self.broker
            if token in ("angel", "angelone", "smartapi"):
                from app.execution.angelone_broker import AngelOneBroker
                if not isinstance(self.broker, AngelOneBroker):
                    return AngelOneBroker()
                return self.broker
        except Exception:
            logger.exception("ATL: failed to instantiate broker for account=%s", raw)
        return self.broker

    def get_runtime_state(self) -> dict:
        st = self._state
        return {
            "enabled": bool(self._settings.get("enabled", False)),
            "live_mode": self._is_live(),
            "global_live": not bool(getattr(settings, "paper_trading", True)),
            "execution_account": self._settings.get("execution_account", "Primary"),
            "strategy_type": self._settings.get("strategy_type", "ATM_STRADDLE"),
            "phase": st.phase,
            "in_trade": self.is_in_trade(),
            "done_for_day": st.done_for_day,
            "halted": st.halted,
            "halt_reason": st.halt_reason,
            "index": self._settings.get("index", "NIFTY"),
            "trading_day": self._settings.get("trading_day", "Daily"),
            "expiry": self._expiry,
            "entry_time": self._settings.get("entry_time", "09:20"),
            "exit_time": self._settings.get("exit_time", "15:15"),
            "strike_interval": self._settings.get("strike_interval", 50),
            "offset_points": self._settings.get("offset_points", 500),
            "rolling_points": self._settings.get("rolling_points", 300),
            "sl_type": self._settings.get("sl_type", "premium_pct"),
            "sl_lower": self._settings.get("sl_lower", 0),
            "sl_upper": self._settings.get("sl_upper", 0),
            "hedge_mode": self._settings.get("hedge_mode", "none"),
            "hedge_otm_points": self._settings.get("hedge_otm_points", 500),
            "ref_spot": st.ref_spot,
            "straddle_strike": st.straddle_strike,
            "straddle_sl_points": st.straddle_sl_points,
            "roll_count": st.roll_count,
            "reform_count": st.reform_count,
            "short_ce": self._leg_dict(st.ce),
            "short_pe": self._leg_dict(st.pe),
            "hedge_ce": self._leg_dict(st.hedge_ce),
            "hedge_pe": self._leg_dict(st.hedge_pe),
            "events": self._events[-40:],
        }

    def set_expiry(self, expiry: str, expiry_date: Optional[date]) -> None:
        self._expiry = expiry
        self._expiry_date = expiry_date

    def _refresh_expiry_for_configured_index(self) -> None:
        """Re-pull expiry from the orchestrator for the currently configured
        index. Called every cycle so a UI index switch (NIFTY ↔ SENSEX)
        takes effect immediately instead of being stuck on the index that
        was configured at process start.
        """
        if self._expiry_provider is None:
            return
        try:
            idx = str(self._settings.get("index", "NIFTY")).upper()
            new_exp, new_exp_date = self._expiry_provider(idx)
        except Exception:
            logger.exception("[ATL] expiry_provider call failed")
            return
        if new_exp and new_exp != self._expiry:
            logger.info(
                "[ATL] Expiry refreshed for %s: %s → %s",
                idx, self._expiry or "-", new_exp,
            )
            self._record_event(
                "expiry_refresh",
                f"Expiry updated for {idx}: {self._expiry or '-'} → {new_exp}",
            )
            self._expiry = new_exp
            self._expiry_date = new_exp_date

    def is_in_trade(self) -> bool:
        return self._state.phase in {"STRANGLE", "STRADDLE"} and not self._state.done_for_day

    async def run_cycle(self, df_today: pd.DataFrame, instrument: InstrumentConfig, cycle: int, peer_in_trade: bool = False) -> None:
        if df_today is None or df_today.empty:
            self._record_diag("no_data", f"No 1-min candles for {instrument.symbol}; data feed empty")
            return

        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")
        if self._today_str != today:
            self.reset_daily()

        # Refresh settings periodically so UI changes apply intraday.
        refresh_every = max(1, int(getattr(settings, "atl_settings_refresh_cycles", 1) or 1))
        if cycle % refresh_every == 0:
            prev_signature = self._settings_signature()
            self._settings = load_atl_settings()
            # Re-pull expiry for the (possibly newly chosen) index. Without
            # this the scanner would keep using whatever expiry was injected
            # at startup, so switching NIFTY ↔ SENSEX in the UI silently
            # breaks option-quote lookups for the new index.
            self._refresh_expiry_for_configured_index()
            new_signature = self._settings_signature()
            signature_changed = new_signature != prev_signature
            # Determine if newly-configured entry_time is still in the future
            # relative to now — that's a clear "user wants a fresh entry" signal.
            entry_in_future = False
            try:
                eh, em = [int(x) for x in str(self._settings.get("entry_time", "09:20")).split(":")]
                entry_dt = datetime(now.year, now.month, now.day, eh, em, tzinfo=IST)
                entry_in_future = now < entry_dt
            except Exception:
                entry_in_future = False

            # Re-arm intraday in two cases:
            #   1. Strategy already completed for the day AND settings changed.
            #   2. User pushed entry_time to a future moment (signature changed
            #      and entry_time is now in the future). This covers the case
            #      where the user manually exited at the broker and wants the
            #      scanner to place fresh orders at the new entry_time —
            #      previously this was ignored because `done_for_day` was False
            #      and the scanner thought it was still holding legs.
            done_rearm = (
                self._state.done_for_day
                and new_signature != self._last_done_signature
            )
            # SAFETY: previously `future_rearm` would silently wipe engine
            # state (legs, phase, entered flag) whenever the user moved the
            # entry_time to a future moment AND we were already in a trade.
            # That wipe did NOT close broker positions, so the broker held
            # orphan legs while the engine immediately placed a fresh
            # straddle on the next tick — visible to the user as an
            # "exit + restraddle" right after entry. Refuse to wipe state
            # while we are actually in a position; require an explicit
            # Force Close from the UI first.
            future_rearm = (
                signature_changed
                and entry_in_future
                and self._state.entered
                and not self.is_in_trade()
            )
            future_rearm_blocked = (
                signature_changed
                and entry_in_future
                and self._state.entered
                and self.is_in_trade()
            )
            if future_rearm_blocked:
                logger.warning(
                    "[ATL] Refusing auto-rearm: in-trade and entry_time moved "
                    "to future. Use 'Force Close ATM' first. sig_old=%s sig_new=%s",
                    prev_signature, new_signature,
                )
                self._record_event(
                    "rearm_blocked",
                    "Settings changed mid-trade — refused to wipe legs. "
                    "Click 'Force Close ATM' to exit broker positions first.",
                )
            if done_rearm or future_rearm:
                self._state.done_for_day = False
                self._state.halted = False
                self._state.halt_reason = ""
                self._state.phase = "IDLE"
                self._state.entered = False
                self._state.entry_in_progress = False
                self._state.ce = None
                self._state.pe = None
                self._state.hedge_ce = None
                self._state.hedge_pe = None
                self._state.straddle_strike = 0.0
                self._state.straddle_sl_points = 0.0
                self._state.is_first_straddle = True
                self._state.roll_count = 0
                self._state.reform_count = 0
                self._diag_last.clear()
                reason = (
                    f"Settings changed and entry_time moved to future — "
                    f"scanner state cleared for fresh entry "
                    f"(sig: {prev_signature} -> {new_signature})"
                    if future_rearm
                    else (
                        f"Settings changed by user — strategy re-armed "
                        f"(sig: {prev_signature} -> {new_signature})"
                    )
                )
                logger.info("[ATL] Re-armed: %s", reason)
                self._record_event("rearm", reason)

        if not self._settings.get("enabled", False):
            self._record_diag(
                "disabled",
                "Strategy disabled (enabled=false in atl_straddle_settings.json). "
                "Click 'Create' or 'Create & Place Now' on Settings → Strategy.",
            )
            return

        configured_index = self._settings.get("index", "NIFTY")
        if instrument.symbol != configured_index:
            # This fires once per non-matching instrument per minute — fine.
            self._record_diag(
                f"wrong_index_{instrument.symbol}",
                f"Skipping {instrument.symbol}; strategy is configured for {configured_index}",
            )
            return

        trading_day = str(self._settings.get("trading_day", "Daily")).title()
        if trading_day != "Daily":
            weekdays = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4,
            }
            target = weekdays.get(trading_day)
            if target is not None and now.weekday() != target:
                self._record_diag(
                    "wrong_day",
                    f"Today is {now.strftime('%A')}; strategy runs only on {trading_day}",
                )
                return

        try:
            entry_h, entry_m = [int(x) for x in str(self._settings.get("entry_time", "09:20")).split(":")]
            exit_h, exit_m = [int(x) for x in str(self._settings.get("exit_time", "15:15")).split(":")]
        except Exception:
            entry_h, entry_m = 9, 20
            exit_h, exit_m = 15, 15

        session_exit_time = datetime(now.year, now.month, now.day, exit_h, exit_m, tzinfo=IST).time()
        if now.time() >= session_exit_time:
            if self.is_in_trade():
                await self.force_close(df_today, instrument, reason="session_exit_time")
                return
            # No new entries after configured exit time.
            if not self._state.done_for_day:
                self._complete_for_day(
                    f"Entry window ended at {exit_h:02d}:{exit_m:02d}; no new entries for today"
                )
            return

        if self._state.done_for_day:
            self._record_diag("done_for_day", "Strategy already completed for the day")
            return

        # ── Detect manual exit at the broker ────────────────────────────
        # If the engine thinks it's in a straddle but the broker shows
        # both SELL legs are gone (qty=0), the user must have closed the
        # position manually (broker terminal, mobile app, or our own
        # positions page that bypasses the scanner). Stop the scanner for
        # the rest of the day so it doesn't auto-place a new straddle on
        # the next tick. The user can move `entry_time` to a future moment
        # (the existing future_rearm path) to schedule a fresh entry.
        if self._state.entered and self._state.phase == "STRADDLE" and self._state.ce and self._state.pe:
            # Grace window: suppress the manual-exit check for N seconds
            # after the most recent leg placement (initial entry OR strike
            # reform). Dhan's positions endpoint is routinely stale for
            # 15-45s after order fills — during a reform the OLD symbols are
            # already gone and the NEW symbols haven't appeared yet, so
            # without this guard a rollover trips manual-exit every time.
            grace_seconds = max(
                0,
                int(self._settings.get("manual_exit_grace_seconds", 90) or 0),
            )
            in_grace = (
                self._last_legs_changed_at is not None
                and (now - self._last_legs_changed_at).total_seconds() < grace_seconds
            )
            try:
                broker_legs = await self._broker_open_strike_legs() if not in_grace else None
            except Exception:
                broker_legs = None
            if in_grace:
                # Keep the miss counter clean across the grace window so the
                # first tick after grace expires starts from zero.
                self._manual_exit_misses = 0
                self._record_diag(
                    "manual_exit_grace",
                    f"Skipping manual-exit check; within {grace_seconds}s grace after last leg change",
                )
            if broker_legs is not None:
                ce_key = (int(round(self._state.ce.strike)), "CE")
                pe_key = (int(round(self._state.pe.strike)), "PE")
                ce_open = ce_key in broker_legs
                pe_open = pe_key in broker_legs
                if not ce_open and not pe_open:
                    # Require two consecutive misses before declaring a
                    # manual exit. Dhan's positions API is occasionally
                    # stale for a few seconds after a fresh entry, which
                    # would otherwise instantly shut the scanner down.
                    self._manual_exit_misses += 1
                    if self._manual_exit_misses < 2:
                        logger.info(
                            "[ATL] Broker shows no open SELL legs (CE %d / PE %d) — "
                            "miss %d/2; waiting one more tick. Broker legs seen=%s",
                            ce_key[0], pe_key[0],
                            self._manual_exit_misses,
                            sorted(broker_legs),
                        )
                        return
                    logger.warning(
                        "[ATL] Manual exit detected: broker shows no open SELL legs "
                        "(CE %d / PE %d) for %d consecutive ticks. Broker legs seen=%s. "
                        "Stopping scanner for today.",
                        ce_key[0], pe_key[0],
                        self._manual_exit_misses,
                        sorted(broker_legs),
                    )
                    self._record_event(
                        "manual_exit_detected",
                        f"Broker shows no open SELL legs for {int(self._state.straddle_strike)} "
                        f"straddle (confirmed {self._manual_exit_misses} ticks). Scanner stopped "
                        f"— change entry_time to a future moment to schedule a fresh entry.",
                    )
                    try:
                        await self.alert_manager.telegram.send(
                            f"🛑 ATL MANUAL EXIT DETECTED ({self._mode_label()})\n"
                            f"Strike: {int(self._state.straddle_strike)}\n"
                            f"Both SELL legs were closed externally.\n"
                            f"Scanner is now stopped for today.\n"
                            f"To re-enter: set a new entry_time in Settings."
                        )
                    except Exception:
                        logger.exception("[ATL] Failed to send manual-exit Telegram alert")
                    # Wipe leg state so re-arm with a future entry_time works
                    # cleanly via the existing future_rearm flow.
                    self._state.phase = "IDLE"
                    self._state.entered = False
                    self._state.ce = None
                    self._state.pe = None
                    self._state.straddle_strike = 0.0
                    self._state.ref_spot = 0.0
                    # NOTE: hedges are intentionally left in state — user
                    # may have closed only the SELL legs and kept hedges.
                    # If they want everything flat, they can use Force Close.
                    self._complete_for_day("manual_exit_at_broker")
                    return
                else:
                    # At least one leg still visible at broker → not a manual
                    # exit. Clear any prior miss so a later transient single
                    # miss doesn't accumulate across unrelated ticks.
                    self._manual_exit_misses = 0

        if peer_in_trade and not self.is_in_trade():
            self._record_diag(
                "peer_in_trade",
                "Peer strategy holds a position; waiting for it to close (mutex)",
            )
            return

        spot = float(df_today.iloc[-1]["close"])
        interval = max(1, int(self._settings.get("strike_interval", 50)))
        offset = max(0, int(self._settings.get("offset_points", 500)))
        rolling = max(1, int(self._settings.get("rolling_points", 300)))
        # ATM-straddle adjustment trigger: when |spot - entry_spot| >= this,
        # close existing SELL legs and re-short at the new ATM strike. Hedges
        # are intentionally left intact across reforms. Falls back to the
        # legacy `rolling_points` key so existing settings keep working.
        adjustment_points = max(
            1,
            int(self._settings.get("adjustment_points", self._settings.get("rolling_points", 100))),
        )
        # SL logic removed per ATM-straddle spec — strategy exits only at
        # exit_time or via emergency force_close. sl_type kept for API
        # back-compat but no longer evaluated.
        sl_type = "none"

        # Initial entry
        if not self._state.entered:
            if self._state.entry_in_progress:
                self._record_diag("entry_in_progress", "Entry already in progress; skipping duplicate cycle")
                return
            if not self._force_entry and (
                now.hour < entry_h or (now.hour == entry_h and now.minute < entry_m)
            ):
                # Precise wake-up: if entry_time is within the next loop tick,
                # sleep exactly until then so the order fires within ~1s of
                # the configured time instead of waiting for the next 10s tick.
                entry_dt_local = datetime(now.year, now.month, now.day, entry_h, entry_m, tzinfo=IST)
                wait_s = (entry_dt_local - now).total_seconds()
                if 0 < wait_s <= 12.0:
                    self._record_diag(
                        "precise_wait",
                        f"Sleeping {wait_s:.1f}s for precise entry at {entry_h:02d}:{entry_m:02d}",
                    )
                    await asyncio.sleep(wait_s)
                    now = datetime.now(IST)
                else:
                    self._record_diag(
                        "before_entry_time",
                        f"Waiting for entry time {entry_h:02d}:{entry_m:02d} (now {now.strftime('%H:%M:%S')})",
                    )
                    return
            if self._force_entry:
                self._record_event(
                    "force_entry",
                    f"Manual 'Place Now' bypassing entry-time gate at {now.strftime('%H:%M:%S')}",
                )
                self._force_entry = False
            self._state.entry_in_progress = True
            rounded = round(spot / interval) * interval
            ce_strike = rounded + offset
            pe_strike = rounded - offset
            # Parallelize CE+PE quote fetches (independent broker calls).
            ce_q, pe_q = await asyncio.gather(
                self._fetch_option_quote(instrument, ce_strike, "CE"),
                self._fetch_option_quote(instrument, pe_strike, "PE"),
            )
            if not ce_q or not pe_q:
                self._state.entry_in_progress = False
                self._record_diag(
                    "no_option_quote",
                    f"Could not fetch option quotes (CE={int(ce_strike)} {'OK' if ce_q else 'MISS'}, "
                    f"PE={int(pe_strike)} {'OK' if pe_q else 'MISS'}); will retry",
                )
                return
            # Parallelize leg builds (each may resolve symbol via broker).
            ce_leg, pe_leg = await asyncio.gather(
                self._build_leg(instrument, ce_strike, "CE", fallback_quote=ce_q),
                self._build_leg(instrument, pe_strike, "PE", fallback_quote=pe_q),
            )
            if not ce_leg or not pe_leg:
                self._state.entry_in_progress = False
                self._record_diag(
                    "leg_build_fail",
                    f"Failed to build option legs (CE={'OK' if ce_leg else 'FAIL'}, "
                    f"PE={'OK' if pe_leg else 'FAIL'}); check broker symbol resolver",
                )
                return

            # Margin-friendly sequence: BUY hedges FIRST so the broker
            # treats the SELL legs as a covered position. Selling naked first
            # spikes margin requirement and risks rejection.
            sel_broker = self._resolve_broker()
            self._record_event(
                "broker",
                f"Routing via {(sel_broker.name if sel_broker else 'PAPER')} "
                f"(execution_account={self._settings.get('execution_account', 'Primary')})",
            )
            hedges_placed = False
            hedges_required = False
            if self._settings.get("hedge_enabled", False):
                hedges_required = True
                await self._ensure_hedges(instrument, rounded)
                hedges_placed = bool(self._state.hedge_ce and self._state.hedge_pe)
                if not hedges_placed:
                    # Hedge failure: flatten whatever was placed and finish for the day.
                    await self._close_hedges(instrument, "entry_fail_hedge", force=True)
                    self._state.hedge_ce = None
                    self._state.hedge_pe = None
                    self._state.entry_in_progress = False
                    self._complete_for_day("Hedge leg placement failed; strategy completed for today")
                    return

            if not await self._execute_entry_legs(instrument, ce_leg, pe_leg, reason="initial_entry"):
                # _execute_entry_legs already logs order_error events with the
                # broker rejection message — no extra diag needed here.
                # Roll back the hedges we just placed so we don't sit on
                # naked long premium when the shorts couldn't be opened.
                if hedges_placed:
                    await self._close_hedges(instrument, "rollback_initial_entry", force=True)
                    self._state.hedge_ce = None
                    self._state.hedge_pe = None
                self._state.entry_in_progress = False
                self._complete_for_day("Entry leg placement failed; strategy completed for today")
                return

            self._state.ce = ce_leg
            self._state.pe = pe_leg
            self._state.entered = True
            self._manual_exit_misses = 0
            self._last_legs_changed_at = datetime.now(IST)
            if ce_strike == pe_strike:
                # Entry is already ATM straddle (offset=0). Avoid immediate
                # close/reopen of identical strikes on the next touch check.
                self._state.phase = "STRADDLE"
                self._state.straddle_strike = ce_strike
                # SL logic removed per ATM-straddle spec.
                self._state.straddle_sl_points = 0.0
            else:
                self._state.phase = "STRANGLE"
            self._state.ref_spot = spot
            self._record_event("entry", f"STRANGLE CE {int(ce_strike)} PE {int(pe_strike)} @ spot {spot:.2f}")
            await self.alert_manager.telegram.send(
                f"⚡ ATL ENTRY ({self._mode_label()})\n"
                f"Index: {instrument.symbol}\n"
                f"Spot: {spot:.2f}\n"
                f"SELL CE {int(ce_strike)} @ ₹{self._state.ce.premium:.2f}\n"
                f"SELL PE {int(pe_strike)} @ ₹{self._state.pe.premium:.2f}"
            )
            self._state.entry_in_progress = False
            return

        # Update current premiums
        if self._state.ce:
            ce_q = await self._fetch_option_quote(instrument, self._state.ce.strike, "CE")
            if ce_q:
                self._state.ce.premium = float(ce_q.get("ltp", 0) or self._state.ce.premium)
        if self._state.pe:
            pe_q = await self._fetch_option_quote(instrument, self._state.pe.strike, "PE")
            if pe_q:
                self._state.pe.premium = float(pe_q.get("ltp", 0) or self._state.pe.premium)

        if not self._state.ce or not self._state.pe:
            return

        # STRANGLE phase logic
        if self._state.phase == "STRANGLE":
            # At-strike conversion
            if spot >= self._state.ce.strike:
                await self._convert_to_straddle(instrument, self._state.ce.strike, spot)
                return
            if spot <= self._state.pe.strike:
                await self._convert_to_straddle(instrument, self._state.pe.strike, spot)
                return

            move_up = spot - self._state.ref_spot
            move_down = self._state.ref_spot - spot
            is_expiry = bool(self._expiry_date and now.date() == self._expiry_date)

            if move_up >= rolling:
                if is_expiry:
                    new_pe = round((spot - rolling) / interval) * interval
                    if not await self._roll_strangle(instrument, self._state.ce.strike, new_pe, reason="roll_up_expiry"):
                        return
                    self._state.ref_spot = spot
                    await self.alert_manager.telegram.send(f"🔁 ATL Roll Up (expiry): PE -> {int(self._state.pe.strike)}")
                elif self._state.roll_count < 1:
                    new_ce = round((self._state.ce.strike + rolling) / interval) * interval
                    new_pe = round((self._state.pe.strike + offset) / interval) * interval
                    if not await self._roll_strangle(instrument, new_ce, new_pe, reason="roll_up"):
                        return
                    self._state.roll_count += 1
                    self._state.ref_spot = spot
                    await self.alert_manager.telegram.send(
                        f"🔁 ATL Roll Up: CE -> {int(self._state.ce.strike)}, PE -> {int(self._state.pe.strike)}"
                    )
                else:
                    await self._convert_to_straddle(instrument, self._state.ce.strike, spot)
                return

            if move_down >= rolling:
                if is_expiry:
                    new_ce = round((spot + rolling) / interval) * interval
                    if not await self._roll_strangle(instrument, new_ce, self._state.pe.strike, reason="roll_down_expiry"):
                        return
                    self._state.ref_spot = spot
                    await self.alert_manager.telegram.send(f"🔁 ATL Roll Down (expiry): CE -> {int(self._state.ce.strike)}")
                elif self._state.roll_count < 1:
                    new_ce = round((self._state.ce.strike - offset) / interval) * interval
                    new_pe = round((self._state.pe.strike - rolling) / interval) * interval
                    if not await self._roll_strangle(instrument, new_ce, new_pe, reason="roll_down"):
                        return
                    self._state.roll_count += 1
                    self._state.ref_spot = spot
                    await self.alert_manager.telegram.send(
                        f"🔁 ATL Roll Down: CE -> {int(self._state.ce.strike)}, PE -> {int(self._state.pe.strike)}"
                    )
                else:
                    await self._convert_to_straddle(instrument, self._state.pe.strike, spot)
                return

        # STRADDLE phase logic — pure spot-distance adjustment.
        # When the live spot has moved >= adjustment_points away from the
        # spot at which the current straddle was opened (NOT the rounded
        # strike), close only the SELL legs and re-short at the new ATM.
        # Hedges are left in place untouched. There is no premium-based
        # SL — the strategy holds till exit_time unless force-closed.
        if (
            self._state.phase == "STRADDLE"
            and self._state.entered
            and self._state.ref_spot > 0
        ):
            drift = abs(spot - self._state.ref_spot)
            if drift >= adjustment_points:
                new_strike = round(spot / interval) * interval
                if new_strike == self._state.straddle_strike:
                    # Spot drifted enough but rounds back to the same
                    # strike (e.g. interval=100, drift=110, strike doesn't
                    # change). Just refresh the reference so we don't keep
                    # re-checking; no order churn needed.
                    self._state.ref_spot = spot
                    self._save_state_to_disk()
                    return
                self._state.reform_count += 1
                self._state.is_first_straddle = False
                logger.info(
                    "[ATL] Reform triggered: spot=%.2f drift=%.2f >= adj=%d "
                    "old_strike=%.0f -> new_strike=%.0f (hedges kept)",
                    spot, drift, adjustment_points,
                    self._state.straddle_strike, new_strike,
                )
                self._record_event(
                    "reform_trigger",
                    f"spot={spot:.2f} drift={drift:.2f} >= adj={adjustment_points} "
                    f"-> {int(self._state.straddle_strike)} → {int(new_strike)} (hedges kept)",
                )
                await self._convert_to_straddle(instrument, new_strike, spot, reform=True)

    async def _convert_to_straddle(self, instrument: InstrumentConfig, strike: float, spot: float, reform: bool = False) -> None:
        if (
            self._state.ce
            and self._state.pe
            and self._state.ce.strike == strike
            and self._state.pe.strike == strike
        ):
            # Already on the requested straddle strike — do not churn orders.
            self._state.phase = "STRADDLE"
            self._state.straddle_strike = strike
            self._state.straddle_sl_points = 0.0  # SL logic removed
            # Refresh the reference spot so the next adjustment trigger is
            # measured from the current spot, not the original entry.
            self._state.ref_spot = spot
            self._record_event("straddle", f"AT-STRIKE strike {int(strike)} spot {spot:.2f} (no switch)")
            return

        ce_q = await self._fetch_option_quote(instrument, strike, "CE")
        pe_q = await self._fetch_option_quote(instrument, strike, "PE")
        if not ce_q or not pe_q:
            return

        ce_leg = await self._build_leg(instrument, strike, "CE", fallback_quote=ce_q)
        pe_leg = await self._build_leg(instrument, strike, "PE", fallback_quote=pe_q)
        if not ce_leg or not pe_leg:
            return

        if not await self._switch_shorts(instrument, ce_leg, pe_leg, reason=("reform_straddle" if reform else "convert_straddle")):
            return

        self._state.phase = "STRADDLE"
        self._state.straddle_strike = strike
        self._state.ce = ce_leg
        self._state.pe = pe_leg
        # Legs just changed — reset the manual-exit miss counter and stamp
        # the change time so the detector's grace window kicks in. Without
        # this, the next 1-3 ticks see broker positions still reflecting
        # the OLD strike (or empty during the swap) and falsely declare
        # a manual exit, shutting the scanner down for the day.
        self._manual_exit_misses = 0
        self._last_legs_changed_at = datetime.now(IST)
        # Reset reference spot so next adjustment trigger is measured from
        # the spot at which this (re)formation happened.
        self._state.ref_spot = spot
        self._state.straddle_sl_points = 0.0  # SL logic removed
        # Hedges: only place if missing. On reform, existing hedges are
        # intentionally KEPT in place (per ATM-straddle spec) — we do not
        # roll/close them when the SELL legs move strikes.
        if self._settings.get("hedge_enabled", False) and (not self._state.hedge_ce or not self._state.hedge_pe):
            await self._ensure_hedges(instrument, strike)

        tag = "REFORM" if reform else "AT-STRIKE"
        hedge_note = ""
        if reform and self._state.hedge_ce and self._state.hedge_pe:
            hedge_note = (
                f"\nHedges held: CE {int(self._state.hedge_ce.strike)} / "
                f"PE {int(self._state.hedge_pe.strike)}"
            )
        self._record_event("straddle", f"{tag} strike {int(strike)} spot {spot:.2f}")
        await self.alert_manager.telegram.send(
            f"⚡ ATL {tag} STRADDLE ({self._mode_label()})\n"
            f"Spot: {spot:.2f}\n"
            f"Strike: {int(strike)}\n"
            f"CE ₹{self._state.ce.premium:.2f} + PE ₹{self._state.pe.premium:.2f}"
            f"{hedge_note}"
        )

    async def force_close(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        reason: str = "eod_force_close",
    ) -> None:
        if not self.is_in_trade():
            return
        spot = float(df_today.iloc[-1]["close"]) if df_today is not None and not df_today.empty else 0.0
        # Close active short legs first.
        await self._close_current_shorts(instrument, reason=reason)
        if self._settings.get("hedge_mode", "none") != "none" and (self._state.hedge_ce or self._state.hedge_pe):
            await self._close_hedges(instrument, reason=reason)
            await self.alert_manager.telegram.send(
                f"🛡️ ATL Hedge Close ({self._mode_label()})\n"
                f"BUY hedges exit: CE {int(self._state.hedge_ce.strike) if self._state.hedge_ce else '-'} / "
                f"PE {int(self._state.hedge_pe.strike) if self._state.hedge_pe else '-'}"
            )
        self._state.phase = "IDLE"
        self._state.hedge_ce = None
        self._state.hedge_pe = None
        if "priority_handoff" in reason:
            self._record_event("handoff", f"ATM closed due to {reason.replace('priority_handoff_', '').upper()} priority handoff @ spot {spot:.2f}")
        else:
            self._record_event("force_close", f"{reason} @ spot {spot:.2f}")
        self._complete_for_day(f"Force close completed ({reason})")
        await self.alert_manager.telegram.send(
            f"🔔 ATL Force Close\nIndex: {instrument.symbol}\nSpot: {spot:.2f}\nReason: {reason}"
        )

    async def _ensure_hedges(self, instrument: InstrumentConfig, ref_strike: float) -> None:
        """Initialize protective hedges once per day using target premium matching."""
        hedge_mode = str(self._settings.get("hedge_mode", "none")).lower()
        if hedge_mode == "none":
            return
        if self._state.hedge_ce and self._state.hedge_pe:
            return

        interval = max(1, int(self._settings.get("strike_interval", 50)))
        if hedge_mode == "otm_points":
            otm_points = max(1, int(self._settings.get("hedge_otm_points", 500) or 500))
            ce_strike = round((ref_strike + otm_points) / interval) * interval
            pe_strike = round((ref_strike - otm_points) / interval) * interval
            # Parallel quote fetch for the two hedge strikes.
            ce_q, pe_q = await asyncio.gather(
                self._fetch_option_quote(instrument, ce_strike, "CE"),
                self._fetch_option_quote(instrument, pe_strike, "PE"),
            )
            ce_leg, pe_leg = await asyncio.gather(
                self._build_leg(instrument, ce_strike, "CE", fallback_quote=ce_q),
                self._build_leg(instrument, pe_strike, "PE", fallback_quote=pe_q),
            )
        else:
            target_premium = max(1.0, float(self._settings.get("hedge_premium", 3) or 3))
            ce_leg, pe_leg = await asyncio.gather(
                self._find_hedge_leg(instrument, ref_strike, "CE", target_premium, interval),
                self._find_hedge_leg(instrument, ref_strike, "PE", target_premium, interval),
            )

        if ce_leg:
            self._state.hedge_ce = ce_leg
        if pe_leg:
            self._state.hedge_pe = pe_leg
        if self._state.hedge_ce or self._state.hedge_pe:
            hedge_lots = int(self._settings.get("hedge_lots", 0) or 0)
            short_lots = int(self._settings.get("lots", 1) or 1)
            lots = hedge_lots if hedge_lots > 0 else short_lots
            # Place hedge BUY orders in parallel — both must succeed.
            ce_task = (
                self._place_leg_order(instrument, self._state.hedge_ce, "BUY", lots, "hedge_entry")
                if self._state.hedge_ce else asyncio.sleep(0, result=True)
            )
            pe_task = (
                self._place_leg_order(instrument, self._state.hedge_pe, "BUY", lots, "hedge_entry")
                if self._state.hedge_pe else asyncio.sleep(0, result=True)
            )
            ce_hedge_ok, pe_hedge_ok = await asyncio.gather(ce_task, pe_task)
            hedge_failed = False
            if self._state.hedge_ce and not ce_hedge_ok:
                self._state.hedge_ce = None
                hedge_failed = True
            if self._state.hedge_pe and not pe_hedge_ok:
                self._state.hedge_pe = None
                hedge_failed = True

            if hedge_failed:
                await self._close_current_shorts(instrument, "hedge_leg_fail")
                await self._close_hedges(instrument, "hedge_leg_fail", force=True)
                self._state.ce = None
                self._state.pe = None
                self._state.hedge_ce = None
                self._state.hedge_pe = None
                self._complete_for_day("Hedge leg failed at broker; flattened and completed for today")
                return

            ce_strike = int(self._state.hedge_ce.strike) if self._state.hedge_ce else "-"
            ce_px = f"₹{self._state.hedge_ce.premium:.2f}" if self._state.hedge_ce else "-"
            pe_strike = int(self._state.hedge_pe.strike) if self._state.hedge_pe else "-"
            pe_px = f"₹{self._state.hedge_pe.premium:.2f}" if self._state.hedge_pe else "-"
            self._record_event("hedge", f"Hedge CE {ce_strike} / PE {pe_strike}")
            await self.alert_manager.telegram.send(
                f"🛡️ ATL Hedge Entry ({self._mode_label()})\n"
                f"BUY CE {ce_strike} @ {ce_px} / "
                f"BUY PE {pe_strike} @ {pe_px}"
            )

    async def _find_hedge_leg(
        self,
        instrument: InstrumentConfig,
        ref_strike: float,
        option_type: str,
        target_premium: float,
        interval: int,
    ) -> Optional[ATLLeg]:
        """Walk OTM and pick the FIRST strike whose LTP ≤ target premium.

        If no strike within range reaches the target, return the closest
        match seen (deepest OTM tried). The walk extends up to 80 strikes
        outward — SENSEX (interval 100) needs to cover ~8000 pts to find
        a ₹5–10 premium when spot is far from ATM; the previous 20-strike
        cap returned strikes still ~₹30–40 above target.
        """
        best: Optional[ATLLeg] = None
        best_diff: Optional[float] = None
        for step in range(1, 81):
            strike = ref_strike + (interval * step if option_type == "CE" else -interval * step)
            q = await self._fetch_option_quote(instrument, strike, option_type)
            if not q:
                continue
            ltp = float(q.get("ltp", 0) or 0)
            if ltp <= 0:
                continue
            # First strike at or below target wins outright — no need to
            # keep walking further OTM where LTP only gets smaller and
            # liquidity disappears.
            if ltp <= target_premium:
                candidate = await self._build_leg(instrument, strike, option_type, fallback_quote=q)
                if candidate:
                    return candidate
                # If build failed, fall back to closest-match logic below.
            diff = abs(ltp - target_premium)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                candidate = await self._build_leg(instrument, strike, option_type, fallback_quote=q)
                if candidate:
                    best = candidate
        if best is None:
            logger.warning(
                "[ATL] No %s hedge candidate found within 80 strikes of %.0f (target=%.2f)",
                option_type, ref_strike, target_premium,
            )
        return best

    def _record_event(self, event_type: str, message: str) -> None:
        self._events.append({
            "time": datetime.now(IST).strftime("%H:%M:%S"),
            "event": event_type,
            "message": message,
            "mode": "live" if self._is_live() else "paper",
        })
        if len(self._events) > 200:
            self._events = self._events[-200:]
        # Persist immediately so a backend restart preserves both the
        # event log and the strategy state (legs, ref_spot, roll/reform
        # counters). Without this the scanner forgets where it entered
        # and cannot compute roll/reform triggers after a restart.
        self._save_state_to_disk()

    def _record_diag(self, key: str, message: str) -> None:
        """Record a diagnostic gate-skip event without spamming the timeline.

        Same `key` only emits once per minute per state so the user can see
        WHY the scanner is idle without flooding the UI with duplicates.
        """
        now = datetime.now(IST)
        last = self._diag_last.get(key)
        if last and (now - last).total_seconds() < 60:
            return
        self._diag_last[key] = now
        # NOTE: do NOT push diag/skip events into self._events. They are
        # not shown in the UI timeline (frontend filters them out), and
        # at ~2 skips/minute they would fill the 200-entry cap within
        # ~1.5 hours and evict real lifecycle events (entry, hedge,
        # complete, manual_exit_detected) — leaving the user with an
        # apparently-empty timeline. Log only.
        logger.info("ATL[%s] skip: %s", self._settings.get("index", "?"), message)

    def _trip_halt(self, reason: str) -> None:
        """Trip the per-day circuit breaker. Stops further entry attempts
        until the user explicitly resets via /api/atm/reset or the next
        trading day.
        """
        if self._state.halted:
            return
        self._state.halted = True
        self._state.halt_reason = reason
        self._state.done_for_day = True  # also stop other gates
        self._record_event("halt", reason)
        logger.error("ATL[%s] HALTED: %s", self._settings.get("index", "?"), reason)

    def reset_halt(self) -> None:
        """Clear the circuit breaker AND in-memory entry state so the
        strategy can attempt entry again. Typically called from the UI
        'Reset' button after the user fixed the broker-side issue or
        manually exited orders at the broker. Without clearing the
        entered/leg state, the scanner would keep tracking phantom legs
        and never place new orders even when settings are updated."""
        was_halted = self._state.halted
        was_entered = self._state.entered
        self._state.halted = False
        self._state.halt_reason = ""
        self._state.done_for_day = False
        # Wipe phantom position state — caller is asserting that nothing
        # is open at the broker side.
        self._state.phase = "IDLE"
        self._state.entered = False
        self._state.entry_in_progress = False
        self._state.ce = None
        self._state.pe = None
        self._state.hedge_ce = None
        self._state.hedge_pe = None
        self._state.straddle_strike = 0.0
        self._state.straddle_sl_points = 0.0
        self._state.is_first_straddle = True
        self._state.roll_count = 0
        self._state.reform_count = 0
        self._diag_last.clear()
        if was_halted or was_entered:
            self._record_event(
                "reset",
                "Reset by user — halt cleared and in-memory position state "
                "wiped; scanner may retry entry",
            )

    def request_force_entry(self) -> None:
        """Request a fresh entry on the next cycle (UI 'Place Now').

        Bypasses the entry-time gate AND clears phantom in-memory position
        state so that if the user previously exited at the broker manually,
        the scanner doesn't think it's still holding legs and skip the
        entry block entirely. Also clears halt/done flags so a single
        broker rejection doesn't permanently block manual retries.
        """
        self._state.halted = False
        self._state.halt_reason = ""
        self._state.done_for_day = False
        # Clear phantom legs/state so the entry block actually runs.
        self._state.phase = "IDLE"
        self._state.entered = False
        self._state.entry_in_progress = False
        self._state.ce = None
        self._state.pe = None
        self._state.hedge_ce = None
        self._state.hedge_pe = None
        self._state.straddle_strike = 0.0
        self._state.straddle_sl_points = 0.0
        self._state.is_first_straddle = True
        self._state.roll_count = 0
        self._state.reform_count = 0
        self._diag_last.clear()
        self._force_entry = True
        self._record_event(
            "force_entry_requested",
            "Place Now requested — phantom state cleared, entry-time gate bypassed",
        )

    def _leg_dict(self, leg: Optional[ATLLeg]) -> Optional[dict]:
        if not leg:
            return None
        return {
            "option_type": leg.option_type,
            "strike": leg.strike,
            "symbol": leg.symbol,
            "symboltoken": leg.symboltoken,
            "exchange": leg.exchange,
            "premium": leg.premium,
        }

    async def _build_leg(
        self,
        instrument: InstrumentConfig,
        strike: float,
        option_type: str,
        fallback_quote: Optional[dict] = None,
    ) -> Optional[ATLLeg]:
        if not self._expiry:
            return None
        symbol = instrument.build_option_symbol(self._expiry, strike, option_type)
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            return None
        quote = fallback_quote or await self._fetch_option_quote(instrument, strike, option_type)
        ltp = float((quote or {}).get("ltp", 0) or 0)
        return ATLLeg(
            option_type=option_type,
            strike=strike,
            symbol=token_info.get("tradingsymbol", symbol),
            symboltoken=token_info.get("symboltoken", ""),
            exchange=token_info.get("exch_seg", "NFO"),
            premium=ltp,
        )

    async def _place_leg_order_via_broker(
        self,
        instrument: InstrumentConfig,
        leg: ATLLeg,
        side: str,
        qty: int,
        reason: str,
        broker=None,
    ) -> bool:
        """Place an ATL leg via the configured broker (Kite/Angel/Dhan)
        abstraction.

        Symbol-string differences between brokers are absorbed inside each
        broker adapter — KiteBroker translates Angel-format symbols to Kite
        tradingsymbols on the fly via KiteClient.resolve_from_angel_symbol.
        """
        target = broker if broker is not None else self.broker
        request = OrderRequest(
            instrument=instrument,
            trading_symbol=leg.symbol,
            symbol_token=leg.symboltoken,
            exchange=leg.exchange or "NFO",
            side=OrderSide(side),
            order_type=OrderType.MARKET,
            product_type=ProductType.CARRYFORWARD,
            quantity=qty,
            price=0.0,
            trigger_price=0.0,
            # Structured fields let the Kite adapter resolve the contract from
            # the broker's instrument master without parsing symbol strings —
            # important because SENSEX/BFO uses a different symbol format than
            # NFO and the two brokers' tradingsymbols don't match either.
            underlying=instrument.option_symbol_prefix or instrument.symbol,
            expiry_date=self._expiry_date,
            strike=float(leg.strike),
            option_type=leg.option_type,
            # ATL must know the FINAL order outcome before moving on. With
            # fast-ack mode, the broker returns OPEN as soon as it accepts
            # the order — but exchange-side rejections (insufficient
            # margin, freeze qty, lot multiple, etc.) arrive seconds later
            # and would otherwise leave the strategy with one short and
            # two naked hedges. Force terminal-status polling for every
            # entry/exit/rollback leg so the rollback path actually fires.
            wait_for_terminal=True,
        )
        try:
            resp = await asyncio.to_thread(target.place_order, request)
        except Exception:
            logger.exception(
                "ATL broker order exception (%s): %s %s %s",
                target.name if target else "?", side, leg.symbol, reason,
            )
            self._record_event("order_error", f"{side} {leg.symbol} exception ({reason})")
            return False

        if not resp or resp.status == OrderStatus.REJECTED:
            msg = (resp.message if resp else "no response") or "rejected"
            logger.error(
                "ATL broker order rejected: %s %s id=%s reason=%s msg=%s",
                side, leg.symbol, getattr(resp, "order_id", ""), reason, msg,
            )
            self._record_event("order_error", f"{side} {leg.symbol} rejected ({msg})")
            return False

        # Fast-ack mode returns OPEN/PENDING once the broker accepts the order.
        # Treat that as successful placement to avoid false strategy halts.
        if resp.status in {OrderStatus.OPEN, OrderStatus.PENDING}:
            self._record_event(
                "order",
                f"{side} {leg.symbol} qty={qty} id={resp.order_id} accepted ({reason})",
            )
            return True

        if resp.status == OrderStatus.COMPLETE or resp.filled_price > 0:
            if resp.filled_price > 0:
                leg.premium = resp.filled_price
            self._record_event(
                "order",
                f"{side} {leg.symbol} qty={qty} id={resp.order_id} ({reason})",
            )
            return True

        logger.error(
            "ATL broker order unresolved: %s %s id=%s status=%s",
            side, leg.symbol, resp.order_id, resp.status.value,
        )
        self._record_event(
            "order_error",
            f"{side} {leg.symbol} unresolved status={resp.status.value} ({reason})",
        )
        return False

    async def _place_leg_order(
        self,
        instrument: InstrumentConfig,
        leg: ATLLeg,
        side: str,
        lots: int,
        reason: str,
    ) -> bool:
        if not self._is_live():
            return True
        if not leg.symbol or not leg.symboltoken:
            return False
        qty = max(1, int(lots)) * max(1, int(instrument.lot_size))

        # Always route through the broker abstraction. The broker adapter
        # (AngelOne / Kite / Dhan) handles SDK-specific details. Per-strategy
        # override: pick from execution_account each cycle so the user can
        # change the strategy's broker without restarting the orchestrator.
        broker = self._resolve_broker()
        if broker is not None:
            return await self._place_leg_order_via_broker(
                instrument, leg, side, qty, reason, broker=broker,
            )

        try:
            self.client.ensure_authenticated()
            params = {
                "variety": "NORMAL",
                "tradingsymbol": leg.symbol,
                "symboltoken": leg.symboltoken,
                "transactiontype": side,
                "exchange": leg.exchange or "NFO",
                "ordertype": "MARKET",
                "producttype": "CARRYFORWARD",
                "duration": "DAY",
                "quantity": str(qty),
                "price": "0",
                "triggerprice": "0",
            }
            resp = await asyncio.to_thread(self.client._smart_api.placeOrder, params)
            order_id = self._extract_order_id(resp)
            if not order_id:
                logger.error("ATL live order failed %s %s %s: %s", side, leg.symbol, reason, resp)
                self._record_event("order_error", f"{side} {leg.symbol} failed ({reason})")
                return False

            status = await self._wait_order_terminal_status(order_id, expected_qty=qty)
            order_state = status.get("status", "")
            fill_price = float(status.get("averageprice", 0) or 0)

            if order_state == "complete" or fill_price > 0:
                if fill_price > 0:
                    leg.premium = fill_price
                self._record_event("order", f"{side} {leg.symbol} qty={qty} id={order_id} ({reason})")
                return True

            if order_state == "rejected":
                rej = status.get("text", "") or "rejected by broker"
                logger.error("ATL order rejected %s %s id=%s reason=%s", side, leg.symbol, order_id, rej)
                self._record_event("order_error", f"{side} {leg.symbol} rejected ({rej})")
                return False

            logger.error(
                "ATL live order unresolved %s %s id=%s status=%s details=%s",
                side,
                leg.symbol,
                order_id,
                order_state or "unknown",
                status,
            )
            self._record_event(
                "order_error",
                f"{side} {leg.symbol} unresolved status={order_state or 'unknown'} ({reason})",
            )
            return False
        except Exception:
            logger.exception("ATL live order exception: %s %s %s", side, leg.symbol, reason)
            self._record_event("order_error", f"{side} {leg.symbol} exception ({reason})")
            return False

    def _extract_order_id(self, resp: Any) -> str:
        """Extract order id from SmartAPI placeOrder response shapes."""
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            data = resp.get("data")
            if isinstance(data, dict):
                oid = data.get("orderid") or data.get("orderId") or data.get("order_id")
                if oid:
                    return str(oid).strip()
            oid = resp.get("orderid") or resp.get("orderId") or resp.get("order_id")
            if oid:
                return str(oid).strip()
        return ""

    async def _wait_order_terminal_status(self, order_id: str, expected_qty: int) -> dict:
        """Poll broker order status until complete/rejected or timeout."""
        for _ in range(ORDER_POLL_RETRIES):
            info = await self._get_order_info(order_id)
            state = str(info.get("status", "") or "").strip().lower()
            if state in {"complete", "rejected", "cancelled", "canceled"}:
                return info

            filled = 0
            try:
                filled = int(float(info.get("filledshares", 0) or 0))
            except Exception:
                filled = 0
            avg = float(info.get("averageprice", 0) or 0)
            if filled >= expected_qty and avg > 0:
                info["status"] = "complete"
                return info

            await asyncio.sleep(ORDER_POLL_DELAY_SECONDS)

        # Return latest snapshot if we timed out.
        return await self._get_order_info(order_id)

    async def _get_order_info(self, order_id: str) -> dict:
        """Read order status from order book or individual order endpoint."""
        try:
            # Newer SmartAPI deployments sometimes expose individual order details.
            if hasattr(self.client._smart_api, "individual_order_details"):
                resp = await asyncio.to_thread(self.client._smart_api.individual_order_details, order_id)
                data = resp.get("data") if isinstance(resp, dict) else None
                if data:
                    rows = data if isinstance(data, list) else [data]
                    for row in rows:
                        if str(row.get("orderid", "")) == str(order_id):
                            return {
                                "status": row.get("orderstatus", ""),
                                "text": row.get("text", ""),
                                "averageprice": row.get("averageprice", 0),
                                "filledshares": row.get("filledshares", "0"),
                            }
        except Exception:
            pass

        try:
            resp = await asyncio.to_thread(self.client._smart_api.orderBook)
            rows = resp.get("data", []) if isinstance(resp, dict) else []
            for row in rows:
                if str(row.get("orderid", "")) == str(order_id):
                    return {
                        "status": row.get("orderstatus", ""),
                        "text": row.get("text", ""),
                        "averageprice": row.get("averageprice", 0),
                        "filledshares": row.get("filledshares", "0"),
                    }
        except Exception:
            logger.debug("ATL order info lookup failed for %s", order_id)
        return {}

    async def _execute_entry_legs(self, instrument: InstrumentConfig, ce_leg: ATLLeg, pe_leg: ATLLeg, reason: str) -> bool:
        lots = int(self._settings.get("lots", 1) or 1)
        # Place both SELL legs in parallel — they're independent broker calls.
        # Order-tier sequencing (hedges before shorts) is preserved by callers.
        ce_ok, pe_ok = await asyncio.gather(
            self._place_leg_order(instrument, ce_leg, "SELL", lots, reason),
            self._place_leg_order(instrument, pe_leg, "SELL", lots, reason),
        )
        if ce_ok and pe_ok:
            return True
        # Best-effort rollback if one leg entered but the other failed.
        if ce_ok and not pe_ok:
            await self._place_leg_order(instrument, ce_leg, "BUY", lots, f"rollback_{reason}")
        if pe_ok and not ce_ok:
            await self._place_leg_order(instrument, pe_leg, "BUY", lots, f"rollback_{reason}")
        # Ensure no residual exposure remains on leg mismatch.
        await self._close_current_shorts(instrument, f"flatten_{reason}")
        await self._close_hedges(instrument, f"flatten_{reason}", force=True)
        self._state.ce = None
        self._state.pe = None
        self._state.hedge_ce = None
        self._state.hedge_pe = None
        self._complete_for_day("One or more legs failed at broker; flattened and completed for today")
        return False

    async def _broker_open_symbols(self) -> Optional[set[str]]:
        """Return the set of trading symbols currently open at the broker.

        Used to skip exit orders for legs the user has already closed
        manually at the broker (otherwise the scanner would place
        orphan SELL/BUY exit orders that have no underlying position
        and end up sitting in the order book or hitting position limits).

        Returns ``None`` when the broker can't be queried \u2014 the caller
        should then fall back to the engine-state behaviour (i.e. attempt
        the close, since we don't actually know).
        """
        broker = self._resolve_broker()
        if broker is None:
            return None
        get_positions = getattr(broker, "get_positions", None)
        if not callable(get_positions):
            return None
        try:
            positions = await asyncio.to_thread(get_positions)
        except Exception:
            logger.exception("[ATL] Could not query broker positions for exit gating")
            return None
        symbols: set[str] = set()
        for p in positions or []:
            sym = ""
            qty = 0
            if hasattr(p, "trading_symbol"):
                sym = (p.trading_symbol or "").strip()
                qty = int(getattr(p, "quantity", 0) or 0)
            elif isinstance(p, dict):
                sym = str(
                    p.get("trading_symbol")
                    or p.get("tradingsymbol")
                    or p.get("tradingSymbol")
                    or ""
                ).strip()
                qty = int(
                    p.get("quantity")
                    or p.get("netqty")
                    or p.get("netQty")
                    or 0
                )
            if sym and qty != 0:
                symbols.add(sym.upper())
        return symbols

    async def _broker_open_strike_legs(self) -> Optional[set[tuple[int, str]]]:
        """Return open broker positions as a set of ``(strike, 'CE'|'PE')`` tuples.

        Format-agnostic alternative to :meth:`_broker_open_symbols` —
        works regardless of whether the broker returns trading symbols in
        AngelOne style (``NIFTY19MAY2624100CE``), Dhan style
        (``NIFTY-19May2026-23700-CE`` / ``NIFTY 19 MAY 23700 CALL``) or
        Kite style (``NIFTY26MAY23700CE``). Used by the manual-exit
        detector so a symbol-format mismatch can never falsely declare
        the legs gone.

        Returns ``None`` when the broker can't be queried so the caller
        falls back to the safe behaviour (assume legs are present).
        """
        broker = self._resolve_broker()
        if broker is None:
            return None
        get_positions = getattr(broker, "get_positions", None)
        if not callable(get_positions):
            return None
        try:
            positions = await asyncio.to_thread(get_positions)
        except Exception:
            logger.exception("[ATL] Could not query broker positions for manual-exit detector")
            return None
        out: set[tuple[int, str]] = set()
        for p in positions or []:
            sym = ""
            qty = 0
            if hasattr(p, "trading_symbol"):
                sym = (p.trading_symbol or "")
                qty = int(getattr(p, "quantity", 0) or 0)
            elif isinstance(p, dict):
                sym = str(
                    p.get("trading_symbol")
                    or p.get("tradingsymbol")
                    or p.get("tradingSymbol")
                    or ""
                )
                qty = int(
                    p.get("quantity")
                    or p.get("netqty")
                    or p.get("netQty")
                    or 0
                )
            if not sym or qty == 0:
                continue
            s = sym.upper()
            # Option type: any of CE, PE, CALL, PUT (case-insensitive,
            # punctuation-tolerant).
            if re.search(r"(?:^|[^A-Z])CALL(?:$|[^A-Z])", s) or re.search(r"CE(?:$|[^A-Z0-9])", s):
                opt = "CE"
            elif re.search(r"(?:^|[^A-Z])PUT(?:$|[^A-Z])", s) or re.search(r"PE(?:$|[^A-Z0-9])", s):
                opt = "PE"
            else:
                continue
            # Strike: pick the largest plausible integer in the symbol.
            # Plausible = at least 3 digits long (covers MIDCPNIFTY 8000
            # → BANKNIFTY 95000). Year (2-digit) and day-of-month (1-2
            # digit) tokens are too short and naturally fall out.
            nums = [int(x) for x in re.findall(r"\d+", s) if len(x) >= 3]
            if not nums:
                continue
            strike = max(nums)
            out.add((strike, opt))
        return out

    async def _close_current_shorts(self, instrument: InstrumentConfig, reason: str) -> bool:
        lots = int(self._settings.get("lots", 1) or 1)
        ok = True
        # Skip legs the user has already closed manually at the broker so we
        # don't fire orphan exit orders against zero positions. Compare by
        # (strike, CE/PE) — symbol-string comparison breaks for Dhan because
        # its tradingSymbol format differs from the AngelOne format stored
        # in state.
        open_legs = await self._broker_open_strike_legs()
        if self._state.ce:
            key = (int(round(self._state.ce.strike)), "CE")
            if open_legs is not None and key not in open_legs:
                logger.info(
                    "[ATL] Skipping CE close (%s): broker shows no open position. reason=%s",
                    self._state.ce.symbol, reason,
                )
                self._record_event(
                    "exit_skipped",
                    f"CE {self._state.ce.symbol} not held at broker (already closed manually)",
                )
            else:
                ok = await self._place_leg_order(instrument, self._state.ce, "BUY", lots, reason) and ok
        if self._state.pe:
            key = (int(round(self._state.pe.strike)), "PE")
            if open_legs is not None and key not in open_legs:
                logger.info(
                    "[ATL] Skipping PE close (%s): broker shows no open position. reason=%s",
                    self._state.pe.symbol, reason,
                )
                self._record_event(
                    "exit_skipped",
                    f"PE {self._state.pe.symbol} not held at broker (already closed manually)",
                )
            else:
                ok = await self._place_leg_order(instrument, self._state.pe, "BUY", lots, reason) and ok
        return ok

    async def _close_hedges(self, instrument: InstrumentConfig, reason: str, force: bool = False) -> bool:
        hedge_lots = int(self._settings.get("hedge_lots", 0) or 0)
        short_lots = int(self._settings.get("lots", 1) or 1)
        lots = hedge_lots if hedge_lots > 0 else short_lots
        ok = True
        # ``force=True`` is used by entry-rollback paths where the hedges
        # were placed seconds ago in the same call and MUST be unwound
        # regardless of what the broker's positions endpoint reports. The
        # default broker-position guard exists only to avoid orphan exits
        # against legs the user manually closed mid-day.
        open_legs = None if force else await self._broker_open_strike_legs()
        if self._state.hedge_ce:
            key = (int(round(self._state.hedge_ce.strike)), "CE")
            if open_legs is not None and key not in open_legs:
                logger.info(
                    "[ATL] Skipping HEDGE CE close (%s): broker shows no open position. reason=%s",
                    self._state.hedge_ce.symbol, reason,
                )
                self._record_event(
                    "exit_skipped",
                    f"HEDGE CE {self._state.hedge_ce.symbol} not held at broker",
                )
            else:
                ok = await self._place_leg_order(instrument, self._state.hedge_ce, "SELL", lots, reason) and ok
        if self._state.hedge_pe:
            key = (int(round(self._state.hedge_pe.strike)), "PE")
            if open_legs is not None and key not in open_legs:
                logger.info(
                    "[ATL] Skipping HEDGE PE close (%s): broker shows no open position. reason=%s",
                    self._state.hedge_pe.symbol, reason,
                )
                self._record_event(
                    "exit_skipped",
                    f"HEDGE PE {self._state.hedge_pe.symbol} not held at broker",
                )
            else:
                ok = await self._place_leg_order(instrument, self._state.hedge_pe, "SELL", lots, reason) and ok
        return ok

    async def _switch_shorts(self, instrument: InstrumentConfig, new_ce: ATLLeg, new_pe: ATLLeg, reason: str) -> bool:
        old_ce = self._state.ce
        old_pe = self._state.pe
        logger.info(
            "[ATL] _switch_shorts: %s | old CE=%s PE=%s -> new CE=%s PE=%s",
            reason,
            old_ce.strike if old_ce else "-", old_pe.strike if old_pe else "-",
            new_ce.strike, new_pe.strike,
        )
        if not await self._close_current_shorts(instrument, f"close_{reason}"):
            await self.alert_manager.telegram.send(f"⚠️ ATL {reason}: failed to close current short legs")
            self._complete_for_day("Failed to close current legs during switch; completed for today")
            return False
        if await self._execute_entry_legs(instrument, new_ce, new_pe, f"open_{reason}"):
            return True
        # Best-effort rollback: restore prior short if new placement failed.
        await self.alert_manager.telegram.send(f"⚠️ ATL {reason}: failed opening new legs, attempting rollback")
        if old_ce and old_pe:
            await self._execute_entry_legs(instrument, old_ce, old_pe, f"rollback_{reason}")
        await self._close_current_shorts(instrument, f"flatten_{reason}")
        await self._close_hedges(instrument, f"flatten_{reason}")
        self._state.ce = None
        self._state.pe = None
        self._state.hedge_ce = None
        self._state.hedge_pe = None
        self._complete_for_day("Switch leg placement failed; flattened and completed for today")
        return False

    async def _roll_strangle(self, instrument: InstrumentConfig, new_ce_strike: float, new_pe_strike: float, reason: str) -> bool:
        ce_q = await self._fetch_option_quote(instrument, new_ce_strike, "CE")
        pe_q = await self._fetch_option_quote(instrument, new_pe_strike, "PE")
        if not ce_q or not pe_q:
            return False
        new_ce = await self._build_leg(instrument, new_ce_strike, "CE", fallback_quote=ce_q)
        new_pe = await self._build_leg(instrument, new_pe_strike, "PE", fallback_quote=pe_q)
        if not new_ce or not new_pe:
            return False
        if not await self._switch_shorts(instrument, new_ce, new_pe, reason=reason):
            return False
        self._state.ce = new_ce
        self._state.pe = new_pe
        # Legs just changed — reset manual-exit guard (see _convert_to_straddle).
        self._manual_exit_misses = 0
        self._last_legs_changed_at = datetime.now(IST)
        self._record_event("roll", f"{reason}: CE {int(new_ce_strike)} PE {int(new_pe_strike)}")
        return True

    async def _fetch_option_quote(self, instrument: InstrumentConfig, strike: float, option_type: str) -> Optional[dict]:
        if not self._expiry:
            return None
        symbol = instrument.build_option_symbol(self._expiry, strike, option_type)
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            return None
        try:
            quote = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.get_option_quote,
                    token_info.get("exch_seg", "NFO"),
                    token_info.get("tradingsymbol", ""),
                    token_info.get("symboltoken", ""),
                ),
                timeout=12,
            )
            return quote
        except Exception:
            return None
