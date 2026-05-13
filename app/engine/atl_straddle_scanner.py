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
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

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
    ):
        self.client = client
        self.alert_manager = alert_manager
        self.broker = broker
        self._settings = load_atl_settings()
        self._today_str = ""
        self._expiry = ""
        self._expiry_date: Optional[date] = None
        self._state = ATLState()
        self._events: list[dict] = []
        self._diag_last: dict[str, datetime] = {}
        self._force_entry: bool = False
        self._last_done_signature: str = ""

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._settings = load_atl_settings()
        self._state = ATLState()
        self._events = []
        self._diag_last = {}
        self._force_entry = False
        self._last_done_signature = ""

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
        sl_type = str(self._settings.get("sl_type", "premium_pct")).lower()
        sl_lower = float(self._settings.get("sl_lower", 0) or 0)
        sl_upper = float(self._settings.get("sl_upper", 0) or 0)

        if sl_type == "spot" and self._state.entered:
            lower_hit = sl_lower > 0 and spot <= sl_lower
            upper_hit = sl_upper > 0 and spot >= sl_upper
            if lower_hit or upper_hit:
                self._record_event(
                    "stoploss",
                    f"SPOT SL breach spot={spot:.2f} lower={sl_lower:.2f} upper={sl_upper:.2f}",
                )
                await self.alert_manager.telegram.send(
                    f"🛑 ATL SPOT SL HIT\n"
                    f"Spot: {spot:.2f}\n"
                    f"SL Range: {sl_lower:.2f} - {sl_upper:.2f}"
                )
                await self.force_close(df_today, instrument, reason="spot_sl_breach")
                return

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
            if ce_strike == pe_strike:
                # Entry is already ATM straddle (offset=0). Avoid immediate
                # close/reopen of identical strikes on the next touch check.
                self._state.phase = "STRADDLE"
                self._state.straddle_strike = ce_strike
                prem_sum = self._state.ce.premium + self._state.pe.premium
                self._state.straddle_sl_points = (
                    prem_sum * (int(self._settings.get("first_straddle_sl_pct", 100)) / 100.0)
                    if sl_type == "premium_pct"
                    else 0.0
                )
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

        # STRADDLE phase logic
        if sl_type == "premium_pct" and self._state.phase == "STRADDLE" and self._state.straddle_sl_points > 0:
            upper = self._state.straddle_strike + self._state.straddle_sl_points
            lower = self._state.straddle_strike - self._state.straddle_sl_points
            if spot >= upper or spot <= lower:
                new_strike = round(spot / interval) * interval
                self._state.reform_count += 1
                self._state.is_first_straddle = False
                logger.info(
                    "[ATL] Reform triggered: spot=%.2f outside [%.2f, %.2f] "
                    "(strike=%.2f sl_points=%.2f) -> new_strike=%.2f",
                    spot, lower, upper, self._state.straddle_strike,
                    self._state.straddle_sl_points, new_strike,
                )
                self._record_event(
                    "reform_trigger",
                    f"spot={spot:.2f} outside [{lower:.2f}, {upper:.2f}] "
                    f"-> new strike {int(new_strike)}",
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
            if str(self._settings.get("sl_type", "premium_pct")).lower() == "premium_pct":
                pct = int(self._settings.get("first_straddle_sl_pct", 100)) if self._state.is_first_straddle else int(self._settings.get("reform_straddle_sl_pct", 60))
                self._state.straddle_sl_points = (self._state.ce.premium + self._state.pe.premium) * (pct / 100.0)
            else:
                self._state.straddle_sl_points = 0.0
            self._record_event("straddle", f"AT-STRIKE strike {int(strike)} spot {spot:.2f} sl {self._state.straddle_sl_points:.2f} (no switch)")
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
        if self._settings.get("hedge_enabled", False) and (not self._state.hedge_ce or not self._state.hedge_pe):
            await self._ensure_hedges(instrument, strike)

        sl_type = str(self._settings.get("sl_type", "premium_pct")).lower()
        pct = int(self._settings.get("first_straddle_sl_pct", 100)) if self._state.is_first_straddle else int(self._settings.get("reform_straddle_sl_pct", 60))
        prem_sum = self._state.ce.premium + self._state.pe.premium
        self._state.straddle_sl_points = prem_sum * (pct / 100.0) if sl_type == "premium_pct" else 0.0

        tag = "REFORM" if reform else "AT-STRIKE"
        self._record_event("straddle", f"{tag} strike {int(strike)} spot {spot:.2f} sl {self._state.straddle_sl_points:.2f}")
        await self.alert_manager.telegram.send(
            f"⚡ ATL {tag} STRADDLE ({self._mode_label()})\n"
            f"Spot: {spot:.2f}\n"
            f"Strike: {int(strike)}\n"
            f"CE ₹{self._state.ce.premium:.2f} + PE ₹{self._state.pe.premium:.2f}\n"
            f"SL points: {self._state.straddle_sl_points:.2f}"
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
        self._record_event("skip", message)
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

    async def _close_current_shorts(self, instrument: InstrumentConfig, reason: str) -> bool:
        lots = int(self._settings.get("lots", 1) or 1)
        ok = True
        # Skip legs the user has already closed manually at the broker so we
        # don't fire orphan exit orders against zero positions.
        open_syms = await self._broker_open_symbols()
        if self._state.ce:
            sym = (self._state.ce.symbol or "").upper()
            if open_syms is not None and sym and sym not in open_syms:
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
            sym = (self._state.pe.symbol or "").upper()
            if open_syms is not None and sym and sym not in open_syms:
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
        # against legs the user manually closed mid-day; it falsely
        # suppresses rollbacks when the broker reports trading symbols in
        # a different format (e.g. Dhan vs AngelOne SENSEX symbology).
        open_syms = None if force else await self._broker_open_symbols()
        if self._state.hedge_ce:
            sym = (self._state.hedge_ce.symbol or "").upper()
            if open_syms is not None and sym and sym not in open_syms:
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
            sym = (self._state.hedge_pe.symbol or "").upper()
            if open_syms is not None and sym and sym not in open_syms:
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
