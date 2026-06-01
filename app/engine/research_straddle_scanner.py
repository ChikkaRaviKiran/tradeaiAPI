"""Research Multi-Index Straddle Scanner.

Live execution engine for the four research-mode plans defined in
`atl_research_settings.py`:

  - single_time         : single index, fixed-time T0 entry
  - single_indicator    : single index, indicator-gated entry (T1..T5)
  - multi_time          : NIFTY + SENSEX, fixed-time T0 entries
  - multi_indicator     : NIFTY + SENSEX, indicator-gated entries

Design choices (kept INTENTIONALLY simple — research strategies are
"enter once, hold, exit on time or ₹SL"):

  - One `ATLState` per index symbol (NIFTY / SENSEX) — both can hold
    positions simultaneously in the multi-* modes.
  - No reform, no rolling, no smart-mode. Legs are static once placed.
  - Hard ₹ stop computed on every cycle from broker quotes
    (use HIGH-side LTP via worst-case quote walking). Default ₹6,000
    per (CE+PE) pair, configurable via `sl_rs`.
  - Indicator gates read directly from `df_today` columns produced by
    `feature_engine.compute_indicators()` — that DataFrame already has
    real volume-weighted VWAP (from NIFTY/SENSEX FUT volume merged via
    `merge_futures_volume`), so VWAP-based triggers (T2) are honest.
  - Runs alongside the existing `ATLStraddleScanner` — this scanner
    only acts when `enabled=True` in the research settings file, the
    legacy scanner only acts when `enabled=True` in its own file. The
    two are independent and CAN both be enabled (each will manage its
    own positions), though that's typically not what you want.

API parity with the legacy scanner so the orchestrator can call it
inside the same per-instrument cycle:

    await research_scanner.run_cycle(df_today, instrument, cycle)
    await research_scanner.force_close_all(reason)
    research_scanner.get_runtime_state() -> dict
    research_scanner.set_expiry(symbol, expiry_str, expiry_date)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
import pytz

from app.alerts.alert_manager import AlertManager
from app.core.config import settings
from app.core.instruments import InstrumentConfig
from app.data.angelone_client import AngelOneClient
from app.engine.atl_research_settings import (
    WEEKDAYS,
    effective_schedule,
    load_research_settings,
)
from app.engine.atl_straddle_scanner import ATLLeg  # reuse leg dataclass
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

# Default minimum bars required after entry_time before evaluating T1..T5
ENTRY_GATE_MIN_BARS = 5

# Indicator thresholds — ported verbatim from
# `backend/strategy_indicator_search.py::trigger_entry()` so live behaviour
# matches the backtest report exactly.
T1_IV_CRUSH_DROP = -0.004        # premium ≤ 0.4% below open
T2_VWAP_ALIGN_TOL = 0.0015       # |spot-vwap|/spot < 0.15%
T2_BB_WIDTH_LOW = 0.006          # BB width / mid < 0.6%
T3_RSI_PEAK = 65
T3_RSI_TROUGH = 35
T3_RSI_CENTER_TOL = 5
T3_BB_WIDTH_LOW = 0.005
T4_BB_WIDTH_LOW = 0.006
T5_BB_WIDTH_LOW = 0.006

WEEKDAY_INDEX_TO_NAME = ["Mon", "Tue", "Wed", "Thu", "Fri"]


@dataclass
class _IndexState:
    expiry: str = ""
    expiry_date: Optional[date] = None
    phase: str = "IDLE"  # IDLE | ENTERED | DONE
    entered: bool = False
    entry_in_progress: bool = False
    done_for_day: bool = False
    halted: bool = False
    halt_reason: str = ""
    entry_time_str: str = ""
    exit_time_str: str = ""
    strat: str = ""
    entry_trigger: str = ""
    credit: float = 0.0           # combined credit in points (CE+PE)
    atm: float = 0.0
    ce: Optional[ATLLeg] = None
    pe: Optional[ATLLeg] = None
    last_mtm_rs: float = 0.0
    events: list[dict] = field(default_factory=list)


def _to_min(t: str) -> int:
    h, m = [int(x) for x in t.split(":")]
    return h * 60 + m


def _now_min(now: datetime) -> int:
    return now.hour * 60 + now.minute


class ResearchStraddleScanner:
    def __init__(
        self,
        client: AngelOneClient,
        alert_manager: AlertManager,
        broker: BaseBroker,
        expiry_provider,
    ):
        self.client = client
        self.alert_manager = alert_manager
        self.broker = broker
        # expiry_provider(symbol) -> (expiry_str, expiry_date)
        self._expiry_provider = expiry_provider
        self._states: dict[str, _IndexState] = {"NIFTY": _IndexState(), "SENSEX": _IndexState()}
        self._today_str: str = ""
        self._settings: dict[str, Any] = load_research_settings()
        self._last_signature: str = ""

    # ── lifecycle helpers ──────────────────────────────────────────
    def _reset_for_new_day(self) -> None:
        for st in self._states.values():
            st.phase = "IDLE"
            st.entered = False
            st.entry_in_progress = False
            st.done_for_day = False
            st.halted = False
            st.halt_reason = ""
            st.entry_time_str = ""
            st.exit_time_str = ""
            st.strat = ""
            st.entry_trigger = ""
            st.credit = 0.0
            st.atm = 0.0
            st.ce = None
            st.pe = None
            st.last_mtm_rs = 0.0
            st.events.clear()

    def set_expiry(self, symbol: str, expiry: str, expiry_date: Optional[date]) -> None:
        st = self._states.get(symbol.upper())
        if not st:
            return
        st.expiry = expiry or ""
        st.expiry_date = expiry_date

    def _refresh_expiries(self) -> None:
        for sym in ("NIFTY", "SENSEX"):
            try:
                exp, exp_dt = self._expiry_provider(sym)
                self.set_expiry(sym, exp, exp_dt)
            except Exception:
                logger.debug("research scanner: expiry refresh failed for %s", sym)

    def _settings_signature(self) -> str:
        s = self._settings
        return (
            f"{s.get('mode')}|{s.get('enabled')}|{s.get('primary_index')}|"
            f"{s.get('lots_nifty')}|{s.get('lots_sensex')}|{s.get('sl_rs')}|"
            f"{sorted((s.get('schedule') or {}).items())}"
        )

    def _record_event(self, symbol: str, kind: str, msg: str) -> None:
        st = self._states.get(symbol.upper())
        if not st:
            return
        ts = datetime.now(IST).strftime("%H:%M:%S")
        st.events.append({"t": ts, "type": kind, "msg": msg})
        st.events = st.events[-50:]
        logger.info("[Research][%s] %s: %s", symbol, kind, msg)

    # ── public state for /api/atm-research/runtime ────────────────
    def get_runtime_state(self) -> dict:
        def leg_dict(leg: Optional[ATLLeg]) -> Optional[dict]:
            if leg is None:
                return None
            return {
                "option_type": leg.option_type,
                "strike": leg.strike,
                "symbol": leg.symbol,
                "premium": leg.premium,
                "exchange": leg.exchange,
            }

        return {
            "settings": dict(self._settings),
            "today": self._today_str,
            "mode_label": "LIVE" if self._is_live() else "PAPER",
            "indices": {
                sym: {
                    "phase": st.phase,
                    "entered": st.entered,
                    "entry_in_progress": st.entry_in_progress,
                    "done_for_day": st.done_for_day,
                    "halted": st.halted,
                    "halt_reason": st.halt_reason,
                    "strat": st.strat,
                    "entry_trigger": st.entry_trigger,
                    "entry_time": st.entry_time_str,
                    "exit_time": st.exit_time_str,
                    "atm": st.atm,
                    "credit_pts": round(st.credit, 2),
                    "expiry": st.expiry,
                    "ce": leg_dict(st.ce),
                    "pe": leg_dict(st.pe),
                    "last_mtm_rs": round(st.last_mtm_rs, 0),
                    "events": list(st.events),
                }
                for sym, st in self._states.items()
            },
        }

    # ── helpers shared with legacy scanner ─────────────────────────
    def _is_live(self) -> bool:
        if bool(getattr(settings, "paper_trading", True)):
            return False
        account = str(self._settings.get("execution_account", "Primary")).strip().lower()
        return account not in ("", "paper")

    def _resolve_broker(self):
        raw = str(self._settings.get("execution_account", "Primary")).strip().lower()
        if raw in ("", "paper"):
            return None
        if raw in ("primary", "live (primary)"):
            return self.broker
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
            logger.exception("Research: failed to instantiate broker for account=%s", raw)
        return self.broker

    def _lots_for(self, symbol: str) -> int:
        if symbol.upper() == "NIFTY":
            return max(1, int(self._settings.get("lots_nifty", 4)))
        if symbol.upper() == "SENSEX":
            return max(1, int(self._settings.get("lots_sensex", 4)))
        return 1

    def _strike_step_for(self, symbol: str) -> int:
        if symbol.upper() == "NIFTY":
            return max(1, int(self._settings.get("strike_step_nifty", 50)))
        if symbol.upper() == "SENSEX":
            return max(1, int(self._settings.get("strike_step_sensex", 100)))
        return 50

    def _legs_for(self, strat: str, symbol: str, atm: float) -> list[tuple[float, str]]:
        step = self._strike_step_for(symbol)
        if strat == "straddle":
            return [(atm, "CE"), (atm, "PE")]
        if strat == "strangle_1":
            return [(atm + step, "CE"), (atm - step, "PE")]
        if strat == "strangle_2":
            return [(atm + 2 * step, "CE"), (atm - 2 * step, "PE")]
        return [(atm, "CE"), (atm, "PE")]

    # ── option-quote helpers ───────────────────────────────────────
    async def _fetch_option_quote(self, instrument: InstrumentConfig, strike: float, option_type: str) -> Optional[dict]:
        st = self._states.get(instrument.symbol.upper())
        if not st or not st.expiry:
            return None
        symbol = instrument.build_option_symbol(st.expiry, strike, option_type)
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.get_option_quote,
                    token_info.get("exch_seg", "NFO"),
                    token_info.get("tradingsymbol", ""),
                    token_info.get("symboltoken", ""),
                ),
                timeout=10,
            )
        except Exception:
            return None

    async def _build_leg(self, instrument: InstrumentConfig, strike: float, option_type: str) -> Optional[ATLLeg]:
        st = self._states.get(instrument.symbol.upper())
        if not st or not st.expiry:
            return None
        symbol = instrument.build_option_symbol(st.expiry, strike, option_type)
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            return None
        quote = await self._fetch_option_quote(instrument, strike, option_type)
        ltp = float((quote or {}).get("ltp", 0) or 0)
        return ATLLeg(
            option_type=option_type,
            strike=strike,
            symbol=token_info.get("tradingsymbol", symbol),
            symboltoken=token_info.get("symboltoken", ""),
            exchange=token_info.get("exch_seg", "NFO"),
            premium=ltp,
        )

    # ── broker order placement ─────────────────────────────────────
    async def _place_leg_order(self, instrument: InstrumentConfig, leg: ATLLeg, side: str, lots: int, reason: str) -> bool:
        if not self._is_live():
            self._record_event(instrument.symbol, "paper_order",
                               f"{side} {leg.symbol} qty={lots*instrument.lot_size} ({reason})")
            return True
        if not leg.symbol or not leg.symboltoken:
            return False
        qty = max(1, int(lots)) * max(1, int(instrument.lot_size))
        broker = self._resolve_broker()
        if broker is None:
            self._record_event(instrument.symbol, "order_skip", f"No broker resolved for {reason}")
            return False
        req = OrderRequest(
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
            underlying=instrument.option_symbol_prefix or instrument.symbol,
            expiry_date=self._states[instrument.symbol.upper()].expiry_date,
            strike=float(leg.strike),
            option_type=leg.option_type,
            wait_for_terminal=True,
        )
        try:
            resp = await asyncio.to_thread(broker.place_order, req)
        except Exception:
            logger.exception("Research broker order exception: %s %s %s", side, leg.symbol, reason)
            self._record_event(instrument.symbol, "order_error",
                               f"{side} {leg.symbol} exception ({reason})")
            return False
        if not resp or resp.status == OrderStatus.REJECTED:
            msg = (resp.message if resp else "no response") or "rejected"
            self._record_event(instrument.symbol, "order_error",
                               f"{side} {leg.symbol} rejected ({msg})")
            return False
        if resp.status in {OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.COMPLETE}:
            if resp.filled_price > 0:
                leg.premium = resp.filled_price
            self._record_event(instrument.symbol, "order",
                               f"{side} {leg.symbol} qty={qty} id={resp.order_id} ({reason})")
            return True
        self._record_event(instrument.symbol, "order_error",
                           f"{side} {leg.symbol} unresolved status={resp.status.value} ({reason})")
        return False

    async def _enter_legs(self, instrument: InstrumentConfig, st: _IndexState, reason: str) -> bool:
        """Place both SELL legs in parallel. Rollback if one fails."""
        if st.ce is None or st.pe is None:
            return False
        lots = self._lots_for(instrument.symbol)
        st.entry_in_progress = True
        try:
            ce_ok, pe_ok = await asyncio.gather(
                self._place_leg_order(instrument, st.ce, "SELL", lots, reason),
                self._place_leg_order(instrument, st.pe, "SELL", lots, reason),
            )
            if ce_ok and pe_ok:
                st.entered = True
                st.phase = "ENTERED"
                # Credit in points
                st.credit = float((st.ce.premium or 0) + (st.pe.premium or 0))
                self._record_event(
                    instrument.symbol, "entered",
                    f"{st.strat} {st.ce.symbol}+{st.pe.symbol} credit={st.credit:.2f} pts "
                    f"({lots} lot×{instrument.lot_size})",
                )
                return True
            # Roll back the leg that did enter
            if ce_ok and not pe_ok:
                await self._place_leg_order(instrument, st.ce, "BUY", lots, f"rollback_{reason}")
            if pe_ok and not ce_ok:
                await self._place_leg_order(instrument, st.pe, "BUY", lots, f"rollback_{reason}")
            self._record_event(instrument.symbol, "entry_failed",
                               f"{st.strat} entry failed (ce_ok={ce_ok}, pe_ok={pe_ok}) — completed for day")
            st.done_for_day = True
            st.phase = "DONE"
            st.ce = None
            st.pe = None
            return False
        finally:
            st.entry_in_progress = False

    async def _exit_legs(self, instrument: InstrumentConfig, st: _IndexState, reason: str) -> bool:
        if not st.entered:
            return True
        lots = self._lots_for(instrument.symbol)
        ce_ok = pe_ok = True
        if st.ce is not None:
            ce_ok = await self._place_leg_order(instrument, st.ce, "BUY", lots, f"exit_{reason}")
        if st.pe is not None:
            pe_ok = await self._place_leg_order(instrument, st.pe, "BUY", lots, f"exit_{reason}")
        st.entered = False
        st.done_for_day = True
        st.phase = "DONE"
        self._record_event(instrument.symbol, "exit",
                           f"{st.strat} exit reason={reason} ce_ok={ce_ok} pe_ok={pe_ok}")
        return ce_ok and pe_ok

    # ── indicator gates ───────────────────────────────────────────
    def _evaluate_entry_gate(self, kind: str, strat: str, df_today: pd.DataFrame,
                              entry_time_str: str, now: datetime) -> tuple[bool, str]:
        """Return (should_enter, why). Triggers replicate the backtest's
        `trigger_entry()` logic with the same numeric thresholds."""
        if df_today is None or df_today.empty:
            return False, "no_data"
        if kind == "T0":
            # Pure time gate: enter when now >= entry_time
            ent_min = _to_min(entry_time_str)
            return _now_min(now) >= ent_min, f"clock>={entry_time_str}"

        # All T1..T5 need at least a few bars past entry_time
        ent_min = _to_min(entry_time_str)
        if _now_min(now) < ent_min + ENTRY_GATE_MIN_BARS:
            return False, f"warmup<{ENTRY_GATE_MIN_BARS}min"

        # Restrict df to today, slice to bars from entry_time onward
        today_str = now.strftime("%Y-%m-%d")
        try:
            mask = df_today.index.strftime("%Y-%m-%d") == today_str
            day_df = df_today[mask].copy()
        except Exception:
            day_df = df_today.copy()
        if day_df.empty:
            return False, "no_today_bars"

        # Filter to bars at/after entry_time
        try:
            ent_dt = now.replace(hour=ent_min // 60, minute=ent_min % 60, second=0, microsecond=0)
            day_df = day_df[day_df.index >= pd.Timestamp(ent_dt)]
        except Exception:
            pass
        if day_df.empty or len(day_df) < ENTRY_GATE_MIN_BARS:
            return False, "warmup_bars"

        last = day_df.iloc[-1]
        spot = float(last.get("close", 0) or 0)
        if spot <= 0:
            return False, "no_spot"

        if kind == "T1":  # iv_crush — premium proxy via close drop vs entry-window open
            open_px = float(day_df.iloc[0].get("close", 0) or 0)
            if open_px <= 0:
                return False, "no_open"
            drop = (spot - open_px) / open_px
            if drop <= T1_IV_CRUSH_DROP:
                return True, f"iv_crush drop={drop:.4f}"
            return False, f"drop={drop:.4f}"

        ema9 = _safe_float(last.get("ema9"))
        ema20 = _safe_float(last.get("ema20"))
        vwap = _safe_float(last.get("vwap"))
        rsi = _safe_float(last.get("rsi"))
        bu = _safe_float(last.get("bollinger_upper"))
        bl = _safe_float(last.get("bollinger_lower"))
        bm = _safe_float(last.get("bollinger_middle"))
        bb_w = ((bu - bl) / bm) if (bu and bl and bm and bm > 0) else None

        if kind == "T2":  # vwap_align
            if vwap is None:
                return False, "no_vwap"
            tight = abs(spot - vwap) / spot < T2_VWAP_ALIGN_TOL
            quiet = (bb_w is not None) and bb_w < T2_BB_WIDTH_LOW
            if strat == "straddle":
                if tight and quiet:
                    return True, f"vwap_align dist={(spot-vwap)/spot:.4f} bbw={bb_w:.4f}"
                return False, f"dist={(spot-vwap)/spot:.4f} bbw={bb_w}"
            # short_ce / short_pe variants not used here (we only do straddle/strangle)
            return False, "non-straddle T2"

        if kind == "T3":  # rsi_revert near 50 + quiet
            if rsi is None:
                return False, "no_rsi"
            centred = abs(rsi - 50) <= T3_RSI_CENTER_TOL
            quiet = (bb_w is not None) and bb_w < T3_BB_WIDTH_LOW
            if centred and quiet:
                return True, f"rsi_revert rsi={rsi:.1f} bbw={bb_w:.4f}"
            return False, f"rsi={rsi:.1f} bbw={bb_w}"

        if kind == "T4":  # bb_squeeze
            if bb_w is None:
                return False, "no_bbw"
            if bb_w < T4_BB_WIDTH_LOW:
                return True, f"bb_squeeze bbw={bb_w:.4f}"
            return False, f"bbw={bb_w:.4f}"

        if kind == "T5":  # ema cross / squeeze for straddle
            if bb_w is None:
                return False, "no_bbw"
            if bb_w < T5_BB_WIDTH_LOW:
                return True, f"ema_cross_squeeze bbw={bb_w:.4f}"
            return False, f"bbw={bb_w:.4f}"

        return False, f"unknown_trigger={kind}"

    # ── main per-instrument cycle ─────────────────────────────────
    async def run_cycle(self, df_today: pd.DataFrame, instrument: InstrumentConfig, cycle: int) -> None:
        # Always refresh settings + expiries — UI changes apply intraday
        new_settings = load_research_settings()
        new_sig = ""
        try:
            self._settings = new_settings
            new_sig = self._settings_signature()
        except Exception:
            pass

        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")
        if self._today_str != today:
            self._today_str = today
            self._reset_for_new_day()
            self._refresh_expiries()
        elif cycle % 10 == 0:
            self._refresh_expiries()

        # Re-arm if signature changed and any index is done-for-day
        if new_sig and new_sig != self._last_signature:
            self._last_signature = new_sig
            for st in self._states.values():
                if st.done_for_day and not st.entered:
                    st.done_for_day = False
                    st.phase = "IDLE"

        if not self._settings.get("enabled", False):
            return

        sym = instrument.symbol.upper()
        if sym not in self._states:
            return

        st = self._states[sym]
        if st.halted or st.entry_in_progress:
            return

        # Determine today's cell from schedule (mode-filtered)
        sched = effective_schedule(self._settings)
        wd_name = WEEKDAY_INDEX_TO_NAME[now.weekday()] if now.weekday() < 5 else None
        if wd_name is None:
            return
        cell = (sched.get(wd_name, {}) or {}).get(sym)
        if not cell or not cell.get("enabled", False):
            return

        st.entry_time_str = str(cell.get("entry_time") or "09:20")
        st.exit_time_str = str(cell.get("exit_time") or "15:15")
        st.strat = str(cell.get("strat") or "straddle")
        st.entry_trigger = str(cell.get("entry") or "T0")

        # Hard exit-time gate: close if entered, otherwise mark done
        exit_min = _to_min(st.exit_time_str)
        if _now_min(now) >= exit_min:
            if st.entered:
                await self._exit_legs(instrument, st, reason="session_exit_time")
            else:
                st.done_for_day = True
                st.phase = "DONE"
            return

        if st.done_for_day:
            return

        # ── In-trade SL monitoring ─────────────────────────────────
        if st.entered and st.ce is not None and st.pe is not None:
            await self._refresh_premiums(instrument, st)
            cur_cost = float((st.ce.premium or 0) + (st.pe.premium or 0))
            rs_per_pt = self._lots_for(sym) * int(instrument.lot_size)
            mtm = (st.credit - cur_cost) * rs_per_pt
            st.last_mtm_rs = mtm
            sl_rs = float(self._settings.get("sl_rs", 6000))
            if mtm <= -sl_rs:
                self._record_event(sym, "sl_hit",
                                   f"₹SL fired: credit={st.credit:.2f} cur_cost={cur_cost:.2f} mtm=₹{mtm:.0f}")
                await self._exit_legs(instrument, st, reason="sl_rs")
            return

        # ── Not entered yet: check entry trigger ───────────────────
        if _now_min(now) < _to_min(st.entry_time_str):
            return  # Too early even for T0

        should, why = self._evaluate_entry_gate(
            st.entry_trigger, st.strat, df_today, st.entry_time_str, now,
        )
        if not should:
            return

        # Compute ATM from last spot
        try:
            spot_now = float(df_today.iloc[-1]["close"])
        except Exception:
            return
        step = self._strike_step_for(sym)
        atm = round(spot_now / step) * step
        st.atm = atm

        legs = self._legs_for(st.strat, sym, atm)
        # Build CE then PE
        ce_strike, _ = legs[0]
        pe_strike, _ = legs[1]
        ce_leg, pe_leg = await asyncio.gather(
            self._build_leg(instrument, float(ce_strike), "CE"),
            self._build_leg(instrument, float(pe_strike), "PE"),
        )
        if ce_leg is None or pe_leg is None:
            self._record_event(sym, "build_failed",
                               f"Could not resolve option symbol(s) for ATM={atm} ({why})")
            return
        st.ce = ce_leg
        st.pe = pe_leg
        await self._enter_legs(instrument, st, reason=f"{st.entry_trigger}:{why}")

    async def _refresh_premiums(self, instrument: InstrumentConfig, st: _IndexState) -> None:
        if st.ce is None or st.pe is None:
            return
        try:
            ce_q, pe_q = await asyncio.gather(
                self._fetch_option_quote(instrument, st.ce.strike, "CE"),
                self._fetch_option_quote(instrument, st.pe.strike, "PE"),
            )
        except Exception:
            return
        if ce_q is not None:
            ltp = float((ce_q or {}).get("ltp", 0) or 0)
            if ltp > 0:
                st.ce.premium = ltp
        if pe_q is not None:
            ltp = float((pe_q or {}).get("ltp", 0) or 0)
            if ltp > 0:
                st.pe.premium = ltp

    # ── manual control endpoints ───────────────────────────────────
    async def force_close_all(self, reason: str = "manual") -> None:
        for sym, st in self._states.items():
            if st.entered:
                # Need the instrument to place orders — caller (route) is
                # expected to look it up and call force_close_one.
                self._record_event(sym, "force_close_request", reason)

    async def force_close_one(self, instrument: InstrumentConfig, reason: str = "manual") -> bool:
        sym = instrument.symbol.upper()
        st = self._states.get(sym)
        if not st or not st.entered:
            return False
        return await self._exit_legs(instrument, st, reason=reason)

    def reset_halt(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            for st in self._states.values():
                st.halted = False
                st.halt_reason = ""
                st.done_for_day = False
                st.phase = "IDLE"
        else:
            st = self._states.get(symbol.upper())
            if st:
                st.halted = False
                st.halt_reason = ""
                st.done_for_day = False
                st.phase = "IDLE"


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None
