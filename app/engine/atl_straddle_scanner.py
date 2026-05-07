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
    done_for_day: bool = False


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

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._settings = load_atl_settings()
        self._state = ATLState()
        self._events = []

    def get_runtime_state(self) -> dict:
        st = self._state
        return {
            "enabled": bool(self._settings.get("enabled", False)),
            "live_mode": not settings.paper_trading,
            "strategy_type": self._settings.get("strategy_type", "ATM_STRADDLE"),
            "phase": st.phase,
            "in_trade": self.is_in_trade(),
            "done_for_day": st.done_for_day,
            "index": self._settings.get("index", "NIFTY"),
            "trading_day": self._settings.get("trading_day", "Daily"),
            "expiry": self._expiry,
            "entry_time": self._settings.get("entry_time", "09:20"),
            "exit_time": self._settings.get("exit_time", "15:15"),
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
            return

        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")
        if self._today_str != today:
            self.reset_daily()

        # Refresh settings periodically so UI changes apply intraday.
        if cycle % 5 == 0:
            self._settings = load_atl_settings()

        if not self._settings.get("enabled", False):
            return

        if instrument.symbol != self._settings.get("index", "NIFTY"):
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
                return

        try:
            entry_h, entry_m = [int(x) for x in str(self._settings.get("entry_time", "09:20")).split(":")]
            exit_h, exit_m = [int(x) for x in str(self._settings.get("exit_time", "15:15")).split(":")]
        except Exception:
            entry_h, entry_m = 9, 20
            exit_h, exit_m = 15, 15

        if now.time() >= datetime(now.year, now.month, now.day, exit_h, exit_m, tzinfo=IST).time() and self.is_in_trade():
            await self.force_close(df_today, instrument)
            return

        if self._state.done_for_day:
            return

        if peer_in_trade and not self.is_in_trade():
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
                await self.force_close(df_today, instrument)
                self._state.done_for_day = True
                return

        # Initial entry
        if not self._state.entered:
            if now.hour < entry_h or (now.hour == entry_h and now.minute < entry_m):
                return
            rounded = round(spot / interval) * interval
            ce_strike = rounded + offset
            pe_strike = rounded - offset
            ce_q = await self._fetch_option_quote(instrument, ce_strike, "CE")
            pe_q = await self._fetch_option_quote(instrument, pe_strike, "PE")
            if not ce_q or not pe_q:
                return
            ce_leg = await self._build_leg(instrument, ce_strike, "CE", fallback_quote=ce_q)
            pe_leg = await self._build_leg(instrument, pe_strike, "PE", fallback_quote=pe_q)
            if not ce_leg or not pe_leg:
                return

            if not await self._execute_entry_legs(instrument, ce_leg, pe_leg, reason="initial_entry"):
                return

            self._state.ce = ce_leg
            self._state.pe = pe_leg
            self._state.entered = True
            self._state.phase = "STRANGLE"
            self._state.ref_spot = spot
            if self._settings.get("hedge_enabled", False):
                await self._ensure_hedges(instrument, rounded)
            self._record_event("entry", f"STRANGLE CE {int(ce_strike)} PE {int(pe_strike)} @ spot {spot:.2f}")
            await self.alert_manager.telegram.send(
                f"⚡ ATL ENTRY ({'LIVE' if not settings.paper_trading else 'PAPER'})\n"
                f"Index: {instrument.symbol}\n"
                f"Spot: {spot:.2f}\n"
                f"SELL CE {int(ce_strike)} @ ₹{self._state.ce.premium:.2f}\n"
                f"SELL PE {int(pe_strike)} @ ₹{self._state.pe.premium:.2f}"
            )
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
                await self._convert_to_straddle(instrument, new_strike, spot, reform=True)

    async def _convert_to_straddle(self, instrument: InstrumentConfig, strike: float, spot: float, reform: bool = False) -> None:
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
            f"⚡ ATL {tag} STRADDLE ({'LIVE' if not settings.paper_trading else 'PAPER'})\n"
            f"Spot: {spot:.2f}\n"
            f"Strike: {int(strike)}\n"
            f"CE ₹{self._state.ce.premium:.2f} + PE ₹{self._state.pe.premium:.2f}\n"
            f"SL points: {self._state.straddle_sl_points:.2f}"
        )

    async def force_close(self, df_today: pd.DataFrame, instrument: InstrumentConfig) -> None:
        if not self.is_in_trade():
            return
        spot = float(df_today.iloc[-1]["close"]) if df_today is not None and not df_today.empty else 0.0
        # Close active short legs first.
        await self._close_current_shorts(instrument, reason="eod_force_close")
        if self._settings.get("hedge_mode", "none") != "none" and (self._state.hedge_ce or self._state.hedge_pe):
            await self._close_hedges(instrument, reason="eod_force_close")
            await self.alert_manager.telegram.send(
                f"🛡️ ATL Hedge Close ({'LIVE' if not settings.paper_trading else 'PAPER'})\n"
                f"BUY hedges exit: CE {int(self._state.hedge_ce.strike) if self._state.hedge_ce else '-'} / "
                f"PE {int(self._state.hedge_pe.strike) if self._state.hedge_pe else '-'}"
            )
        self._state.done_for_day = True
        self._state.phase = "IDLE"
        self._state.hedge_ce = None
        self._state.hedge_pe = None
        self._record_event("force_close", f"EOD close @ spot {spot:.2f}")
        await self.alert_manager.telegram.send(
            f"🔔 ATL EOD Force Close\nIndex: {instrument.symbol}\nSpot: {spot:.2f}"
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
            ce_q = await self._fetch_option_quote(instrument, ce_strike, "CE")
            pe_q = await self._fetch_option_quote(instrument, pe_strike, "PE")
            ce_leg = await self._build_leg(instrument, ce_strike, "CE", fallback_quote=ce_q)
            pe_leg = await self._build_leg(instrument, pe_strike, "PE", fallback_quote=pe_q)
        else:
            target_premium = max(1.0, float(self._settings.get("hedge_premium", 3) or 3))
            ce_leg = await self._find_hedge_leg(instrument, ref_strike, "CE", target_premium, interval)
            pe_leg = await self._find_hedge_leg(instrument, ref_strike, "PE", target_premium, interval)

        if ce_leg:
            self._state.hedge_ce = ce_leg
        if pe_leg:
            self._state.hedge_pe = pe_leg
        if self._state.hedge_ce or self._state.hedge_pe:
            hedge_lots = int(self._settings.get("hedge_lots", 0) or 0)
            short_lots = int(self._settings.get("lots", 1) or 1)
            lots = hedge_lots if hedge_lots > 0 else short_lots
            if self._state.hedge_ce and not await self._place_leg_order(instrument, self._state.hedge_ce, "BUY", lots, "hedge_entry"):
                self._state.hedge_ce = None
            if self._state.hedge_pe and not await self._place_leg_order(instrument, self._state.hedge_pe, "BUY", lots, "hedge_entry"):
                self._state.hedge_pe = None
            ce_strike = int(self._state.hedge_ce.strike) if self._state.hedge_ce else "-"
            ce_px = f"₹{self._state.hedge_ce.premium:.2f}" if self._state.hedge_ce else "-"
            pe_strike = int(self._state.hedge_pe.strike) if self._state.hedge_pe else "-"
            pe_px = f"₹{self._state.hedge_pe.premium:.2f}" if self._state.hedge_pe else "-"
            self._record_event("hedge", f"Hedge CE {ce_strike} / PE {pe_strike}")
            await self.alert_manager.telegram.send(
                f"🛡️ ATL Hedge Entry ({'LIVE' if not settings.paper_trading else 'PAPER'})\n"
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
        best: Optional[ATLLeg] = None
        best_diff: Optional[float] = None
        for step in range(1, 21):
            strike = ref_strike + (interval * step if option_type == "CE" else -interval * step)
            q = await self._fetch_option_quote(instrument, strike, option_type)
            if not q:
                continue
            ltp = float(q.get("ltp", 0) or 0)
            if ltp <= 0:
                continue
            diff = abs(ltp - target_premium)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                candidate = await self._build_leg(instrument, strike, option_type, fallback_quote=q)
                if candidate:
                    best = candidate
            if ltp <= target_premium:
                break
        return best

    def _record_event(self, event_type: str, message: str) -> None:
        self._events.append({
            "time": datetime.now(IST).strftime("%H:%M:%S"),
            "event": event_type,
            "message": message,
            "mode": "live" if not settings.paper_trading else "paper",
        })
        if len(self._events) > 200:
            self._events = self._events[-200:]

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
    ) -> bool:
        """Place an ATL leg via the configured broker (Kite/Angel) abstraction.

        Symbol-string differences between brokers are absorbed inside each
        broker adapter — KiteBroker translates Angel-format symbols to Kite
        tradingsymbols on the fly via KiteClient.resolve_from_angel_symbol.
        """
        request = OrderRequest(
            instrument=instrument,
            trading_symbol=leg.symbol,
            symbol_token=leg.symboltoken,
            exchange=leg.exchange or "NFO",
            side=OrderSide(side),
            order_type=OrderType.MARKET,
            product_type=ProductType.INTRADAY,
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
        )
        try:
            resp = await asyncio.to_thread(self.broker.place_order, request)
        except Exception:
            logger.exception(
                "ATL broker order exception (%s): %s %s %s",
                self.broker.name if self.broker else "?", side, leg.symbol, reason,
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
        if settings.paper_trading:
            return True
        if not leg.symbol or not leg.symboltoken:
            return False
        qty = max(1, int(lots)) * max(1, int(instrument.lot_size))

        # Always route through the broker abstraction. The broker adapter
        # (AngelOne or Kite) handles SmartAPI / KiteConnect specifics and
        # the symbol-format translation. Falls back to legacy SmartAPI path
        # only if no broker is configured (defensive — should not happen).
        if self.broker is not None:
            return await self._place_leg_order_via_broker(
                instrument, leg, side, qty, reason
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
                "producttype": "INTRADAY",
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
        ce_ok = await self._place_leg_order(instrument, ce_leg, "SELL", lots, reason)
        pe_ok = await self._place_leg_order(instrument, pe_leg, "SELL", lots, reason)
        if ce_ok and pe_ok:
            return True
        # Best-effort rollback if one leg entered but the other failed.
        if ce_ok and not pe_ok:
            await self._place_leg_order(instrument, ce_leg, "BUY", lots, f"rollback_{reason}")
        if pe_ok and not ce_ok:
            await self._place_leg_order(instrument, pe_leg, "BUY", lots, f"rollback_{reason}")
        return False

    async def _close_current_shorts(self, instrument: InstrumentConfig, reason: str) -> bool:
        lots = int(self._settings.get("lots", 1) or 1)
        ok = True
        if self._state.ce:
            ok = await self._place_leg_order(instrument, self._state.ce, "BUY", lots, reason) and ok
        if self._state.pe:
            ok = await self._place_leg_order(instrument, self._state.pe, "BUY", lots, reason) and ok
        return ok

    async def _close_hedges(self, instrument: InstrumentConfig, reason: str) -> bool:
        hedge_lots = int(self._settings.get("hedge_lots", 0) or 0)
        short_lots = int(self._settings.get("lots", 1) or 1)
        lots = hedge_lots if hedge_lots > 0 else short_lots
        ok = True
        if self._state.hedge_ce:
            ok = await self._place_leg_order(instrument, self._state.hedge_ce, "SELL", lots, reason) and ok
        if self._state.hedge_pe:
            ok = await self._place_leg_order(instrument, self._state.hedge_pe, "SELL", lots, reason) and ok
        return ok

    async def _switch_shorts(self, instrument: InstrumentConfig, new_ce: ATLLeg, new_pe: ATLLeg, reason: str) -> bool:
        old_ce = self._state.ce
        old_pe = self._state.pe
        if not await self._close_current_shorts(instrument, f"close_{reason}"):
            await self.alert_manager.telegram.send(f"⚠️ ATL {reason}: failed to close current short legs")
            return False
        if await self._execute_entry_legs(instrument, new_ce, new_pe, f"open_{reason}"):
            return True
        # Best-effort rollback: restore prior short if new placement failed.
        await self.alert_manager.telegram.send(f"⚠️ ATL {reason}: failed opening new legs, attempting rollback")
        if old_ce and old_pe:
            await self._execute_entry_legs(instrument, old_ce, old_pe, f"rollback_{reason}")
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
