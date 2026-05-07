"""PDH/PDL Breakout Scanner — Previous Day High/Low breakout (PAPER-ONLY).

Strategy summary (backtested on 131 NIFTY days, 2025-10-13 → 2026-04-24):
  - Setup:    every trading day (no NR/gap filter).
  - Bars:     5-min OHLC (resampled from 1-min).
  - Entry:    first 5-min bar that CLOSES above prev-day HIGH (CE)
              or CLOSES below prev-day LOW (PE), before 14:30 cutoff.
  - Stop:     15 pts adverse (spot)
  - Target:   60 pts favorable (spot)  → R:R = 1:4
  - EOD:      force exit at 15:20

Honest backtest result (5-min bars, intrabar high/low for SL/TP):
  N=121 trades, Spot WR ~64%, Spot PF 1.45, Spot total +600 pts.
  Option PF 4.86, Option WR 72.7%, Option expectancy +12.3 pts/trade,
  Option total +1485 pts over 121 trades.

This scanner is OBSERVE-ONLY: it sends Telegram alerts but does NOT
record trades to the paper-trader DB. It also respects a 4-way mutex
with Config-P / Move-Det / NR5 — it will not enter while any other
scanner holds an open trade, and will block them while it itself is
in a trade.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional

import pandas as pd
import pytz
from sqlalchemy import select

from app.alerts.alert_manager import AlertManager
from app.core.config import settings
from app.core.instruments import InstrumentConfig
from app.core.models import (
    AlertItem,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
)
from app.data.angelone_client import AngelOneClient
from app.db.models import AsyncSessionLocal, IndexCandle
from app.engine import scanner_exec_settings as exec_settings

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── PDH/PDL Parameters (LOCKED — matches backtest W1 S15 T60) ─────────
STOP_PTS = 15.0
TARGET_PTS = 60.0
BAR_MINUTES = 5
ENTRY_CUTOFF = dtime(14, 30)
EOD_FORCE_CLOSE = dtime(15, 20)


def _resample_to_bar_minutes(df: pd.DataFrame, minutes: int = BAR_MINUTES) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    rule = f"{int(minutes)}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(how="any")
    return out


class PDHPDLTrade:
    """Tracks a single PDH/PDL paper trade from entry to exit."""

    def __init__(
        self,
        trade_date: str,
        side: str,              # 'CE' or 'PE'
        entry_time: str,
        entry_spot: float,
        prev_high: float,
        prev_low: float,
        stop_level: float,
        target_level: float,
        option_symbol: str = "",
        option_entry_price: float = 0.0,
        strike_price: float = 0.0,
    ):
        self.trade_date = trade_date
        self.side = side
        self.entry_time = entry_time
        self.entry_spot = entry_spot
        self.prev_high = prev_high
        self.prev_low = prev_low
        self.stop_level = stop_level
        self.target_level = target_level
        self.option_symbol = option_symbol
        self.option_entry_price = option_entry_price
        self.strike_price = strike_price
        # Exit tracking
        self.exited = False
        self.exit_time: str = ""
        self.exit_spot: float = 0.0
        self.exit_reason: str = ""
        self.option_exit_price: float = 0.0
        # Live execution tracking
        self.symboltoken: str = ""
        self.exchange: str = "NFO"
        self.lot_size: int = 0
        self.lots: int = 0
        self.live_executed: bool = False
        self.entry_order_id: str = ""
        self.exit_order_id: str = ""


class PDHPDLBreakoutScanner:
    """PDH/PDL breakout scanner — paper-only, runs each minute during market hours.

    Lifecycle:
      1. At 09:14 / first cycle: query DB for prev day's HIGH and LOW.
      2. Each cycle (until 14:30): on each new 5-min bar close, check whether
         it crossed prev_high (CE) or prev_low (PE). First trigger -> entry.
      3. While in trade: each cycle check stop / target / EOD.
      4. At 15:20: force exit if still open.

    Mutex: orchestrator passes ``peer_in_trade=True`` whenever any other
    scanner currently holds an open trade. The scanner skips new entries
    in that state. It also exposes ``is_in_trade()`` so peers can do the same.
    """

    def __init__(
        self,
        client: AngelOneClient,
        alert_manager: AlertManager,
        broker: Optional[object] = None,
    ):
        self.client = client
        self.alert_manager = alert_manager
        self.broker = broker

        # Daily state
        self._today_str: str = ""
        self._setup_checked = False
        self._is_tradeable_day = False
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._signal_found_today = False
        self._active_trade: Optional[PDHPDLTrade] = None
        self._expiry: str = ""
        self._expiry_date: Optional[date] = None

    # ── Public API used by orchestrator ─────────────────────────────

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._setup_checked = False
        self._is_tradeable_day = False
        self._prev_high = None
        self._prev_low = None
        self._signal_found_today = False
        self._active_trade = None

    def set_expiry(self, expiry: str, expiry_date: Optional[date] = None) -> None:
        self._expiry = expiry
        self._expiry_date = expiry_date

    def is_in_trade(self) -> bool:
        return self._active_trade is not None and not self._active_trade.exited

    async def run_cycle(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        cycle: int,
        peer_in_trade: bool = False,
    ) -> None:
        if df_today is None or df_today.empty:
            return
        if isinstance(df_today.index, pd.DatetimeIndex):
            df_today = df_today.between_time("09:15", "15:30")
            if df_today.empty:
                return
            df_today = _resample_to_bar_minutes(df_today, BAR_MINUTES)
            if df_today.empty:
                return

        now = datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")
        if self._today_str != today_str:
            if self._today_str:
                logger.info(
                    "[PDHPDL] Day rollover: %s → %s. Resetting daily state.",
                    self._today_str, today_str,
                )
            self.reset_daily()
            self._today_str = today_str

        # ── If we have an active trade, manage it first ─────────────
        if self._active_trade and not self._active_trade.exited:
            await self._check_exit(df_today, instrument, now)
            return

        # ── Mutex: skip if any peer scanner is in a trade ───────────
        if peer_in_trade:
            return

        # ── Already done for today ──────────────────────────────────
        if self._signal_found_today:
            return

        # ── Setup check (one-shot, on first cycle of the day) ───────
        if not self._setup_checked:
            await self._check_setup(now)
            self._setup_checked = True

        if not self._is_tradeable_day:
            return

        # ── Past entry cutoff? Stop scanning ────────────────────────
        if now.time() > ENTRY_CUTOFF:
            self._signal_found_today = True
            logger.info("[PDHPDL] %s — past entry cutoff %s, no breakout today",
                        self._today_str, ENTRY_CUTOFF)
            await self.alert_manager.telegram.send(
                f"⏰ PDH/PDL — Entry Cutoff Passed\n"
                f"Date: {self._today_str}\n"
                f"No breakout by {ENTRY_CUTOFF.strftime('%H:%M')}. Done for today."
            )
            return

        # ── Look for breakout: 5-min bar that CLOSED above PDH or below PDL ─
        last = df_today.iloc[-1]
        last_close = float(last["close"])
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_ts = last.name if isinstance(df_today.index, pd.DatetimeIndex) else last.get("timestamp")

        side: Optional[str] = None
        entry_spot = 0.0
        if (
            self._prev_high is not None
            and last_high >= self._prev_high
            and last_close >= self._prev_high
        ):
            side = "CE"
            entry_spot = last_close
        elif (
            self._prev_low is not None
            and last_low <= self._prev_low
            and last_close <= self._prev_low
        ):
            side = "PE"
            entry_spot = last_close
        if side is None:
            return

        # ── Build entry ─────────────────────────────────────────────
        entry_time = last_ts.strftime("%H:%M") if hasattr(last_ts, "strftime") else now.strftime("%H:%M")
        stop_level = entry_spot - STOP_PTS if side == "CE" else entry_spot + STOP_PTS
        target_level = entry_spot + TARGET_PTS if side == "CE" else entry_spot - TARGET_PTS

        # ATM strike + option quote
        strike = instrument.nearest_strike(entry_spot, side)
        option_symbol = ""
        option_ltp = 0.0
        option_bid = 0.0
        option_ask = 0.0
        spread_pct = 0.0
        if self._expiry:
            option_symbol = instrument.build_option_symbol(self._expiry, strike, side)
            quote = await self._fetch_option_quote(instrument, strike, side)
            if quote:
                option_ltp = float(quote.get("ltp", 0.0) or 0.0)
                option_bid = float(quote.get("best_bid", 0.0) or 0.0)
                option_ask = float(quote.get("best_ask", 0.0) or 0.0)
                spread_pct = float(quote.get("spread_pct", 0.0) or 0.0)

        self._active_trade = PDHPDLTrade(
            trade_date=self._today_str,
            side=side,
            entry_time=entry_time,
            entry_spot=entry_spot,
            prev_high=self._prev_high or 0.0,
            prev_low=self._prev_low or 0.0,
            stop_level=stop_level,
            target_level=target_level,
            option_symbol=option_symbol,
            option_entry_price=option_ltp,
            strike_price=strike,
        )
        self._signal_found_today = True

        # Resolve symbol token + lot size for live execution
        symboltoken = ""
        exchange = "NFO"
        if option_symbol:
            token_info = self.client._search_symbol(option_symbol)
            if token_info:
                symboltoken = token_info.get("token", "")
                exchange = token_info.get("exch_seg", "NFO")
        self._active_trade.symboltoken = symboltoken
        self._active_trade.exchange = exchange
        self._active_trade.lot_size = int(instrument.lot_size)

        # Decide whether to place a live order
        cfg = exec_settings.load("pdh_pdl")
        live_exec = (
            bool(cfg.get("live_execution"))
            and not settings.paper_trading
            and self.broker is not None
            and option_ltp > 0
            and bool(symboltoken)
        )
        order_status = "OBSERVE ONLY — paper trade"
        if live_exec:
            lots = max(1, int(cfg.get("manual_lots", 1) or 1))
            self._active_trade.lots = lots
            qty = lots * int(instrument.lot_size)
            ok, oid = await self._place_option_order(
                instrument, self._active_trade, "BUY", qty, tag="entry",
            )
            self._active_trade.entry_order_id = oid
            self._active_trade.live_executed = bool(ok)
            order_status = (
                f"LIVE ORDER PLACED — id={oid}" if ok else "LIVE ORDER FAILED"
            )

        expiry_display = self._expiry or "N/A"
        days_to_expiry = (
            (self._expiry_date - now.date()).days if self._expiry_date else "?"
        )
        msg = (
            f"⚡ PDH/PDL BREAKOUT — {side} SIGNAL (PAPER)\n"
            f"{'=' * 36}\n"
            f"\n"
            f"📅 Date: {self._today_str}\n"
            f"⏰ Entry: {entry_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Entry: {entry_spot:.2f}\n"
            f"  Stop:  {stop_level:.2f}  ({-STOP_PTS if side == 'CE' else STOP_PTS:+.0f} pts)\n"
            f"  Target:{target_level:.2f}  ({TARGET_PTS if side == 'CE' else -TARGET_PTS:+.0f} pts)\n"
            f"  R:R    1 : {TARGET_PTS / STOP_PTS:.1f}\n"
            f"\n"
            f"🧮 SETUP\n"
            f"  Prev day H: {self._prev_high:.2f}\n"
            f"  Prev day L: {self._prev_low:.2f}\n"
            f"  Bar close broke: {'PDH' if side == 'CE' else 'PDL'}\n"
            f"\n"
            f"🎯 OPTION ({side})\n"
            f"  Strike: {int(strike)} {side}\n"
            f"  Symbol: {option_symbol}\n"
            f"  Expiry: {expiry_display} ({days_to_expiry}d)\n"
            f"  LTP: ₹{option_ltp:.2f}\n"
            f"  Bid: ₹{option_bid:.2f} | Ask: ₹{option_ask:.2f}\n"
            f"  Spread: {spread_pct:.1f} %\n"
            f"\n"
            f"📋 EXIT RULES\n"
            f"  1. Spot stop: {stop_level:.2f}\n"
            f"  2. Spot target: {target_level:.2f}\n"
            f"  3. EOD: 15:20 force close\n"
            f"\n"
            f"🛒 {order_status}"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[PDHPDL] ENTRY %s @ spot=%.2f stop=%.2f tgt=%.2f opt=%s ₹%.2f",
            side, entry_spot, stop_level, target_level, option_symbol, option_ltp,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=f"PDH/PDL BREAKOUT {side} — NIFTY {int(strike)} {side}",
            message=msg,
            timestamp=now,
        )
        await self.alert_manager.record(alert)

    # ── Setup detection ─────────────────────────────────────────────

    async def _check_setup(self, now: datetime) -> None:
        """Query DB for previous trading day's HIGH and LOW."""
        try:
            cutoff = (now.date() - timedelta(days=10)).strftime("%Y-%m-%d")
            today_str = now.strftime("%Y-%m-%d")
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(IndexCandle.date, IndexCandle.high, IndexCandle.low)
                    .where(IndexCandle.instrument == "NIFTY")
                    .where(IndexCandle.date >= cutoff)
                    .where(IndexCandle.date < today_str)
                    .order_by(IndexCandle.date, IndexCandle.timestamp)
                )
                rows = (await session.execute(stmt)).all()
        except Exception:
            logger.exception("[PDHPDL] DB lookup for prev daily H/L failed")
            return

        if not rows:
            logger.warning("[PDHPDL] No prior daily candles found in DB")
            return

        per_day: dict[str, list[tuple[float, float]]] = {}
        for d, h, l in rows:
            per_day.setdefault(d, []).append((float(h), float(l)))
        if not per_day:
            return

        last_day = sorted(per_day.keys())[-1]
        highs = [r[0] for r in per_day[last_day]]
        lows = [r[1] for r in per_day[last_day]]
        self._prev_high = max(highs)
        self._prev_low = min(lows)
        self._is_tradeable_day = True

        logger.info(
            "[PDHPDL] %s TRADEABLE: prev=%s prev_H=%.2f prev_L=%.2f",
            self._today_str, last_day, self._prev_high, self._prev_low,
        )
        await self.alert_manager.telegram.send(
            f"📊 PDH/PDL — Day Tradeable ✅\n"
            f"Date: {self._today_str}\n"
            f"Prev day ({last_day}):\n"
            f"  H: {self._prev_high:.2f}\n"
            f"  L: {self._prev_low:.2f}\n"
            f"Watching for 5-min close ABOVE PDH (CE) or BELOW PDL (PE)\n"
            f"until {ENTRY_CUTOFF.strftime('%H:%M')}.\n"
            f"Stop {STOP_PTS:.0f} · Target {TARGET_PTS:.0f} (R:R 1:{TARGET_PTS/STOP_PTS:.1f})"
        )

    # ── Exit management ─────────────────────────────────────────────

    async def _check_exit(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        now: datetime,
    ) -> None:
        trade = self._active_trade
        if trade is None or trade.exited:
            return
        if df_today is None or df_today.empty:
            return

        last = df_today.iloc[-1]
        current_high = float(last["high"])
        current_low = float(last["low"])
        current_close = float(last["close"])

        exit_reason = ""
        exit_spot = current_close

        if trade.side == "CE":
            if current_low <= trade.stop_level:
                exit_reason = "stop"
                exit_spot = trade.stop_level
            elif current_high >= trade.target_level:
                exit_reason = "target"
                exit_spot = trade.target_level
        else:
            if current_high >= trade.stop_level:
                exit_reason = "stop"
                exit_spot = trade.stop_level
            elif current_low <= trade.target_level:
                exit_reason = "target"
                exit_spot = trade.target_level

        if not exit_reason and now.time() >= EOD_FORCE_CLOSE:
            exit_reason = "eod_close"
            exit_spot = current_close

        if not exit_reason:
            return

        trade.exited = True
        trade.exit_time = now.strftime("%H:%M")
        trade.exit_spot = exit_spot
        trade.exit_reason = exit_reason

        spot_pnl = (exit_spot - trade.entry_spot) if trade.side == "CE" else (trade.entry_spot - exit_spot)
        is_win = spot_pnl > 0

        # Live option quote at exit
        option_exit_ltp = 0.0
        if trade.option_symbol and self._expiry:
            quote = await self._fetch_option_quote(instrument, trade.strike_price, trade.side)
            if quote:
                option_exit_ltp = float(quote.get("ltp", 0.0) or 0.0)
                trade.option_exit_price = option_exit_ltp

        actual_option_pnl_str = ""
        if trade.option_entry_price > 0 and option_exit_ltp > 0:
            opt_pts = option_exit_ltp - trade.option_entry_price
            opt_pct = opt_pts / trade.option_entry_price * 100
            actual_option_pnl_str = (
                f"\n  Actual option PnL: {opt_pts:+.2f} pts "
                f"({opt_pct:+.1f}%)  ₹{trade.option_entry_price:.2f} → ₹{option_exit_ltp:.2f}"
            )

        emoji = "✅" if is_win else "❌"
        # Place SELL exit order if we placed a live BUY entry
        exit_order_status = "OBSERVE ONLY — paper trade"
        if trade.live_executed and trade.lots > 0:
            qty = trade.lots * int(instrument.lot_size)
            ok, oid = await self._place_option_order(
                instrument, trade, "SELL", qty, tag=f"exit_{exit_reason}",
            )
            trade.exit_order_id = oid
            exit_order_status = (
                f"LIVE EXIT PLACED — id={oid}" if ok else "LIVE EXIT FAILED"
            )
        msg = (
            f"{emoji} PDH/PDL — EXIT {'WIN' if is_win else 'LOSS'} (PAPER)\n"
            f"{'=' * 36}\n"
            f"\n"
            f"📅 Date: {trade.trade_date}\n"
            f"⏰ Entry: {trade.entry_time} → Exit: {trade.exit_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Side: {trade.side}\n"
            f"  Entry:  {trade.entry_spot:.2f}\n"
            f"  Exit:   {exit_spot:.2f}\n"
            f"  PnL:    {spot_pnl:+.2f} pts\n"
            f"\n"
            f"🎯 OPTION ({trade.side})\n"
            f"  {trade.option_symbol}\n"
            f"  Entry LTP: ₹{trade.option_entry_price:.2f}\n"
            f"  Exit  LTP: ₹{option_exit_ltp:.2f}"
            f"{actual_option_pnl_str}\n"
            f"\n"
            f"📋 Exit Reason: {exit_reason}\n"
            f"\n"
            f"🛒 {exit_order_status}"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[PDHPDL] EXIT %s %s @ spot=%.2f pnl=%+.2f pts (reason=%s)",
            "WIN" if is_win else "LOSS", trade.side, exit_spot, spot_pnl, exit_reason,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=f"PDH/PDL {'WIN' if is_win else 'LOSS'} — {exit_reason}",
            message=msg,
            timestamp=now,
        )
        await self.alert_manager.record(alert)

    async def force_close(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
    ) -> None:
        if self._active_trade and not self._active_trade.exited:
            now = datetime.now(IST)
            await self._check_exit(df_today, instrument, now)
            if self._active_trade and not self._active_trade.exited:
                self._active_trade.exited = True
                self._active_trade.exit_reason = "eod_force"
                if df_today is not None and not df_today.empty:
                    self._active_trade.exit_spot = float(df_today.iloc[-1]["close"])
                self._active_trade.exit_time = now.strftime("%H:%M")
                await self.alert_manager.telegram.send(
                    f"🔔 PDH/PDL — EOD Force Close\n"
                    f"Exit spot: {self._active_trade.exit_spot:.2f}\n"
                    f"Reason: End of day"
                )

    # ── Helpers ─────────────────────────────────────────────────────

    async def _place_option_order(
        self,
        instrument: InstrumentConfig,
        trade: PDHPDLTrade,
        side: str,
        quantity: int,
        tag: str,
    ) -> tuple[bool, str]:
        """Place an option order via the configured broker abstraction."""
        if not trade.option_symbol or quantity <= 0:
            return False, ""
        if self.broker is None:
            logger.error("[PDHPDL] No broker configured — cannot place %s order", side)
            return False, ""
        request = OrderRequest(
            instrument=instrument,
            trading_symbol=trade.option_symbol,
            symbol_token=trade.symboltoken,
            exchange=trade.exchange or "NFO",
            side=OrderSide(side),
            order_type=OrderType.MARKET,
            product_type=ProductType.INTRADAY,
            quantity=quantity,
            price=0.0,
            trigger_price=0.0,
            underlying=instrument.option_symbol_prefix or instrument.symbol,
            expiry_date=self._expiry_date,
            strike=float(trade.strike_price or 0),
            option_type=trade.side,
        )
        try:
            resp = await asyncio.to_thread(self.broker.place_order, request)
        except Exception:
            logger.exception(
                "[PDHPDL] Broker order exception side=%s symbol=%s tag=%s",
                side, trade.option_symbol, tag,
            )
            return False, ""
        if not resp or resp.status == OrderStatus.REJECTED:
            msg = (resp.message if resp else "no response") or "rejected"
            logger.error(
                "[PDHPDL] Order rejected side=%s symbol=%s tag=%s msg=%s",
                side, trade.option_symbol, tag, msg,
            )
            return False, getattr(resp, "order_id", "") or ""
        order_id = resp.order_id or ""
        logger.info(
            "[PDHPDL] %s order placed via %s: %s qty=%d id=%s tag=%s",
            side, self.broker.name, trade.option_symbol, quantity, order_id, tag,
        )
        return True, order_id

    async def _fetch_option_quote(
        self,
        instrument: InstrumentConfig,
        strike: float,
        option_type: str,
    ) -> Optional[dict]:
        if not self._expiry:
            return None
        symbol = instrument.build_option_symbol(self._expiry, strike, option_type)
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            logger.warning("[PDHPDL] Token not found for %s", symbol)
            return None
        try:
            quote = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.get_option_quote,
                    "NFO",
                    token_info.get("tradingsymbol", ""),
                    token_info.get("symboltoken", ""),
                ),
                timeout=15,
            )
            return quote
        except asyncio.TimeoutError:
            logger.warning("[PDHPDL] Quote fetch timed out for %s", symbol)
            return None
        except Exception:
            logger.exception("[PDHPDL] Error fetching quote for %s", symbol)
            return None
