"""Range Breakout Scanner — Consolidation → Breakout (PAPER-ONLY).

Strategy summary (backtested on NIFTY, 2025-10-13 → 2026-04-24):
  - Setup:    every trading day — ADX < 20 (range-bound) + 30-candle range < 0.80%
  - Window:   09:45–10:30 AM (post-ORB consolidation zone)
  - Entry:    1-min close ABOVE 30-candle range high (CE) or BELOW range low (PE)
              + RSI ≥ 58 (CE) / ≤ 42 (PE)  + body ratio ≥ 0.45
              + option entry price ≥ ₹80
  - Stop:     -15% of option premium
  - Target1:  +15%  (T1 — breakeven trail)
  - Target2:  +30%  (T2 — full exit)
  - EOD:      force exit at 15:20

Backtest result (Option C config):
  N=41 trades, WR=68.3%, PF=3.52, Max DD ₹9,823 per lot, total PnL ₹45,611/lot.

This scanner is OBSERVE-ONLY: it sends Telegram alerts and records
AlertItems to DB but does NOT write to the paper-trader trades DB.
It respects the common mutex with all other scanners — no new entry
while ANY peer holds an open trade.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional

import pandas as pd
import pytz

from app.alerts.alert_manager import AlertManager
from app.core.instruments import InstrumentConfig
from app.core.models import AlertItem, OptionsMetrics
from app.data.angelone_client import AngelOneClient
from app.strategies.range_breakout import (
    RangeBreakoutStrategy,
    WINDOW_START,
    WINDOW_END,
    RANGE_LOOKBACK,
    ADX_THRESHOLD,
    RANGE_PCT_THRESHOLD,
    MIN_BODY_RATIO,
    CALL_RSI_MIN,
    PUT_RSI_MAX,
    SL_PCT,
    MIN_ENTRY_PREMIUM,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

EOD_FORCE_CLOSE = dtime(15, 20)
T1_PCT = SL_PCT          # T1 = +1R (same as SL distance → symmetric)
T2_PCT = SL_PCT * 2.0    # T2 = +2R

# Singleton strategy instance for signal detection
_strategy = RangeBreakoutStrategy()
_EMPTY_OPTS = OptionsMetrics()


class RBTrade:
    """Tracks a single Range Breakout paper trade from entry to exit."""

    def __init__(
        self,
        trade_date: str,
        side: str,               # 'CE' or 'PE'
        entry_time: str,
        entry_spot: float,
        range_high: float,
        range_low: float,
        range_pct: float,
        adx: float,
        rsi: float,
        option_symbol: str,
        option_entry_price: float,
        option_sl_price: float,
        option_t1_price: float,
        option_t2_price: float,
        strike_price: float,
    ):
        self.trade_date = trade_date
        self.side = side
        self.entry_time = entry_time
        self.entry_spot = entry_spot
        self.range_high = range_high
        self.range_low = range_low
        self.range_pct = range_pct
        self.adx = adx
        self.rsi = rsi
        self.option_symbol = option_symbol
        self.option_entry_price = option_entry_price
        self.option_sl_price = option_sl_price
        self.option_t1_price = option_t1_price
        self.option_t2_price = option_t2_price
        self.strike_price = strike_price
        # Trail state
        self.t1_hit = False
        self.option_trail_sl: float = option_sl_price   # moves to entry after T1, then trails
        # Exit tracking
        self.exited = False
        self.exit_time: str = ""
        self.exit_spot: float = 0.0
        self.exit_reason: str = ""
        self.option_exit_price: float = 0.0


class RangeBreakoutScanner:
    """Range Breakout scanner — paper-only, runs each 1-min cycle during market hours.

    Lifecycle:
      1. Each cycle: feed df_today (1-min OHLCV + ADX + RSI) to RangeBreakoutStrategy.evaluate().
      2. On signal: fetch live option quote, apply min-premium filter, enter trade.
      3. While in trade: each cycle fetch live option LTP, check SL / T1 / T2 / EOD.
      4. At 15:20: force exit if still open.

    Mutex: orchestrator passes ``peer_in_trade=True`` whenever any other scanner
    currently holds an open trade. The scanner skips new entries in that state.
    It also exposes ``is_in_trade()`` so peers can do the same.
    """

    def __init__(
        self,
        client: AngelOneClient,
        alert_manager: AlertManager,
    ):
        self.client = client
        self.alert_manager = alert_manager

        # Daily state
        self._today_str: str = ""
        self._signal_found_today = False
        self._active_trade: Optional[RBTrade] = None
        self._expiry: str = ""
        self._expiry_date: Optional[date] = None

    # ── Public API used by orchestrator ─────────────────────────────

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
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

        now = datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")
        if self._today_str != today_str:
            if self._today_str:
                logger.info("[RB] Day rollover: %s → %s. Resetting state.", self._today_str, today_str)
            self.reset_daily()
            self._today_str = today_str

        # ── If in trade: manage exit first ──────────────────────────
        if self._active_trade and not self._active_trade.exited:
            await self._check_exit(df_today, instrument, now)
            return

        # ── Mutex: skip new entries if any peer is in a trade ───────
        if peer_in_trade:
            return

        # ── Already fired today ─────────────────────────────────────
        if self._signal_found_today:
            return

        # ── Past entry window? Stop scanning ────────────────────────
        if now.time() > WINDOW_END:
            if not self._signal_found_today:
                self._signal_found_today = True   # prevent repeated log
                logger.info("[RB] %s — past entry window %s, no signal today",
                            today_str, WINDOW_END.strftime("%H:%M"))
            return

        # ── Delegate to strategy for signal detection ─────────────
        try:
            signal = _strategy.evaluate(df_today, _EMPTY_OPTS, float(df_today.iloc[-1]["close"]))
        except Exception:
            logger.exception("[RB] Strategy evaluate() raised an exception")
            return

        if signal is None:
            return

        side = signal.option_type.value  # 'CE' or 'PE'

        # ── Fetch live option quote ──────────────────────────────────
        entry_spot = float(df_today.iloc[-1]["close"])
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

        # ── Min entry premium filter ─────────────────────────────────
        if MIN_ENTRY_PREMIUM > 0 and option_ltp > 0 and option_ltp < MIN_ENTRY_PREMIUM:
            logger.info(
                "[RB] SKIP %s: option LTP ₹%.2f < min ₹%.0f",
                side, option_ltp, MIN_ENTRY_PREMIUM,
            )
            # Don't set signal_found_today — keep watching in case a better candle forms
            return

        # ── Build trade ──────────────────────────────────────────────
        entry_time = now.strftime("%H:%M")
        details = signal.details or {}
        range_high = details.get("range_high", 0.0)
        range_low = details.get("range_low", 0.0)
        range_pct = details.get("range_pct", 0.0)
        adx = details.get("adx", 0.0)
        rsi = details.get("rsi", 0.0)

        opt_sl = round(option_ltp * (1 - SL_PCT), 2) if option_ltp > 0 else 0.0
        opt_t1 = round(option_ltp * (1 + T1_PCT), 2) if option_ltp > 0 else 0.0
        opt_t2 = round(option_ltp * (1 + T2_PCT), 2) if option_ltp > 0 else 0.0

        self._active_trade = RBTrade(
            trade_date=today_str,
            side=side,
            entry_time=entry_time,
            entry_spot=entry_spot,
            range_high=range_high,
            range_low=range_low,
            range_pct=range_pct,
            adx=adx,
            rsi=rsi,
            option_symbol=option_symbol,
            option_entry_price=option_ltp,
            option_sl_price=opt_sl,
            option_t1_price=opt_t1,
            option_t2_price=opt_t2,
            strike_price=strike,
        )
        self._signal_found_today = True

        expiry_display = self._expiry or "N/A"
        days_to_expiry = (self._expiry_date - now.date()).days if self._expiry_date else "?"
        breakout_level = range_high if side == "CE" else range_low
        sl_pts = round(option_ltp * SL_PCT, 2)
        t1_pts = round(option_ltp * T1_PCT, 2)
        t2_pts = round(option_ltp * T2_PCT, 2)

        msg = (
            f"🟢 RANGE BREAKOUT — {side} SIGNAL (PAPER)\n"
            f"{'=' * 38}\n"
            f"\n"
            f"📅 Date: {today_str}\n"
            f"⏰ Entry: {entry_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Spot: {entry_spot:.2f}\n"
            f"  Breakout level: {breakout_level:.2f}  ({'above range_high' if side == 'CE' else 'below range_low'})\n"
            f"  Range: [{range_low:.2f}, {range_high:.2f}]  ({range_pct:.2f}%)\n"
            f"\n"
            f"📐 INDICATORS\n"
            f"  ADX: {adx:.1f}  (need < {ADX_THRESHOLD:.0f} — ranging)\n"
            f"  RSI: {rsi:.1f}  ({'need ≥ ' + str(int(CALL_RSI_MIN)) if side == 'CE' else 'need ≤ ' + str(int(PUT_RSI_MAX))})\n"
            f"\n"
            f"🎯 OPTION ({side})\n"
            f"  Strike: {int(strike)} {side}\n"
            f"  Symbol: {option_symbol}\n"
            f"  Expiry: {expiry_display} ({days_to_expiry}d)\n"
            f"  Entry LTP : ₹{option_ltp:.2f}\n"
            f"  Bid: ₹{option_bid:.2f} | Ask: ₹{option_ask:.2f}\n"
            f"  Spread: {spread_pct:.1f}%\n"
            f"\n"
            f"📋 EXIT RULES\n"
            f"  SL  (-{SL_PCT*100:.0f}%): ₹{opt_sl:.2f}  (-₹{sl_pts:.2f})\n"
            f"  T1  (+{T1_PCT*100:.0f}%): ₹{opt_t1:.2f}  (+₹{t1_pts:.2f})  → trail SL to entry\n"
            f"  T2  (+{T2_PCT*100:.0f}%): ₹{opt_t2:.2f}  (+₹{t2_pts:.2f})  → full exit\n"
            f"  EOD : 15:20 force close\n"
            f"\n"
            f"⚠️ OBSERVE ONLY — paper trade, no execution"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[RB] ENTRY %s @ spot=%.2f opt=₹%.2f SL=₹%.2f T1=₹%.2f T2=₹%.2f sym=%s",
            side, entry_spot, option_ltp, opt_sl, opt_t1, opt_t2, option_symbol,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=f"RB BREAKOUT {side} — NIFTY {int(strike)} {side}",
            message=msg,
            timestamp=now,
        )
        await self.alert_manager.record(alert)

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

        # Fetch live option LTP
        option_ltp = 0.0
        if trade.option_symbol and self._expiry:
            quote = await self._fetch_option_quote(instrument, trade.strike_price, trade.side)
            if quote:
                option_ltp = float(quote.get("ltp", 0.0) or 0.0)

        exit_reason = ""

        if option_ltp > 0:
            # T2 check first (full exit)
            if option_ltp >= trade.option_t2_price:
                exit_reason = "target2"
            # T1 check — move trail SL to entry
            elif not trade.t1_hit and option_ltp >= trade.option_t1_price:
                trade.t1_hit = True
                trade.option_trail_sl = round(trade.option_entry_price * 1.005, 2)
                logger.info("[RB] T1 HIT — trail SL moved to ₹%.2f", trade.option_trail_sl)
                await self.alert_manager.telegram.send(
                    f"🎯 RB T1 HIT — {trade.side} (PAPER)\n"
                    f"Option at ₹{option_ltp:.2f} — T1 ₹{trade.option_t1_price:.2f} reached\n"
                    f"Trail SL moved to ₹{trade.option_trail_sl:.2f} (entry breakeven)\n"
                    f"T2 target: ₹{trade.option_t2_price:.2f}"
                )
            # SL check (uses trail after T1)
            if not exit_reason and option_ltp <= trade.option_trail_sl:
                exit_reason = "trailing_sl" if trade.t1_hit else "stoploss"

        # EOD force close
        if not exit_reason and now.time() >= EOD_FORCE_CLOSE:
            exit_reason = "eod_close"

        if not exit_reason:
            return

        # Spot at exit
        last = df_today.iloc[-1]
        exit_spot = float(last["close"])

        trade.exited = True
        trade.exit_time = now.strftime("%H:%M")
        trade.exit_spot = exit_spot
        trade.exit_reason = exit_reason
        trade.option_exit_price = option_ltp

        is_win = option_ltp > trade.option_entry_price if (trade.option_entry_price > 0 and option_ltp > 0) else False
        opt_pts = (option_ltp - trade.option_entry_price) if trade.option_entry_price > 0 else 0.0
        opt_pct = (opt_pts / trade.option_entry_price * 100) if trade.option_entry_price > 0 else 0.0

        emoji = "✅" if is_win else "❌"
        exit_label = {"target2": "T2 TARGET", "stoploss": "STOP LOSS",
                      "trailing_sl": "TRAIL SL", "eod_close": "EOD CLOSE"}.get(exit_reason, exit_reason.upper())
        msg = (
            f"{emoji} RANGE BREAKOUT — EXIT {exit_label} (PAPER)\n"
            f"{'=' * 38}\n"
            f"\n"
            f"📅 Date: {trade.trade_date}\n"
            f"⏰ Entry: {trade.entry_time} → Exit: {trade.exit_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Side: {trade.side}\n"
            f"  Entry: {trade.entry_spot:.2f} → Exit: {exit_spot:.2f}\n"
            f"\n"
            f"🎯 OPTION ({trade.side})\n"
            f"  {trade.option_symbol}\n"
            f"  Entry : ₹{trade.option_entry_price:.2f}\n"
            f"  Exit  : ₹{option_ltp:.2f}\n"
            f"  PnL   : {opt_pts:+.2f} pts ({opt_pct:+.1f}%)\n"
            f"  T1 hit: {'Yes' if trade.t1_hit else 'No'}\n"
            f"\n"
            f"📋 Exit: {exit_label}\n"
            f"\n"
            f"⚠️ OBSERVE ONLY — paper trade"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[RB] EXIT %s %s opt=₹%.2f→₹%.2f (%+.1f%%) reason=%s",
            "WIN" if is_win else "LOSS", trade.side,
            trade.option_entry_price, option_ltp, opt_pct, exit_reason,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=f"RB {'WIN' if is_win else 'LOSS'} — {exit_label}",
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
                    f"🔔 Range Breakout — EOD Force Close\n"
                    f"Exit spot: {self._active_trade.exit_spot:.2f}\n"
                    f"Reason: End of day"
                )

    # ── Helpers ─────────────────────────────────────────────────────

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
            logger.warning("[RB] Token not found for %s", symbol)
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
        except Exception:
            logger.warning("[RB] Option quote fetch failed for %s", symbol)
            return None
