"""Move Detection Scanner — Production v2 AGGRESSIVE tier.

Expansion → Hold → Continuation move detection with optimised filters.
Backtested: 16 trades, 10W/6L (62.5% win), +1.90% index, +104.6% option PnL
over 125 trading days.

AGGRESSIVE tier filters:
  - Bearish direction only (skip all bullish)
  - Trending days only (skip choppy AND mild_chop)
  - Confidence ≥ 80
  - confirmed_trend_start only
  - Entry window: candle 45–135 (~10:00–11:30 AM)
  - Weekly filter: max 1 trade per ISO week (first signal)

Exit rules (same as Config P):
  1. Structure break — close > expansion body midpoint
  2. VWAP break — close > VWAP × 1.001
  3. Trailing stop — after 10 candles in profit, trail using 5-candle swing high + buffer
  4. Max hold — 120 candles (~2 hours)
  5. End of day — 15:20 force close

Key difference from Config P:
  - Uses scan_all_moves() + select_production_trade() pipeline (multi-move scan)
  - Higher confidence threshold (80 vs 70)
  - Config P uses detect_move() (single latest move)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, date, time as dtime
from typing import Optional

import pandas as pd
import pytz

from app.alerts.alert_manager import AlertManager
from app.core.instruments import InstrumentConfig
from app.core.models import AlertItem
from app.data.angelone_client import AngelOneClient
from app.engine.feature_engine import (
    FeatureEngine,
    assess_day_quality,
    scan_all_moves,
    select_production_trade,
)

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── AGGRESSIVE tier parameters (LOCKED — matches backtest exactly) ────
MIN_CONFIDENCE = 80
MIN_CONFIDENCE_BULLISH = 999    # Effectively skip all bullish
EARLIEST_CANDLE = 45            # ~10:00 AM
DEADLINE_CANDLE = 135           # ~11:30 AM
TRAIL_LOOKBACK = 5
TRAIL_BUFFER_PCT = 0.02         # 0.02% buffer on trailing stop
OPTION_LEVERAGE = 55            # Approximate ATM PUT leverage
MAX_HOLD_CANDLES = 120          # ~2 hours max
DAY_ASSESSMENT_TIME = dtime(10, 0)

SCANNER_TAG = "MOVE-DET"        # Telegram message prefix


class ActiveTrade:
    """Tracks an active Move Detection signal from entry through exit."""

    def __init__(
        self,
        trade_date: str,
        signal_time: str,
        entry_time: str,
        entry_price: float,
        entry_candle_idx: int,
        expansion_idx: int,
        body_mid: float,
        confidence: int,
        expansion_ratio: float,
        ema_aligned: bool,
        vwap_aligned: bool,
        direction: str = "bearish",
        option_symbol: str = "",
        option_entry_price: float = 0.0,
        strike_price: float = 0.0,
        spot_at_entry: float = 0.0,
    ):
        self.trade_date = trade_date
        self.signal_time = signal_time
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.entry_candle_idx = entry_candle_idx
        self.expansion_idx = expansion_idx
        self.body_mid = body_mid
        self.confidence = confidence
        self.expansion_ratio = expansion_ratio
        self.ema_aligned = ema_aligned
        self.vwap_aligned = vwap_aligned
        self.direction = direction
        self.option_symbol = option_symbol
        self.option_entry_price = option_entry_price
        self.strike_price = strike_price
        self.spot_at_entry = spot_at_entry

        # Exit tracking
        self.best_price = entry_price
        self.candles_in_trade = 0
        self.trailing_active = False
        self.exited = False
        self.exit_price: float = 0.0
        self.exit_time: str = ""
        self.exit_reason: str = ""
        self.option_exit_price: float = 0.0


class MoveDetectionScanner:
    """Production v2 AGGRESSIVE scanner — runs each minute during market hours.

    Lifecycle:
      1. At 10:00 AM (candle 45): assess day quality (trending only)
      2. Each cycle 10:00-11:30: scan_all_moves → select_production_trade with AGGRESSIVE filters
      3. On signal: fetch option quote, send Telegram entry alert
      4. While in trade: monitor exit conditions each cycle
      5. On exit: send Telegram exit alert with PnL
      6. At 15:20: force-close any open trade
    """

    def __init__(
        self,
        client: AngelOneClient,
        feature_engine: FeatureEngine,
        alert_manager: AlertManager,
    ):
        self.client = client
        self.fe = feature_engine
        self.alert_manager = alert_manager

        # Daily state
        self._day_assessed = False
        self._day_tradeable = False
        self._day_quality: dict = {}
        self._signal_found_today = False
        self._active_trade: Optional[ActiveTrade] = None
        self._last_trade_week: Optional[int] = None  # ISO week number
        self._today_str: str = ""
        self._expiry: str = ""
        self._expiry_date: Optional[date] = None

    def reset_daily(self) -> None:
        """Reset state for a new trading day."""
        self._day_assessed = False
        self._day_tradeable = False
        self._day_quality = {}
        self._signal_found_today = False
        self._active_trade = None
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        # Don't reset _last_trade_week — persists across days for weekly filter

    def set_expiry(self, expiry: str, expiry_date: Optional[date] = None) -> None:
        """Set the weekly option expiry string (e.g. '22APR26')."""
        self._expiry = expiry
        self._expiry_date = expiry_date

    async def run_cycle(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        cycle: int,
    ) -> None:
        """Run one Move Detection scan cycle. Called every 60s by orchestrator."""
        if df_today is None or df_today.empty:
            return

        if isinstance(df_today.index, pd.DatetimeIndex):
            # Count only regular session candles; ignore any pre-open/non-session rows.
            df_today = df_today.between_time("09:15", "15:30")
            if df_today.empty:
                return

        now = datetime.now(IST)
        candle_count = len(df_today)
        today_str = now.strftime("%Y-%m-%d")
        if self._today_str != today_str:
            # Day rollover detected mid-loop (or first run) — force a fresh reset
            # so daily flags / Telegram messages reflect today's date, not yesterday's.
            if self._today_str:
                logger.info(
                    "[MoveDet] Day rollover detected: %s → %s. Resetting daily state.",
                    self._today_str, today_str,
                )
            self.reset_daily()
            self._today_str = today_str

        # ── If we have an active trade, check exits first ────────────
        if self._active_trade and not self._active_trade.exited:
            await self._check_exit(df_today, instrument, now)
            return  # Don't look for new signals while in a trade

        # ── Already found a signal today — skip ──────────────────────
        if self._signal_found_today:
            return

        # ── Weekly filter: only 1 trade per ISO week ─────────────────
        current_week = now.isocalendar()[1]
        if self._last_trade_week == current_week:
            return  # Already traded this week

        # ── Day quality assessment (once, after 10:00 and >=45 candles) ──
        if (
            not self._day_assessed
            and now.time() >= DAY_ASSESSMENT_TIME
            and candle_count >= EARLIEST_CANDLE
        ):
            self._day_quality = assess_day_quality(df_today, check_candles=EARLIEST_CANDLE)
            self._day_assessed = True

            vwap_trend = self._day_quality.get("vwap_trend", "unknown")
            tradeable = self._day_quality.get("tradeable", False)

            # AGGRESSIVE: skip choppy AND mild_chop days (trending only)
            if not tradeable or vwap_trend == "mild_chop":
                self._day_tradeable = False
                reason = "choppy" if not tradeable else "mild_chop"
                logger.info(
                    "[MoveDet] Day %s SKIPPED: %s at %s (candles=%d, VWAP crosses=%d, trend=%s)",
                    self._today_str,
                    reason,
                    now.strftime("%H:%M"),
                    candle_count,
                    self._day_quality.get("vwap_crosses", 0),
                    vwap_trend,
                )
                await self.alert_manager.telegram.send(
                    f"📊 {SCANNER_TAG} — Day Skip\n"
                    f"Date: {self._today_str}\n"
                    f"Assessed at: {now.strftime('%H:%M')} (candles={candle_count})\n"
                    f"Reason: {reason}\n"
                    f"VWAP crosses: {self._day_quality.get('vwap_crosses', 0)}\n"
                    f"Trend: {vwap_trend}\n"
                    f"No trades today."
                )
                return
            else:
                self._day_tradeable = True
                logger.info(
                    "[MoveDet] Day %s TRADEABLE at %s: trend=%s, candles=%d, VWAP crosses=%d, direction=%.2f%%",
                    self._today_str,
                    now.strftime("%H:%M"),
                    vwap_trend,
                    candle_count,
                    self._day_quality.get("vwap_crosses", 0),
                    self._day_quality.get("net_move_pct", 0),
                )
                await self.alert_manager.telegram.send(
                    f"📊 {SCANNER_TAG} — Day Assessment\n"
                    f"Date: {self._today_str}\n"
                    f"Assessed at: {now.strftime('%H:%M')} (candles={candle_count})\n"
                    f"Status: TRADEABLE ✅\n"
                    f"VWAP trend: {vwap_trend}\n"
                    f"VWAP crosses: {self._day_quality.get('vwap_crosses', 0)}\n"
                    f"Direction: {self._day_quality.get('net_move_pct', 0):.3f}%\n"
                    f"Scanning for bearish signals (AGGRESSIVE)..."
                )

        if not self._day_tradeable:
            return

        # ── Past deadline? No more scanning ──────────────────────────
        if candle_count > DEADLINE_CANDLE:
            if not self._signal_found_today:
                self._signal_found_today = True  # Stop scanning
                logger.info("[MoveDet] Past deadline (candle %d > %d) — no signal today", candle_count, DEADLINE_CANDLE)
                await self.alert_manager.telegram.send(
                    f"⏰ {SCANNER_TAG} — Deadline Passed\n"
                    f"Date: {self._today_str}\n"
                    f"No qualifying bearish signal by 11:30 AM.\n"
                    f"Done for today."
                )
            return

        # ── Not enough candles yet ───────────────────────────────────
        if candle_count < EARLIEST_CANDLE:
            return

        # ── Scan all moves and apply production filters ──────────────
        moves = scan_all_moves(df_today)
        if not moves:
            return

        trades = select_production_trade(
            df_today,
            moves,
            min_confidence=MIN_CONFIDENCE,
            min_confidence_bullish=MIN_CONFIDENCE_BULLISH,
            earliest_candle=EARLIEST_CANDLE,
            deadline_candle=DEADLINE_CANDLE,
            max_trades=1,
        )

        if not trades:
            return

        # AGGRESSIVE: bearish only
        trades = [t for t in trades if t["direction"] == "bearish"]
        if not trades:
            return

        trade = trades[0]

        # ── SIGNAL DETECTED ──────────────────────────────────────────
        logger.info(
            "[MoveDet] SIGNAL DETECTED: %s conf=%d exp_ratio=%.2f ema=%s vwap=%s",
            trade["direction"], trade["confidence"], trade["expansion_ratio"],
            trade.get("ema_aligned"), trade.get("vwap_aligned"),
        )

        self._signal_found_today = True
        spot_price = float(df_today.iloc[-1]["close"])

        # Calculate ATM PUT strike
        strike = instrument.nearest_strike(spot_price, "PE")

        # Compute expansion body midpoint for exit tracking
        sig_idx = trade["idx"]
        o_i = float(df_today.iloc[sig_idx]["open"])
        c_i = float(df_today.iloc[sig_idx]["close"])
        body_mid = (max(o_i, c_i) + min(o_i, c_i)) / 2

        # Entry price and candle index
        entry_price = trade["final_entry"]
        entry_candle_idx = trade["final_entry_idx"]

        # Entry time
        if entry_candle_idx < len(df_today):
            entry_ts = df_today.iloc[entry_candle_idx].get("timestamp")
            entry_time = entry_ts.strftime("%H:%M") if hasattr(entry_ts, "strftime") else str(entry_ts)
        else:
            entry_time = now.strftime("%H:%M")

        # Signal time
        sig_ts = df_today.iloc[sig_idx].get("timestamp")
        signal_time = sig_ts.strftime("%H:%M") if hasattr(sig_ts, "strftime") else str(sig_ts)

        # Build option symbol and fetch live quote
        option_symbol = ""
        option_ltp = 0.0
        option_bid = 0.0
        option_ask = 0.0
        spread_pct = 0.0

        if self._expiry:
            option_symbol = instrument.build_option_symbol(self._expiry, strike, "PE")
            quote = await self._fetch_option_quote(instrument, strike, "PE")
            if quote:
                option_ltp = quote.get("ltp", 0.0)
                option_bid = quote.get("best_bid", 0.0)
                option_ask = quote.get("best_ask", 0.0)
                spread_pct = quote.get("spread_pct", 0.0)

        # Signal candle OHLC
        sig_o = round(float(df_today.iloc[sig_idx]["open"]), 2)
        sig_h = round(float(df_today.iloc[sig_idx]["high"]), 2)
        sig_l = round(float(df_today.iloc[sig_idx]["low"]), 2)
        sig_c = round(float(df_today.iloc[sig_idx]["close"]), 2)

        # EMA values
        ema9 = float(df_today.iloc[sig_idx].get("ema9", 0)) if "ema9" in df_today.columns else 0
        ema20 = float(df_today.iloc[sig_idx].get("ema20", 0)) if "ema20" in df_today.columns else 0
        vwap = float(df_today.iloc[sig_idx].get("vwap", 0)) if "vwap" in df_today.columns else 0

        # Pullback info
        pullback_info = ""
        if trade.get("pullback"):
            pb = trade["pullback"]
            pullback_info = (
                f"\n  Pullback: depth={pb.get('pullback_depth', 0):.1f}%, "
                f"entry improved to {trade['final_entry']:.2f}"
            )

        # Create active trade
        self._active_trade = ActiveTrade(
            trade_date=self._today_str,
            signal_time=signal_time,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_candle_idx=entry_candle_idx,
            expansion_idx=sig_idx,
            body_mid=body_mid,
            confidence=trade["confidence"],
            expansion_ratio=trade["expansion_ratio"],
            ema_aligned=trade.get("ema_aligned", False),
            vwap_aligned=trade.get("vwap_aligned", False),
            direction=trade["direction"],
            option_symbol=option_symbol,
            option_entry_price=option_ltp,
            strike_price=strike,
            spot_at_entry=spot_price,
        )

        # Update weekly filter
        self._last_trade_week = now.isocalendar()[1]

        # Calculate expected levels
        sl_points = round(body_mid - entry_price, 2)
        sl_pct_option = round(abs(sl_points) / entry_price * 100 * OPTION_LEVERAGE, 1)

        # ── SEND ENTRY ALERT ─────────────────────────────────────────
        expiry_display = self._expiry if self._expiry else "N/A"
        days_to_expiry = (self._expiry_date - now.date()).days if self._expiry_date else "?"

        msg = (
            f"🔴 {SCANNER_TAG} — BEARISH SIGNAL\n"
            f"{'='*35}\n"
            f"\n"
            f"📅 Date: {self._today_str}\n"
            f"⏰ Signal: {signal_time} | Entry: {entry_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Entry (Sell): {entry_price:.2f}\n"
            f"  Spot now: {spot_price:.2f}\n"
            f"  SL level (body mid): {body_mid:.2f}\n"
            f"  SL distance: {sl_points:+.2f} pts\n"
            f"\n"
            f"📈 SIGNAL DETAILS\n"
            f"  Confidence: {trade['confidence']}/100\n"
            f"  Expansion: {trade['expansion_ratio']:.1f}×\n"
            f"  Hold candles: {trade['hold_candles']}\n"
            f"  EMA aligned: {'✅' if trade.get('ema_aligned') else '❌'}\n"
            f"  VWAP aligned: {'✅' if trade.get('vwap_aligned') else '❌'}\n"
            f"  Signal candle: O={sig_o} H={sig_h} L={sig_l} C={sig_c}\n"
            f"  EMA9={ema9:.2f} | EMA20={ema20:.2f} | VWAP={vwap:.2f}"
            f"{pullback_info}\n"
            f"\n"
            f"🎯 OPTION (PUT)\n"
            f"  Strike: {int(strike)} PE\n"
            f"  Symbol: {option_symbol}\n"
            f"  Expiry: {expiry_display} ({days_to_expiry}d)\n"
            f"  LTP: ₹{option_ltp:.2f}\n"
            f"  Bid: ₹{option_bid:.2f} | Ask: ₹{option_ask:.2f}\n"
            f"  Spread: {spread_pct:.1f}%\n"
            f"  Est. SL loss: ~{sl_pct_option:.0f}% of premium\n"
            f"\n"
            f"📋 EXIT RULES\n"
            f"  1. Structure break: close > {body_mid:.2f}\n"
            f"  2. VWAP break: close > VWAP\n"
            f"  3. Trailing stop: after 10 candles in profit\n"
            f"  4. EOD: 15:20 force close\n"
            f"\n"
            f"⚠️ OBSERVE ONLY — No auto-execution"
        )

        await self.alert_manager.telegram.send(msg)
        logger.info("[MoveDet] Entry alert sent: %s %d PE @ ₹%.2f", option_symbol, strike, option_ltp)

        # Also store in UI alert store
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=f"{SCANNER_TAG} SIGNAL — NIFTY {int(strike)} PE",
            message=msg,
            timestamp=now,
        )
        self.alert_manager.store.add(alert)

    async def _check_exit(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        now: datetime,
    ) -> None:
        """Check exit conditions for active trade."""
        trade = self._active_trade
        if trade is None or trade.exited:
            return

        candle_count = len(df_today)
        if candle_count == 0:
            return

        last = df_today.iloc[-1]
        current_close = float(last["close"])
        current_high = float(last["high"])
        current_low = float(last["low"])

        trade.candles_in_trade = candle_count - trade.entry_candle_idx
        trade.best_price = min(trade.best_price, current_low)  # Bearish: lower is better

        exit_reason = ""
        exit_price = current_close

        # ── Exit 1: Structure break — close above body midpoint ──────
        if current_close > trade.body_mid:
            exit_reason = "structure_break"
            exit_price = current_close

        # ── Exit 2: VWAP break — close above VWAP ───────────────────
        if not exit_reason and "vwap" in df_today.columns:
            vwap_val = last.get("vwap")
            if vwap_val is not None and not pd.isna(vwap_val):
                if current_close > float(vwap_val) * 1.001:
                    exit_reason = "vwap_break"
                    exit_price = current_close

        # ── Exit 3: Trailing stop ────────────────────────────────────
        if not exit_reason and trade.candles_in_trade >= 10 and trade.best_price < trade.entry_price:
            lookback_start = max(trade.entry_candle_idx + 1, candle_count - TRAIL_LOOKBACK)
            if lookback_start < candle_count:
                swing_high = float(df_today.iloc[lookback_start:candle_count]["high"].max())
                trail_stop = swing_high * (1 + TRAIL_BUFFER_PCT / 100)
                if current_close > trail_stop and trail_stop < trade.body_mid:
                    exit_reason = "trailing_stop"
                    exit_price = current_close
                    trade.trailing_active = True

        # ── Exit 4: Max hold ─────────────────────────────────────────
        if not exit_reason and trade.candles_in_trade >= MAX_HOLD_CANDLES:
            exit_reason = "max_hold"
            exit_price = current_close

        # ── Exit 5: EOD force close (15:20) ──────────────────────────
        if not exit_reason and now.time() >= dtime(15, 20):
            exit_reason = "eod_close"
            exit_price = current_close

        if not exit_reason:
            return

        # ── PROCESS EXIT ─────────────────────────────────────────────
        trade.exited = True
        trade.exit_price = exit_price
        trade.exit_time = now.strftime("%H:%M")
        trade.exit_reason = exit_reason

        # Calculate PnL (bearish: profit when price drops)
        index_pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
        index_pnl_pts = trade.entry_price - exit_price
        option_pnl_pct = index_pnl_pct * OPTION_LEVERAGE
        is_win = index_pnl_pct > 0

        # Fetch current option price
        option_exit_ltp = 0.0
        if trade.option_symbol and self._expiry:
            quote = await self._fetch_option_quote(instrument, trade.strike_price, "PE")
            if quote:
                option_exit_ltp = quote.get("ltp", 0.0)
                trade.option_exit_price = option_exit_ltp

        # Calculate actual option PnL if we have both prices
        actual_option_pnl = ""
        if trade.option_entry_price > 0 and option_exit_ltp > 0:
            actual_pnl_pct = (option_exit_ltp - trade.option_entry_price) / trade.option_entry_price * 100
            actual_option_pnl = f"\n  Actual option PnL: {actual_pnl_pct:+.1f}% (₹{trade.option_entry_price:.2f} → ₹{option_exit_ltp:.2f})"

        emoji = "✅" if is_win else "❌"
        msg = (
            f"{emoji} {SCANNER_TAG} — EXIT {'WIN' if is_win else 'LOSS'}\n"
            f"{'='*35}\n"
            f"\n"
            f"📅 Date: {trade.trade_date}\n"
            f"⏰ Entry: {trade.entry_time} → Exit: {trade.exit_time}\n"
            f"⏱️ Duration: {trade.candles_in_trade} candles (~{trade.candles_in_trade} min)\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Entry: {trade.entry_price:.2f}\n"
            f"  Exit: {exit_price:.2f}\n"
            f"  PnL: {index_pnl_pts:+.2f} pts ({index_pnl_pct:+.3f}%)\n"
            f"  Best price: {trade.best_price:.2f} ({trade.entry_price - trade.best_price:+.2f} pts)\n"
            f"\n"
            f"🎯 OPTION (PUT)\n"
            f"  {trade.option_symbol}\n"
            f"  Entry LTP: ₹{trade.option_entry_price:.2f}\n"
            f"  Exit LTP: ₹{option_exit_ltp:.2f}\n"
            f"  Est. option PnL: {option_pnl_pct:+.1f}% (55× leverage)"
            f"{actual_option_pnl}\n"
            f"\n"
            f"📋 Exit Reason: {exit_reason}\n"
            f"  Confidence was: {trade.confidence}/100\n"
            f"  Expansion was: {trade.expansion_ratio:.1f}×\n"
            f"\n"
            f"⚠️ OBSERVE ONLY"
        )

        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[MoveDet] EXIT %s: %s, PnL=%+.2f pts (%+.3f%%), option est=%+.1f%%",
            "WIN" if is_win else "LOSS", exit_reason,
            index_pnl_pts, index_pnl_pct, option_pnl_pct,
        )

        # Store in UI
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=f"{SCANNER_TAG} {'WIN' if is_win else 'LOSS'} — {exit_reason}",
            message=msg,
            timestamp=now,
        )
        self.alert_manager.store.add(alert)

    async def force_close(self, df_today: pd.DataFrame, instrument: InstrumentConfig) -> None:
        """Force close any active trade (called at EOD 15:20)."""
        if self._active_trade and not self._active_trade.exited:
            now = datetime.now(IST)
            await self._check_exit(df_today, instrument, now)
            # If still not exited (shouldn't happen — EOD triggers), force it
            if not self._active_trade.exited:
                self._active_trade.exited = True
                self._active_trade.exit_reason = "eod_force"
                if df_today is not None and not df_today.empty:
                    self._active_trade.exit_price = float(df_today.iloc[-1]["close"])
                self._active_trade.exit_time = now.strftime("%H:%M")
                await self.alert_manager.telegram.send(
                    f"🔔 {SCANNER_TAG} — EOD Force Close\n"
                    f"Exit price: {self._active_trade.exit_price:.2f}\n"
                    f"Reason: End of day"
                )

    async def _fetch_option_quote(
        self,
        instrument: InstrumentConfig,
        strike: float,
        option_type: str,
    ) -> Optional[dict]:
        """Fetch live option quote from AngelOne."""
        if not self._expiry:
            return None
        symbol = instrument.build_option_symbol(self._expiry, strike, option_type)
        token_info = self.client._search_symbol(symbol)
        if not token_info:
            logger.warning("[MoveDet] Token not found for %s", symbol)
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
            logger.warning("[MoveDet] Quote fetch timed out for %s", symbol)
            return None
        except Exception:
            logger.exception("[MoveDet] Error fetching quote for %s", symbol)
            return None
