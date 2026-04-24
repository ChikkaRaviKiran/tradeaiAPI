"""NR5 Breakout Scanner — Volatility Contraction → Expansion (PAPER-ONLY).

Strategy summary (backtested on 131 NIFTY days, 2025-10-13 → 2026-04-24):
  - Setup:  yesterday's daily range = NARROWEST of last 5 trading days (NR5)
  - Filter: skip if abs(today_open vs prev_close) gap > 1.5 %
  - Entry:  break of prev-day high + 2 (CE) or prev-day low − 2 (PE)
  - Stop:   12 pts adverse (spot)
  - Target: 50 pts favorable (spot)  → R:R = 1:4.2
  - EOD:    force exit at 15:20

Backtest result: N=23 trades, WR=43.5 %, PF=3.21, expectancy +14.96 pts,
total +344 pts, MDD -72 pts.

This scanner is OBSERVE-ONLY: it sends Telegram alerts but does NOT
record trades to the paper-trader DB. It also respects a 3-way mutex
with Config-P / Move-Det — it will not enter while any other scanner
holds an open trade, and will block them while it is itself in a trade.
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
from app.core.instruments import InstrumentConfig
from app.core.models import AlertItem
from app.data.angelone_client import AngelOneClient
from app.db.models import AsyncSessionLocal, IndexCandle

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── NR5 Parameters (LOCKED — matches backtest) ────────────────────────
NR_LEN = 5                 # narrow-range lookback (yesterday is the narrowest of last NR_LEN days)
BREAKOUT_BUFFER = 2.0      # spot points beyond prev high/low to trigger entry
STOP_PTS = 12.0            # adverse spot move that stops the trade out
TARGET_PTS = 50.0          # favorable spot move that books target
GAP_MAX_PCT = 1.5          # skip the day if abs(gap %) exceeds this
ENTRY_CUTOFF = dtime(14, 0)
EOD_FORCE_CLOSE = dtime(15, 20)


class NR5Trade:
    """Tracks a single NR5 paper trade from entry to exit."""

    def __init__(
        self,
        trade_date: str,
        side: str,              # 'CE' or 'PE'
        entry_time: str,
        entry_spot: float,
        prev_high: float,
        prev_low: float,
        prev_range: float,
        gap_pct: float,
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
        self.prev_range = prev_range
        self.gap_pct = gap_pct
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


class NR5BreakoutScanner:
    """NR5 breakout scanner — paper-only, runs each minute during market hours.

    Lifecycle:
      1. At 09:14 / first cycle: query DB for last NR_LEN+1 daily ranges to
         determine if today qualifies as an NR5 setup day.
      2. Each cycle (until 14:00): on a tradeable NR5 day, watch latest spot
         vs prev_high / prev_low. First breakout triggers an entry alert.
      3. While in trade: each cycle check stop / target / EOD.
      4. At 15:20: force exit if still open.

    Mutex: orchestrator passes ``peer_in_trade=True`` whenever Config-P or
    Move-Det currently holds an open trade. The scanner skips new entries
    in that state. It also exposes ``is_in_trade()`` so peers can do the same.
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
        self._setup_checked = False
        self._is_nr5_day = False
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._prev_range: Optional[float] = None
        self._gap_pct: Optional[float] = None
        self._gap_skip = False
        self._signal_found_today = False
        self._active_trade: Optional[NR5Trade] = None
        self._expiry: str = ""
        self._expiry_date: Optional[date] = None

    # ── Public API used by orchestrator ─────────────────────────────

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._setup_checked = False
        self._is_nr5_day = False
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._prev_range = None
        self._gap_pct = None
        self._gap_skip = False
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

        now = datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")
        if self._today_str != today_str:
            if self._today_str:
                logger.info(
                    "[NR5] Day rollover: %s → %s. Resetting daily state.",
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
            await self._check_setup(df_today, now)
            self._setup_checked = True

        if not self._is_nr5_day or self._gap_skip:
            return

        # ── Past entry cutoff? Stop scanning ────────────────────────
        if now.time() > ENTRY_CUTOFF:
            self._signal_found_today = True
            logger.info("[NR5] %s — past entry cutoff %s, no breakout today",
                        self._today_str, ENTRY_CUTOFF)
            await self.alert_manager.telegram.send(
                f"⏰ NR5 — Entry Cutoff Passed\n"
                f"Date: {self._today_str}\n"
                f"NR5 day but no breakout by {ENTRY_CUTOFF.strftime('%H:%M')}.\n"
                f"Done for today."
            )
            return

        # ── Look for breakout in the most recent candle ─────────────
        last = df_today.iloc[-1]
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_ts = last.name if isinstance(df_today.index, pd.DatetimeIndex) else last.get("timestamp")

        side: Optional[str] = None
        entry_spot = 0.0
        if self._prev_high is not None and last_high >= self._prev_high + BREAKOUT_BUFFER:
            side = "CE"
            entry_spot = self._prev_high + BREAKOUT_BUFFER
        elif self._prev_low is not None and last_low <= self._prev_low - BREAKOUT_BUFFER:
            side = "PE"
            entry_spot = self._prev_low - BREAKOUT_BUFFER
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

        self._active_trade = NR5Trade(
            trade_date=self._today_str,
            side=side,
            entry_time=entry_time,
            entry_spot=entry_spot,
            prev_high=self._prev_high or 0.0,
            prev_low=self._prev_low or 0.0,
            prev_range=self._prev_range or 0.0,
            gap_pct=self._gap_pct or 0.0,
            stop_level=stop_level,
            target_level=target_level,
            option_symbol=option_symbol,
            option_entry_price=option_ltp,
            strike_price=strike,
        )
        self._signal_found_today = True

        expiry_display = self._expiry or "N/A"
        days_to_expiry = (
            (self._expiry_date - now.date()).days if self._expiry_date else "?"
        )
        msg = (
            f"⚡ NR5 BREAKOUT — {side} SIGNAL (PAPER)\n"
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
            f"  Prev range: {self._prev_range:.2f} pts (NR{NR_LEN})\n"
            f"  Today gap:  {self._gap_pct:+.2f} %\n"
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
            f"⚠️ OBSERVE ONLY — paper trade, no execution"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[NR5] ENTRY %s @ spot=%.2f stop=%.2f tgt=%.2f opt=%s ₹%.2f",
            side, entry_spot, stop_level, target_level, option_symbol, option_ltp,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=f"NR5 BREAKOUT {side} — NIFTY {int(strike)} {side}",
            message=msg,
            timestamp=now,
        )
        await self.alert_manager.record(alert)

    # ── Setup detection ─────────────────────────────────────────────

    async def _check_setup(self, df_today: pd.DataFrame, now: datetime) -> None:
        """Query DB for last NR_LEN+1 daily ranges, confirm NR setup, compute gap."""
        try:
            # Use the database. Pull last (NR_LEN + 5) calendar days of
            # 1-min candles to be safe (covers weekends/holidays) and aggregate
            # to daily H/L/C in Python.
            cutoff = (now.date() - timedelta(days=NR_LEN + 8)).strftime("%Y-%m-%d")
            today_str = now.strftime("%Y-%m-%d")
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(IndexCandle.date, IndexCandle.high, IndexCandle.low, IndexCandle.close)
                    .where(IndexCandle.instrument == "NIFTY")
                    .where(IndexCandle.date >= cutoff)
                    .where(IndexCandle.date < today_str)
                    .order_by(IndexCandle.date, IndexCandle.timestamp)
                )
                rows = (await session.execute(stmt)).all()
        except Exception:
            logger.exception("[NR5] DB lookup for prior daily ranges failed")
            return

        if not rows:
            logger.warning("[NR5] No prior daily candles found in DB; cannot evaluate NR%d", NR_LEN)
            return

        # Aggregate to daily H/L/C
        per_day: dict[str, list[tuple[float, float, float]]] = {}
        for d, h, l, c in rows:
            per_day.setdefault(d, []).append((float(h), float(l), float(c)))
        daily = []
        for d in sorted(per_day.keys()):
            highs = [r[0] for r in per_day[d]]
            lows = [r[1] for r in per_day[d]]
            close = per_day[d][-1][2]
            daily.append((d, max(highs), min(lows), close, max(highs) - min(lows)))

        if len(daily) < NR_LEN:
            logger.info("[NR5] Not enough prior daily data: have %d, need %d", len(daily), NR_LEN)
            return

        last_n = daily[-NR_LEN:]
        ranges = [r[4] for r in last_n]
        yest = last_n[-1]
        is_nr = yest[4] == min(ranges)

        self._prev_high = yest[1]
        self._prev_low = yest[2]
        self._prev_close = yest[3]
        self._prev_range = yest[4]

        # Gap %
        if not df_today.empty and self._prev_close:
            today_open = float(df_today.iloc[0]["open"])
            self._gap_pct = (today_open - self._prev_close) / self._prev_close * 100
        else:
            self._gap_pct = 0.0

        if not is_nr:
            self._is_nr5_day = False
            logger.info(
                "[NR5] %s NOT an NR%d day. yest_range=%.2f, last %d ranges=%s",
                self._today_str, NR_LEN, yest[4], NR_LEN,
                [round(r, 2) for r in ranges],
            )
            return

        if abs(self._gap_pct) > GAP_MAX_PCT:
            self._is_nr5_day = True
            self._gap_skip = True
            logger.info(
                "[NR5] %s NR%d setup but gap %.2f%% > %.2f%% — skip day",
                self._today_str, NR_LEN, self._gap_pct, GAP_MAX_PCT,
            )
            await self.alert_manager.telegram.send(
                f"📊 NR5 — Gap Skip\n"
                f"Date: {self._today_str}\n"
                f"NR{NR_LEN} setup ✅ but gap {self._gap_pct:+.2f}% > ±{GAP_MAX_PCT}%.\n"
                f"No trade today."
            )
            return

        self._is_nr5_day = True
        logger.info(
            "[NR5] %s TRADEABLE: NR%d (range=%.2f, last %d=%s), gap=%+.2f%%, prev_H=%.2f prev_L=%.2f",
            self._today_str, NR_LEN, yest[4], NR_LEN,
            [round(r, 2) for r in ranges], self._gap_pct,
            self._prev_high, self._prev_low,
        )
        await self.alert_manager.telegram.send(
            f"📊 NR5 — Day Tradeable ✅\n"
            f"Date: {self._today_str}\n"
            f"NR{NR_LEN} setup confirmed (yest range = narrowest of last {NR_LEN}).\n"
            f"Prev H: {self._prev_high:.2f}\n"
            f"Prev L: {self._prev_low:.2f}\n"
            f"Prev range: {self._prev_range:.2f} pts\n"
            f"Gap: {self._gap_pct:+.2f}%\n"
            f"Watching for break of prev H+{BREAKOUT_BUFFER:.0f} (CE) or prev L−{BREAKOUT_BUFFER:.0f} (PE)\n"
            f"until {ENTRY_CUTOFF.strftime('%H:%M')}."
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
        msg = (
            f"{emoji} NR5 — EXIT {'WIN' if is_win else 'LOSS'} (PAPER)\n"
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
            f"⚠️ OBSERVE ONLY — paper trade"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[NR5] EXIT %s %s @ spot=%.2f pnl=%+.2f pts (reason=%s)",
            "WIN" if is_win else "LOSS", trade.side, exit_spot, spot_pnl, exit_reason,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=f"NR5 {'WIN' if is_win else 'LOSS'} — {exit_reason}",
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
                    f"🔔 NR5 — EOD Force Close\n"
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
            logger.warning("[NR5] Token not found for %s", symbol)
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
            logger.warning("[NR5] Quote fetch timed out for %s", symbol)
            return None
        except Exception:
            logger.exception("[NR5] Error fetching quote for %s", symbol)
            return None
