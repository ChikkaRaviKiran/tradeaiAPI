"""14:30 Liquidity-Vacuum Breakout Scanner — PAPER-ONLY.

Strategy summary (backtested on 131 NIFTY days, 2025-10-13 → 2026-04-24):
  - Setup window:  14:25 → 14:55 IST (afternoon liquidity vacuum).
  - Coil filter:   the 5-min bar ending 14:25 must have a range smaller than
                   60% of the day's average 5-min range so far.
  - Entry:         on a 1-min bar between 14:25 and 14:44 IST that breaks
                   the coil-high (CE) or coil-low (PE).
  - Strike:        ATM, nearest weekly expiry.
  - Option SL:     -25% premium.
  - Option TGT:    +40% premium.
  - Time exit:     14:55 IST.

Backtest (option ATM-buying with 1% slippage):
  N=18 trades, WR 66.7%, PF 5.80, total +178 pts/lot (~₹13.4k @ lot=75).

Honoured 5-way mutex with Config-P / Move-Det / NR5 / PDH-PDL — never
opens a new trade while a peer holds an open trade. OBSERVE-ONLY:
sends Telegram alerts but does not write to the paper-trader DB.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, date, time as dtime
from statistics import mean
from typing import Optional

import pandas as pd
import pytz

from app.alerts.alert_manager import AlertManager
from app.core.instruments import InstrumentConfig
from app.core.models import AlertItem
from app.data.angelone_client import AngelOneClient

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Vacuum parameters (LOCKED — matches backtest sweet spot) ──────────
COIL_RANGE_PCT      = 0.60     # coil range must be < 60% of day's avg 5-min range
ARM_TIME            = dtime(14, 25)   # check coil at this 5-min close
WIN_END             = dtime(14, 45)   # last entry bar
ENTRY_LATEST        = dtime(14, 44)   # do not enter on the 14:45 bar
TIME_EXIT           = dtime(14, 55)   # forced exit
OPT_SL_PCT          = 25.0     # -25% premium
OPT_TGT_PCT         = 40.0     # +40% premium

BAR_MINUTES = 5  # for coil resampling


def _resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(f"{BAR_MINUTES}min", label="left", closed="left").agg(agg).dropna(how="any")
    return out


class VacuumTrade:
    """Tracks a single 14:30-Vacuum paper trade from entry to exit."""

    def __init__(
        self,
        trade_date: str,
        side: str,                # 'CE' or 'PE'
        entry_time: str,
        entry_spot: float,
        coil_high: float,
        coil_low: float,
        coil_range: float,
        avg_range: float,
        option_symbol: str,
        option_entry_price: float,
        strike_price: float,
    ):
        self.trade_date = trade_date
        self.side = side
        self.entry_time = entry_time
        self.entry_spot = entry_spot
        self.coil_high = coil_high
        self.coil_low = coil_low
        self.coil_range = coil_range
        self.avg_range = avg_range
        self.option_symbol = option_symbol
        self.option_entry_price = option_entry_price
        self.strike_price = strike_price
        # Computed exit thresholds (premium-based)
        self.option_sl_price = (
            round(option_entry_price * (1 - OPT_SL_PCT / 100.0), 2)
            if option_entry_price > 0 else 0.0
        )
        self.option_target_price = (
            round(option_entry_price * (1 + OPT_TGT_PCT / 100.0), 2)
            if option_entry_price > 0 else 0.0
        )
        # Exit tracking
        self.exited = False
        self.exit_time: str = ""
        self.exit_spot: float = 0.0
        self.exit_reason: str = ""
        self.option_exit_price: float = 0.0


class VacuumBreakoutScanner:
    """14:30-Vacuum breakout scanner — paper-only.

    Lifecycle:
      1. Until 14:25: idle.
      2. At/after 14:25 (and before 14:45): compute the coil bar from the 5-min
         resample. If the coil range < COIL_RANGE_PCT × avg → arm watch.
      3. Watch 1-min bars between 14:25 and 14:44 for a break of coil-high (CE)
         or coil-low (PE). First trigger → entry.
      4. While in trade: each cycle check premium SL / target. Force exit at 14:55.

    Mutex: orchestrator passes ``peer_in_trade=True`` whenever any other
    scanner currently holds an open trade. New entries are skipped in that state.
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
        self._setup_checked = False        # have we computed the coil today?
        self._is_tradeable_day = False     # coil filter passed
        self._coil_high: Optional[float] = None
        self._coil_low: Optional[float] = None
        self._coil_range: float = 0.0
        self._avg_range: float = 0.0
        self._signal_found_today = False
        self._active_trade: Optional[VacuumTrade] = None
        self._expiry: str = ""
        self._expiry_date: Optional[date] = None

    # ── Public API ─────────────────────────────────────────────────

    def reset_daily(self) -> None:
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._setup_checked = False
        self._is_tradeable_day = False
        self._coil_high = None
        self._coil_low = None
        self._coil_range = 0.0
        self._avg_range = 0.0
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
        if not isinstance(df_today.index, pd.DatetimeIndex):
            return

        # Constrain to today's market session
        df_today = df_today.between_time("09:15", "15:30")
        if df_today.empty:
            return

        now = datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")
        if self._today_str != today_str:
            if self._today_str:
                logger.info("[VACUUM] Day rollover: %s → %s — resetting state.",
                            self._today_str, today_str)
            self.reset_daily()
            self._today_str = today_str

        # ── Manage open trade first ───────────────────────────────
        if self._active_trade and not self._active_trade.exited:
            await self._check_exit(df_today, instrument, now)
            return

        # ── Mutex: skip if any peer is in a trade ─────────────────
        if peer_in_trade:
            return

        # ── Done for the day? ─────────────────────────────────────
        if self._signal_found_today:
            return

        # ── Window check ──────────────────────────────────────────
        if now.time() < ARM_TIME:
            return  # too early
        if now.time() > WIN_END:
            # past entry window without firing
            if not self._signal_found_today:
                self._signal_found_today = True
                logger.info("[VACUUM] %s — past entry window %s, no break today",
                            self._today_str, WIN_END)
                if self._is_tradeable_day:
                    await self.alert_manager.telegram.send(
                        f"⏰ 14:30 Vacuum — No Break\n"
                        f"Date: {self._today_str}\n"
                        f"Coil was set ({self._coil_low:.2f} – {self._coil_high:.2f}, "
                        f"range {self._coil_range:.1f}) but never broke by {WIN_END.strftime('%H:%M')}."
                    )
            return

        # ── Setup check (one-shot at first cycle ≥ 14:25) ─────────
        if not self._setup_checked:
            self._compute_coil(df_today)
            self._setup_checked = True
            if self._is_tradeable_day:
                await self.alert_manager.telegram.send(
                    f"📊 14:30 Vacuum — Coil Detected ✅\n"
                    f"Date: {self._today_str}\n"
                    f"Coil high: {self._coil_high:.2f}\n"
                    f"Coil low : {self._coil_low:.2f}\n"
                    f"Range {self._coil_range:.1f} ({100*self._coil_range/self._avg_range:.0f}% of avg {self._avg_range:.1f})\n"
                    f"Watching for 1-min break until {WIN_END.strftime('%H:%M')}."
                )
            else:
                logger.info("[VACUUM] %s — coil filter NOT met (range %.1f vs avg %.1f)",
                            self._today_str, self._coil_range, self._avg_range)

        if not self._is_tradeable_day:
            return

        # ── Watch 1-min bars in (ARM_TIME, ENTRY_LATEST] for a break ──
        forward = df_today[
            (df_today.index.time > ARM_TIME) & (df_today.index.time <= ENTRY_LATEST)
        ]
        if forward.empty:
            return

        side: Optional[str] = None
        entry_spot: float = 0.0
        entry_ts = None
        for ts, row in forward.iterrows():
            if float(row["high"]) > self._coil_high:
                side = "CE"
                entry_spot = self._coil_high + 0.5
                entry_ts = ts
                break
            if float(row["low"]) < self._coil_low:
                side = "PE"
                entry_spot = self._coil_low - 0.5
                entry_ts = ts
                break

        if side is None:
            return

        await self._open_trade(side, entry_spot, entry_ts, instrument, now)

    # ── Setup detection ─────────────────────────────────────────────

    def _compute_coil(self, df_today: pd.DataFrame) -> None:
        """Resample to 5-min, find the bar ending at 14:25, compare its range
        to the average of all 5-min bars up to and including that time."""
        bars_5m = _resample_to_5min(df_today)
        if bars_5m is None or bars_5m.empty:
            return
        # The bar ending at 14:25 is labelled 14:20 (label='left')
        coil_label = bars_5m.index[
            (bars_5m.index.time >= dtime(14, 20)) & (bars_5m.index.time <= dtime(14, 25))
        ]
        if len(coil_label) == 0:
            return
        coil_idx = coil_label[-1]
        coil_bar = bars_5m.loc[coil_idx]
        coil_high = float(coil_bar["high"])
        coil_low = float(coil_bar["low"])
        coil_range = coil_high - coil_low

        bars_until_coil = bars_5m.loc[:coil_idx]
        if len(bars_until_coil) < 6:
            return  # not enough samples
        ranges = (bars_until_coil["high"] - bars_until_coil["low"]).clip(lower=0.01)
        avg_range = float(ranges.mean())

        self._coil_high = coil_high
        self._coil_low = coil_low
        self._coil_range = coil_range
        self._avg_range = avg_range
        self._is_tradeable_day = coil_range < (COIL_RANGE_PCT * avg_range)

    # ── Entry ───────────────────────────────────────────────────────

    async def _open_trade(
        self,
        side: str,
        entry_spot: float,
        entry_ts,
        instrument: InstrumentConfig,
        now: datetime,
    ) -> None:
        entry_time = entry_ts.strftime("%H:%M") if hasattr(entry_ts, "strftime") else now.strftime("%H:%M")
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

        self._active_trade = VacuumTrade(
            trade_date=self._today_str,
            side=side,
            entry_time=entry_time,
            entry_spot=entry_spot,
            coil_high=self._coil_high or 0.0,
            coil_low=self._coil_low or 0.0,
            coil_range=self._coil_range,
            avg_range=self._avg_range,
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
            f"⚡ 14:30 VACUUM — {side} BREAKOUT (PAPER)\n"
            f"{'=' * 36}\n"
            f"\n"
            f"📅 Date: {self._today_str}\n"
            f"⏰ Entry: {entry_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Entry: {entry_spot:.2f}\n"
            f"  Coil:  {self._coil_low:.2f} – {self._coil_high:.2f}\n"
            f"  Coil range: {self._coil_range:.1f} pts "
            f"({100*self._coil_range/max(0.01,self._avg_range):.0f}% of avg {self._avg_range:.1f})\n"
            f"\n"
            f"🎯 OPTION ({side}) — ATM\n"
            f"  Strike: {int(strike)} {side}\n"
            f"  Symbol: {option_symbol}\n"
            f"  Expiry: {expiry_display} ({days_to_expiry}d)\n"
            f"  LTP: ₹{option_ltp:.2f}\n"
            f"  Bid: ₹{option_bid:.2f} | Ask: ₹{option_ask:.2f}\n"
            f"  Spread: {spread_pct:.1f} %\n"
            f"\n"
            f"📋 EXIT RULES\n"
            f"  1. Premium SL : ₹{self._active_trade.option_sl_price:.2f}  (-{OPT_SL_PCT:.0f}%)\n"
            f"  2. Premium TGT: ₹{self._active_trade.option_target_price:.2f}  (+{OPT_TGT_PCT:.0f}%)\n"
            f"  3. Time exit  : {TIME_EXIT.strftime('%H:%M')}\n"
            f"\n"
            f"⚠️ OBSERVE ONLY — paper trade, no execution"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[VACUUM] ENTRY %s @ spot=%.2f opt=%s ₹%.2f sl=₹%.2f tgt=₹%.2f",
            side, entry_spot, option_symbol, option_ltp,
            self._active_trade.option_sl_price, self._active_trade.option_target_price,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=f"VACUUM {side} — NIFTY {int(strike)} {side}",
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

        # Live option quote drives exit (premium-based SL/TGT)
        option_ltp = 0.0
        if trade.option_symbol and self._expiry:
            quote = await self._fetch_option_quote(instrument, trade.strike_price, trade.side)
            if quote:
                option_ltp = float(quote.get("ltp", 0.0) or 0.0)

        exit_reason = ""
        if option_ltp > 0:
            if option_ltp <= trade.option_sl_price:
                exit_reason = "stop"
            elif option_ltp >= trade.option_target_price:
                exit_reason = "target"

        if not exit_reason and now.time() >= TIME_EXIT:
            exit_reason = "time"

        if not exit_reason:
            return

        # Spot at exit (informational)
        last = df_today.iloc[-1]
        exit_spot = float(last["close"])

        trade.exited = True
        trade.exit_time = now.strftime("%H:%M")
        trade.exit_spot = exit_spot
        trade.exit_reason = exit_reason
        trade.option_exit_price = option_ltp

        is_win = (
            option_ltp > trade.option_entry_price
            if trade.option_entry_price > 0 and option_ltp > 0
            else False
        )
        opt_pts = (option_ltp - trade.option_entry_price) if trade.option_entry_price > 0 else 0.0
        opt_pct = (
            opt_pts / trade.option_entry_price * 100
            if trade.option_entry_price > 0 else 0.0
        )

        emoji = "✅" if is_win else "❌"
        msg = (
            f"{emoji} 14:30 VACUUM — EXIT {'WIN' if is_win else 'LOSS'} (PAPER)\n"
            f"{'=' * 36}\n"
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
            f"  Entry: ₹{trade.option_entry_price:.2f}\n"
            f"  Exit : ₹{option_ltp:.2f}\n"
            f"  PnL  : {opt_pts:+.2f} pts ({opt_pct:+.1f}%)\n"
            f"\n"
            f"📋 Exit Reason: {exit_reason}\n"
            f"\n"
            f"⚠️ OBSERVE ONLY — paper trade"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[VACUUM] EXIT %s %s opt=₹%.2f → ₹%.2f (%+.1f%%) reason=%s",
            "WIN" if is_win else "LOSS", trade.side,
            trade.option_entry_price, option_ltp, opt_pct, exit_reason,
        )
        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=f"VACUUM {'WIN' if is_win else 'LOSS'} — {exit_reason}",
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
                    f"🔔 14:30 Vacuum — Force Close\n"
                    f"Exit spot: {self._active_trade.exit_spot:.2f}\n"
                    f"Reason: End of window"
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
            logger.warning("[VACUUM] Token not found for %s", symbol)
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
            logger.warning("[VACUUM] Quote fetch timed out for %s", symbol)
            return None
        except Exception:
            logger.exception("[VACUUM] Error fetching quote for %s", symbol)
            return None
