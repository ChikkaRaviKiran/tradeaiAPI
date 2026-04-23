"""AI-GPT Scanner — third independent scanner using a 3-stage GPT pipeline.

Runs alongside ConfigPScanner and MoveDetectionScanner. Independent state,
independent trade tracking, independent Telegram alerts. Does NOT share
weekly filters or active-trade slots with the other scanners.

Cadence:
  - orchestrator calls run_cycle every 1 min
  - this scanner only invokes the AI pipeline once every 5 min (on minute
    marks 0/5/10/.../55), to control cost and match the spec's "every 5 min"
  - in-trade monitoring also runs on the 5-min cadence

Pipeline (per cycle, when allowed):
  1. Build 5-min payload from 1-min df_today (resample OHLCV)
  2. Interpreter → Reasoning → Validator
  3. ENTER iff confidence >= MIN_CONFIDENCE AND validator says APPROVED
  4. While in trade: AI monitor every 5 min → EXIT / PARTIAL_EXIT / HOLD
  5. EOD 15:20 force close

Risk controls (per spec):
  - Max 1 open trade in this engine
  - Skip trade on any AI failure
  - No new entries after NO_NEW_ENTRY_TIME (15:00)
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
from app.ai.ai_gpt_pipeline import AIGPTPipeline
from app.core.instruments import InstrumentConfig
from app.core.models import AlertItem
from app.data.angelone_client import AngelOneClient
from app.engine.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Parameters ──────────────────────────────────────────────────────
MIN_CONFIDENCE = 75              # Spec: confidence >= 75 required
EARLIEST_TIME = dtime(9, 35)     # First AI run after 9:35 (need ≥4 5-min candles)
NO_NEW_ENTRY_TIME = dtime(15, 0)
EOD_FORCE_CLOSE_TIME = dtime(15, 20)
OPTION_LEVERAGE = 55             # ATM premium leverage estimate (matches other scanners)
SCANNER_TAG = "AI-GPT"
AI_CYCLE_MINUTES = 5             # Run AI pipeline every N minutes


class ActiveTrade:
    """Tracks an active AI-GPT trade from entry through exit."""

    def __init__(
        self,
        trade_date: str,
        entry_time: str,
        entry_spot: float,
        direction: str,             # "bullish" / "bearish"
        option_type: str,           # "CE" / "PE"
        confidence: int,
        setup_quality: str,
        expected_move: str,
        bias_reason: str,
        validator_reason: str,
        option_symbol: str = "",
        option_entry_price: float = 0.0,
        strike_price: float = 0.0,
    ):
        self.trade_date = trade_date
        self.entry_time = entry_time
        self.entry_spot = entry_spot
        self.direction = direction
        self.option_type = option_type
        self.confidence = confidence
        self.setup_quality = setup_quality
        self.expected_move = expected_move
        self.bias_reason = bias_reason
        self.validator_reason = validator_reason
        self.option_symbol = option_symbol
        self.option_entry_price = option_entry_price
        self.strike_price = strike_price

        # Exit state
        self.best_spot = entry_spot
        self.exited = False
        self.exit_spot: float = 0.0
        self.exit_time: str = ""
        self.exit_reason: str = ""
        self.option_exit_price: float = 0.0


class AIGPTScanner:
    """Third independent scanner — GPT-driven 3-stage decision pipeline.

    Lifecycle:
      1. Each minute: run_cycle invoked by orchestrator
      2. If we hold an open trade: every 5 min, ask AI monitor → maybe exit
      3. Else if minute mark is on 5-min boundary AND past EARLIEST_TIME:
         build payload → interpret → reason → validate → maybe enter
      4. At 15:20: force close any open trade
    """

    def __init__(
        self,
        client: AngelOneClient,
        feature_engine: FeatureEngine,
        alert_manager: AlertManager,
        pipeline: Optional[AIGPTPipeline],
    ):
        self.client = client
        self.fe = feature_engine
        self.alert_manager = alert_manager
        self.pipeline = pipeline   # If None → scanner is disabled (no key/model)

        # Daily state
        self._active_trade: Optional[ActiveTrade] = None
        self._today_str: str = ""
        self._expiry: str = ""
        self._expiry_date: Optional[date] = None
        self._last_ai_run_minute: Optional[int] = None  # to enforce 5-min cadence
        self._last_ai_run_date: Optional[str] = None
        self._ai_failure_count_today: int = 0
        # Visibility into AI activity (surfaced via /api/system/status)
        self._last_run_at: Optional[str] = None        # "HH:MM" of most recent AI cycle
        self._last_decision: Optional[str] = None      # short label (e.g. "no_trade", "low_conf:60", "rejected", "failed", "signal CE", "hold")
        self._last_decision_detail: Optional[str] = None
        self._ai_runs_today: int = 0

    def reset_daily(self) -> None:
        """Reset daily state for a fresh trading day."""
        self._active_trade = None
        self._today_str = datetime.now(IST).strftime("%Y-%m-%d")
        self._last_ai_run_minute = None
        self._last_ai_run_date = None
        self._ai_failure_count_today = 0
        self._last_run_at = None
        self._last_decision = None
        self._last_decision_detail = None
        self._ai_runs_today = 0

    async def _record_cycle(
        self,
        now: datetime,
        decision: str,
        detail: str,
        *,
        emit_alert: bool = True,
    ) -> None:
        """Track AI activity for status/UI visibility and (optionally) push a
        compact info alert so users can see the scanner is alive each cycle.
        Telegram is NOT spammed — record() only writes to UI store + DB.
        """
        self._last_run_at = now.strftime("%H:%M")
        self._last_decision = decision
        self._last_decision_detail = detail
        self._ai_runs_today += 1
        logger.info("[%s] cycle %s — %s | %s", SCANNER_TAG, self._last_run_at, decision, detail)
        if not emit_alert:
            return
        try:
            alert = AlertItem(
                id=str(uuid.uuid4())[:8],
                alert_type="info",
                title=f"{SCANNER_TAG} cycle {self._last_run_at} — {decision}",
                message=detail,
                timestamp=now,
            )
            await self.alert_manager.record(alert)
        except Exception:
            logger.debug("[%s] failed to record cycle alert", SCANNER_TAG, exc_info=True)

    def set_expiry(self, expiry: str, expiry_date: Optional[date] = None) -> None:
        self._expiry = expiry
        self._expiry_date = expiry_date

    # ── Main entry point (called every 1 min by orchestrator) ──────
    async def run_cycle(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        cycle: int,
    ) -> None:
        if self.pipeline is None:
            return  # Disabled
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
                    "[%s] Day rollover detected: %s → %s. Resetting daily state.",
                    SCANNER_TAG, self._today_str, today_str,
                )
            self.reset_daily()
            self._today_str = today_str

        # ── Gate: only run AI on 5-min boundaries ───────────────────
        if not self._is_ai_cycle(now):
            return

        # ── If holding a trade: monitor for exit ────────────────────
        if self._active_trade and not self._active_trade.exited:
            await self._monitor_trade(df_today, instrument, now)
            return

        # ── Entry path ──────────────────────────────────────────────
        if now.time() < EARLIEST_TIME:
            return  # too early; not enough 5-min candles
        if now.time() >= NO_NEW_ENTRY_TIME:
            return  # late session — no new entries
        if self._ai_failure_count_today >= 5:
            return  # too many AI failures today, stop trying

        try:
            await self._try_enter(df_today, instrument, now)
        except Exception:
            logger.exception("[%s] _try_enter raised", SCANNER_TAG)

    # ── 5-min cadence gate ──────────────────────────────────────────
    def _is_ai_cycle(self, now: datetime) -> bool:
        """True at most once per 5-min slot."""
        if now.minute % AI_CYCLE_MINUTES != 0:
            return False
        slot_key = now.minute
        date_key = now.strftime("%Y-%m-%d")
        if (
            self._last_ai_run_minute == slot_key
            and self._last_ai_run_date == date_key
        ):
            return False
        self._last_ai_run_minute = slot_key
        self._last_ai_run_date = date_key
        return True

    # ── Build 5-min payload from 1-min df ───────────────────────────
    def _build_payload(self, df_today: pd.DataFrame, now: datetime) -> Optional[dict]:
        """Resample to 5-min and build the spec-shaped AI input payload."""
        if not isinstance(df_today.index, pd.DatetimeIndex):
            return None
        ohlc = df_today[["open", "high", "low", "close"]].resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(ohlc) < 4:
            return None

        vol = None
        if "volume" in df_today.columns:
            vol = df_today["volume"].resample("5min").sum().reindex(ohlc.index).fillna(0)

        last_n = ohlc.tail(15)
        candles = []
        for ts, row in last_n.iterrows():
            entry = {
                "time": ts.strftime("%H:%M"),
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
            }
            if vol is not None:
                entry["v"] = int(vol.loc[ts]) if ts in vol.index else 0
            candles.append(entry)

        # Indicators on 1-min frame (more accurate); pull last value
        last_min = df_today.iloc[-1]
        vwap = float(last_min["vwap"]) if "vwap" in df_today.columns and not pd.isna(last_min.get("vwap")) else None
        ema9 = float(last_min["ema9"]) if "ema9" in df_today.columns and not pd.isna(last_min.get("ema9")) else None
        ema20 = float(last_min["ema20"]) if "ema20" in df_today.columns and not pd.isna(last_min.get("ema20")) else None

        cur_vol = 0
        avg_vol = 0
        if vol is not None and len(vol) >= 11:
            cur_vol = int(vol.iloc[-1])
            avg_vol = int(vol.iloc[-11:-1].mean())

        payload: dict = {
            "candles": candles,
            "vwap": round(vwap, 2) if vwap is not None else None,
            "ema": {
                "ema9": round(ema9, 2) if ema9 is not None else None,
                "ema20": round(ema20, 2) if ema20 is not None else None,
            },
            "volume": {"current": cur_vol, "average": avg_vol},
            "time": now.strftime("%H:%M"),
        }
        return payload

    # ── Entry attempt ───────────────────────────────────────────────
    async def _try_enter(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        now: datetime,
    ) -> None:
        payload = self._build_payload(df_today, now)
        if payload is None:
            await self._record_cycle(now, "skipped", "insufficient candles to build payload", emit_alert=False)
            return

        interpretation = await self.pipeline.interpret(payload)
        if interpretation is None:
            self._ai_failure_count_today += 1
            await self._record_cycle(
                now, "AI FAILED",
                f"interpret stage failed (fail {self._ai_failure_count_today}/5)",
            )
            return

        reasoning = await self.pipeline.reason(payload, interpretation)
        if reasoning is None:
            self._ai_failure_count_today += 1
            await self._record_cycle(
                now, "AI FAILED",
                f"reasoning stage failed (fail {self._ai_failure_count_today}/5)",
            )
            return

        # Hard filters
        trade_side = str(reasoning.get("trade", "NONE")).upper()
        bias = str(reasoning.get("bias", "?"))
        phase = str(interpretation.get("market_phase", "?"))
        momentum = str(interpretation.get("momentum", "?"))
        if trade_side not in ("CE", "PE"):
            await self._record_cycle(
                now, "no trade",
                f"bias={bias} | phase={phase} | momentum={momentum}",
            )
            return

        try:
            confidence = int(reasoning.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        if confidence < MIN_CONFIDENCE:
            logger.info("[%s] Reasoning conf=%d < %d — skip", SCANNER_TAG, confidence, MIN_CONFIDENCE)
            await self._record_cycle(
                now, f"low confidence ({confidence}<{MIN_CONFIDENCE})",
                f"trade={trade_side} | bias={bias} | phase={phase}",
            )
            return

        validation = await self.pipeline.validate(payload, interpretation, reasoning)
        if validation is None:
            self._ai_failure_count_today += 1
            await self._record_cycle(
                now, "AI FAILED",
                f"validator stage failed (fail {self._ai_failure_count_today}/5)",
            )
            return
        if str(validation.get("decision", "")).upper() != "APPROVED":
            logger.info(
                "[%s] Validator rejected: %s",
                SCANNER_TAG, validation.get("reason", "no reason"),
            )
            await self._record_cycle(
                now, "validator REJECTED",
                f"trade={trade_side} conf={confidence} | reason={validation.get('reason', '?')}",
            )
            return

        # ── Place (paper) trade ─────────────────────────────────────
        await self._record_cycle(
            now, f"SIGNAL {trade_side}",
            f"conf={confidence} | bias={bias} | risk={validation.get('risk', '?')}",
            emit_alert=False,  # entry path emits its own rich alert
        )
        await self._enter_trade(
            df_today, instrument, now,
            interpretation, reasoning, validation,
        )

    async def _enter_trade(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        now: datetime,
        interpretation: dict,
        reasoning: dict,
        validation: dict,
    ) -> None:
        spot = float(df_today.iloc[-1]["close"])
        option_type = str(reasoning["trade"]).upper()
        direction = "bullish" if option_type == "CE" else "bearish"
        strike = instrument.nearest_strike(spot, option_type)

        option_symbol = ""
        option_ltp = 0.0
        option_bid = 0.0
        option_ask = 0.0
        spread_pct = 0.0
        if self._expiry:
            option_symbol = instrument.build_option_symbol(self._expiry, strike, option_type)
            quote = await self._fetch_option_quote(instrument, strike, option_type)
            if quote:
                option_ltp = quote.get("ltp", 0.0)
                option_bid = quote.get("best_bid", 0.0)
                option_ask = quote.get("best_ask", 0.0)
                spread_pct = quote.get("spread_pct", 0.0)

        self._active_trade = ActiveTrade(
            trade_date=self._today_str,
            entry_time=now.strftime("%H:%M"),
            entry_spot=spot,
            direction=direction,
            option_type=option_type,
            confidence=int(reasoning.get("confidence", 0)),
            setup_quality=str(reasoning.get("setup_quality", "?")),
            expected_move=str(reasoning.get("expected_move", "?")),
            bias_reason=str(interpretation.get("market_phase", "?"))
                + " / " + str(interpretation.get("momentum", "?")),
            validator_reason=str(validation.get("reason", "?")),
            option_symbol=option_symbol,
            option_entry_price=option_ltp,
            strike_price=strike,
        )

        expiry_display = self._expiry if self._expiry else "N/A"
        days_to_expiry = (self._expiry_date - now.date()).days if self._expiry_date else "?"
        emoji = "🟢" if direction == "bullish" else "🔴"
        msg = (
            f"{emoji} {SCANNER_TAG} — {direction.upper()} SIGNAL\n"
            f"{'='*35}\n"
            f"\n"
            f"📅 Date: {self._today_str}\n"
            f"⏰ Entry: {self._active_trade.entry_time}\n"
            f"\n"
            f"🤖 AI DECISION\n"
            f"  Bias: {reasoning.get('bias', '?')}\n"
            f"  Setup quality: {self._active_trade.setup_quality}\n"
            f"  Confidence: {self._active_trade.confidence}/100\n"
            f"  Expected move: {self._active_trade.expected_move}\n"
            f"  Validator: APPROVED ({validation.get('risk', '?')})\n"
            f"  Reason: {self._active_trade.validator_reason}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Entry: {spot:.2f}\n"
            f"  Trend: {interpretation.get('trend', '?')} / "
            f"{interpretation.get('strength', '?')}\n"
            f"  Phase: {interpretation.get('market_phase', '?')}\n"
            f"  Momentum: {interpretation.get('momentum', '?')}\n"
            f"\n"
            f"🎯 OPTION ({option_type})\n"
            f"  Strike: {int(strike)} {option_type}\n"
            f"  Symbol: {option_symbol}\n"
            f"  Expiry: {expiry_display} ({days_to_expiry}d)\n"
            f"  LTP: ₹{option_ltp:.2f}\n"
            f"  Bid: ₹{option_bid:.2f} | Ask: ₹{option_ask:.2f}\n"
            f"  Spread: {spread_pct:.1f}%\n"
            f"\n"
            f"📋 EXIT RULES\n"
            f"  AI monitor every 5 min → HOLD / EXIT / PARTIAL_EXIT\n"
            f"  EOD: 15:20 force close\n"
            f"\n"
            f"⚠️ OBSERVE ONLY — No auto-execution"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[%s] ENTRY: %s %s %d @ ₹%.2f conf=%d",
            SCANNER_TAG, direction, option_type, int(strike), option_ltp,
            self._active_trade.confidence,
        )

        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="signal",
            title=f"{SCANNER_TAG} {direction.upper()} — NIFTY {int(strike)} {option_type}",
            message=msg,
            timestamp=now,
        )
        await self.alert_manager.record(alert)

    # ── Monitor open trade ──────────────────────────────────────────
    async def _monitor_trade(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        now: datetime,
    ) -> None:
        trade = self._active_trade
        if trade is None or trade.exited:
            return

        # EOD force close first (cheaper than calling AI)
        if now.time() >= EOD_FORCE_CLOSE_TIME:
            await self._exit_trade(df_today, instrument, now, "eod_close")
            return

        payload = self._build_payload(df_today, now)
        if payload is None:
            return
        spot = float(df_today.iloc[-1]["close"])
        # Track best spot in trade direction
        if trade.direction == "bullish":
            trade.best_spot = max(trade.best_spot, spot)
        else:
            trade.best_spot = min(trade.best_spot, spot)

        trade_context = {
            "direction": trade.direction,
            "option_type": trade.option_type,
            "entry_spot": trade.entry_spot,
            "entry_time": trade.entry_time,
            "current_spot": round(spot, 2),
            "spot_move_pts": round(
                spot - trade.entry_spot if trade.direction == "bullish"
                else trade.entry_spot - spot, 2,
            ),
            "best_spot": round(trade.best_spot, 2),
            "expected_move": trade.expected_move,
            "entry_confidence": trade.confidence,
        }

        decision = await self.pipeline.monitor(payload, trade_context)
        if decision is None:
            self._ai_failure_count_today += 1
            await self._record_cycle(
                now, "AI FAILED",
                f"monitor stage failed (fail {self._ai_failure_count_today}/5)",
            )
            return

        action = str(decision.get("action", "HOLD")).upper()
        mon_reason = str(decision.get("reason", ""))
        if action == "EXIT" or action == "PARTIAL_EXIT":
            await self._record_cycle(
                now, action,
                f"{trade.option_type} | spot_move={trade_context['spot_move_pts']}pts | {mon_reason}",
                emit_alert=False,  # exit path emits its own rich alert
            )
            reason = f"ai_{action.lower()}: {mon_reason}"
            await self._exit_trade(df_today, instrument, now, reason)
        else:
            await self._record_cycle(
                now, "HOLD",
                f"{trade.option_type} | spot_move={trade_context['spot_move_pts']}pts | {mon_reason}",
            )

    # ── Exit trade ──────────────────────────────────────────────────
    async def _exit_trade(
        self,
        df_today: pd.DataFrame,
        instrument: InstrumentConfig,
        now: datetime,
        reason: str,
    ) -> None:
        trade = self._active_trade
        if trade is None or trade.exited:
            return

        spot = float(df_today.iloc[-1]["close"])
        trade.exited = True
        trade.exit_spot = spot
        trade.exit_time = now.strftime("%H:%M")
        trade.exit_reason = reason

        # PnL (bullish: profit when spot rises; bearish: when spot drops)
        if trade.direction == "bullish":
            spot_pnl_pts = spot - trade.entry_spot
        else:
            spot_pnl_pts = trade.entry_spot - spot
        spot_pnl_pct = spot_pnl_pts / trade.entry_spot * 100 if trade.entry_spot else 0
        option_pnl_pct = spot_pnl_pct * OPTION_LEVERAGE
        is_win = spot_pnl_pct > 0

        option_exit_ltp = 0.0
        if trade.option_symbol and self._expiry:
            quote = await self._fetch_option_quote(instrument, trade.strike_price, trade.option_type)
            if quote:
                option_exit_ltp = quote.get("ltp", 0.0)
                trade.option_exit_price = option_exit_ltp

        actual_option_pnl = ""
        if trade.option_entry_price > 0 and option_exit_ltp > 0:
            actual_pnl_pct = (
                (option_exit_ltp - trade.option_entry_price)
                / trade.option_entry_price * 100
            )
            actual_option_pnl = (
                f"\n  Actual option PnL: {actual_pnl_pct:+.1f}% "
                f"(₹{trade.option_entry_price:.2f} → ₹{option_exit_ltp:.2f})"
            )

        emoji = "✅" if is_win else "❌"
        msg = (
            f"{emoji} {SCANNER_TAG} — EXIT {'WIN' if is_win else 'LOSS'}\n"
            f"{'='*35}\n"
            f"\n"
            f"📅 Date: {trade.trade_date}\n"
            f"⏰ Entry: {trade.entry_time} → Exit: {trade.exit_time}\n"
            f"\n"
            f"📊 SPOT (NIFTY)\n"
            f"  Entry: {trade.entry_spot:.2f}\n"
            f"  Exit: {spot:.2f}\n"
            f"  PnL: {spot_pnl_pts:+.2f} pts ({spot_pnl_pct:+.3f}%)\n"
            f"  Best: {trade.best_spot:.2f}\n"
            f"\n"
            f"🎯 OPTION ({trade.option_type})\n"
            f"  {trade.option_symbol}\n"
            f"  Entry LTP: ₹{trade.option_entry_price:.2f}\n"
            f"  Exit LTP: ₹{option_exit_ltp:.2f}\n"
            f"  Est. option PnL: {option_pnl_pct:+.1f}% (55× leverage)"
            f"{actual_option_pnl}\n"
            f"\n"
            f"📋 Exit Reason: {reason}\n"
            f"  Entry confidence was: {trade.confidence}/100\n"
            f"  Setup: {trade.setup_quality}\n"
            f"\n"
            f"⚠️ OBSERVE ONLY"
        )
        await self.alert_manager.telegram.send(msg)
        logger.info(
            "[%s] EXIT %s: %s pnl=%+.2fpts (%+.3f%%)",
            SCANNER_TAG, "WIN" if is_win else "LOSS",
            reason, spot_pnl_pts, spot_pnl_pct,
        )

        alert = AlertItem(
            id=str(uuid.uuid4())[:8],
            alert_type="exit",
            title=f"{SCANNER_TAG} {'WIN' if is_win else 'LOSS'} — {reason}",
            message=msg,
            timestamp=now,
        )
        await self.alert_manager.record(alert)

    async def force_close(self, df_today: pd.DataFrame, instrument: InstrumentConfig) -> None:
        """Force close any active trade (called at EOD by orchestrator)."""
        if self._active_trade and not self._active_trade.exited:
            now = datetime.now(IST)
            await self._exit_trade(df_today, instrument, now, "eod_force")

    # ── Option quote fetch (mirror MoveDet) ────────────────────────
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
            logger.warning("[%s] Token not found for %s", SCANNER_TAG, symbol)
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
            logger.warning("[%s] Quote fetch timed out for %s", SCANNER_TAG, symbol)
            return None
        except Exception:
            logger.exception("[%s] Error fetching quote for %s", SCANNER_TAG, symbol)
            return None
