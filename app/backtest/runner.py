"""MOB-only backtest runner for UI API.

Runs exclusively the Momentum Option Buying (MOB) strategy.
All other strategies are disabled.

Provides async job management, progress tracking, and Excel export.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import settings
from app.core.holidays import compute_weekly_expiry
from app.core.instruments import get_enabled_instruments, get_instrument
from app.core.models import OptionsMetrics
from app.engine.feature_engine import FeatureEngine
from app.strategies.momentum_option_buying import MomentumOptionBuyingStrategy, is_range_bound_session
from app.strategies.orb_vwap import ORBVWAPStrategy
from app.analysis.engine import StrategyTester, TradeResult as AnalysisTradeResult, TimeWindow
from app.analysis.day_types import EnhancedDayClassifier

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Strategies available for backtesting
ALL_STRATEGIES = {
    "RANGE_BREAKOUT": None,     # Handled via _run_analysis_backtest_task
    "EMA_BREAKOUT": None,       # Handled via _run_analysis_backtest_task
    "MOMENTUM_BREAKOUT": None,  # Handled via _run_analysis_backtest_task
    "ALL_THREE": None,          # Combined: runs all 3 strategies as portfolio
}

# ── MOB parameters (from settings, overridable via env vars) ──────────

MOB_MAX_TRADES_PER_DAY = settings.mob_max_trades_per_day
MOB_CONSECUTIVE_LOSS_STOP = settings.mob_consecutive_loss_stop
MOB_SLIPPAGE_PCT = settings.mob_slippage_pct
MOB_SL_PCT = settings.mob_sl_pct
MOB_BROKERAGE_PER_LOT = settings.mob_brokerage_per_lot
EOD_EXIT_TIME = dtime(settings.mob_eod_exit_hour, settings.mob_eod_exit_minute)
STARTING_CAPITAL = settings.mob_starting_capital


# ── Result types ──────────────────────────────────────────────────────

@dataclass
class BacktestProgress:
    job_id: str
    status: str = "pending"  # pending | loading | running | completed | failed
    total_days: int = 0
    processed_days: int = 0
    current_date: str = ""
    message: str = ""
    error: str = ""


@dataclass
class BacktestResult:
    job_id: str
    start_date: str
    end_date: str
    instruments: list[str]
    initial_capital: float
    ending_capital: float
    total_pnl: float
    return_pct: float
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    config_used: dict = field(default_factory=dict)


# ── Active jobs registry ──────────────────────────────────────────────

_active_jobs: dict[str, BacktestProgress] = {}
_job_results: dict[str, BacktestResult] = {}


def get_job_progress(job_id: str) -> Optional[BacktestProgress]:
    return _active_jobs.get(job_id)


def get_job_result(job_id: str) -> Optional[BacktestResult]:
    return _job_results.get(job_id)


def list_jobs() -> list[dict]:
    jobs = []
    for jid, prog in _active_jobs.items():
        entry = {
            "job_id": jid,
            "status": prog.status,
            "total_days": prog.total_days,
            "processed_days": prog.processed_days,
            "message": prog.message,
        }
        res = _job_results.get(jid)
        if res:
            entry.update({
                "start_date": res.start_date,
                "end_date": res.end_date,
                "total_pnl": res.total_pnl,
                "return_pct": res.return_pct,
                "total_trades": res.total_trades,
                "win_rate": res.win_rate,
            })
        jobs.append(entry)
    return jobs


# ── Data Loaders ──────────────────────────────────────────────────────

async def _load_index_candles(engine, symbol, start_date, end_date):
    """Load 1-min index candles. Returns {date_str: DataFrame}."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text(
                "SELECT date, timestamp, open, high, low, close, volume "
                "FROM index_candles "
                "WHERE instrument = :inst AND date >= :s AND date <= :e "
                "ORDER BY timestamp"
            ),
            {"inst": symbol, "s": start_date, "e": end_date},
        )
        rows = result.fetchall()
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["date", "timestamp", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    daily = {}
    for ds, grp in df.groupby("date"):
        grp = grp.drop(columns=["date"])
        if len(grp) >= 30:
            daily[ds] = grp
    return daily


async def _load_option_candles_batch(engine, symbol, dates):
    """Pre-load option candles for given dates."""
    if not dates:
        return {}
    cache = {}
    batch_size = 10
    for i in range(0, len(dates), batch_size):
        batch = dates[i:i + batch_size]
        placeholders = ", ".join(f":d{j}" for j in range(len(batch)))
        params = {"inst": symbol}
        for j, d in enumerate(batch):
            params[f"d{j}"] = d
        async with AsyncSession(engine) as session:
            result = await session.execute(
                text(
                    f"SELECT date, strike, option_type, timestamp, open, high, low, close, volume "
                    f"FROM option_candles "
                    f"WHERE instrument = :inst AND date IN ({placeholders}) "
                    f"ORDER BY date, strike, option_type, timestamp"
                ),
                params,
            )
            rows = result.fetchall()
        groups = defaultdict(list)
        for row in rows:
            key = (symbol, str(row[0]), float(row[1]), row[2])
            groups[key].append(row[3:])
        for key, candle_rows in groups.items():
            odf = pd.DataFrame(candle_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                odf[c] = pd.to_numeric(odf[c], errors="coerce")
            odf["timestamp"] = pd.to_datetime(odf["timestamp"])
            odf.set_index("timestamp", inplace=True)
            if odf.index.tz is None:
                odf.index = odf.index.tz_localize("Asia/Kolkata")
            cache[key] = odf
    return cache


# ── MOB Exit Engine (v2 — Enhanced) ───────────────────────────────────
# Changes from v1:
#   1. Partial exit at T1: book 50% at T1, rest continues to T2/trail
#   2. Time-based exit: if no T1 within MAX_HOLD_BARS, exit to avoid decay
#   3. Tighter trail: 2-candle low instead of 3 after T2
#   4. Exit slippage: model realistic slippage on exit
#   5. Smart EOD: exit losers at 15:05, let winners run to 15:10
# ─────────────────────────────────────────────────────────────────────

MAX_HOLD_BARS = settings.mob_max_hold_bars  # Exit if no T1 hit within N candles
EXIT_SLIPPAGE_PCT = settings.mob_exit_slippage_pct / 100  # Convert % to decimal
EOD_EXIT_LOSER = dtime(15, 5)  # Exit losers 5 min early
T1_PARTIAL_PCT = settings.mob_t1_partial_pct  # Book this fraction at T1


def _run_mob_exit(opt_df, entry_opt_idx, entry_price, sl, t1, t2, one_r):
    """MOB exit v2: partial at T1, tight trail after T2, time-based exit.

    Returns (exit_price, exit_reason, exit_time, partial_exit_info).
    partial_exit_info is a dict with {"price", "bars"} if T1 partial occurred.
    """
    t1_hit = False
    t2_hit = False
    recent_lows = []
    bars_since_entry = 0
    partial_exit = None  # Track partial exit at T1

    for idx in range(entry_opt_idx + 1, len(opt_df)):
        bar = opt_df.iloc[idx]
        bar_time = opt_df.index[idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bars_since_entry += 1

        bt = bar_time.time() if hasattr(bar_time, "time") else bar_time.to_pydatetime().time()

        # Smart EOD: exit losers early, let winners run
        is_losing = bar_close < entry_price
        if is_losing and bt >= EOD_EXIT_LOSER:
            exit_p = bar_close * (1 - EXIT_SLIPPAGE_PCT)
            return exit_p, "eod_close", bar_time, partial_exit
        if bt >= EOD_EXIT_TIME:
            exit_p = bar_close * (1 - EXIT_SLIPPAGE_PCT)
            return exit_p, "eod_close", bar_time, partial_exit

        # Stoploss check (with exit slippage)
        if bar_low <= sl:
            reason = "trailing_sl" if (t1_hit or t2_hit) else "stoploss"
            exit_p = sl * (1 - EXIT_SLIPPAGE_PCT)
            return exit_p, reason, bar_time, partial_exit

        # Time-based exit: avoid decay if no T1 within MAX_HOLD_BARS
        if not t1_hit and bars_since_entry >= MAX_HOLD_BARS:
            exit_p = bar_close * (1 - EXIT_SLIPPAGE_PCT)
            return exit_p, "time_exit", bar_time, partial_exit

        # T2 (+2R) → lock 1R profit, enable tight trailing
        if not t2_hit and bar_high >= t2:
            t2_hit = True
            t1_hit = True
            sl = max(sl, entry_price + one_r)
            if partial_exit is None:
                partial_exit = {"price": t1, "bars": bars_since_entry}

        # T1 (+1R) → partial exit + move SL to cost + 1%
        if not t1_hit and bar_high >= t1:
            t1_hit = True
            sl = max(sl, entry_price * 1.01)  # Raised from 0.5% to 1% buffer
            partial_exit = {"price": t1, "bars": bars_since_entry}

        # Trail after T2 using 2-candle low (tighter than 3-candle)
        if t2_hit and len(recent_lows) >= 2:
            trail_level = min(recent_lows[-2:])
            if trail_level > sl:
                sl = trail_level

        recent_lows.append(bar_low)

    # Last bar
    exit_p = float(opt_df.iloc[-1]["close"]) * (1 - EXIT_SLIPPAGE_PCT)
    return exit_p, "eod_close", opt_df.index[-1], partial_exit


# ── Day Simulation ────────────────────────────────────────────────────

def _simulate_mob_day(dt, data, option_cache, instruments_cfg,
                      instrument_symbols, fe, strategy, capital):
    """Simulate one day of MOB trading. Returns (day_trades, updated_capital)."""
    day_trades = []
    mob_trades_today = 0
    mob_consecutive_losses = 0
    instrument_traded_today = set()
    sl_hit_today = False  # Daily SL stop flag
    day_direction = None  # v2: track direction for alignment across instruments

    for symbol in instrument_symbols:
        instrument = instruments_cfg[symbol]
        if dt not in data.get(symbol, {}):
            continue

        # Daily SL stop: skip remaining instruments after first stoploss
        if settings.mob_daily_sl_stop and sl_hit_today:
            break

        df = data[symbol][dt].copy()
        df = fe.compute_indicators(df)

        if len(df) < 15:
            continue

        # ── Range-bound filter: skip if session looks choppy ──
        if settings.mob_range_filter_enabled and is_range_bound_session(
            df, float(df.iloc[min(settings.mob_range_check_bars - 1, len(df) - 1)]["close"]),
            adx_threshold=settings.mob_range_adx_threshold,
            opening_range_pct=settings.mob_range_opening_pct,
            vwap_cross_limit=settings.mob_range_vwap_crosses,
            min_signals=settings.mob_range_min_signals,
            check_bars=settings.mob_range_check_bars,
        ):
            continue

        for bar_idx in range(14, len(df)):
            # Daily limits
            if mob_trades_today >= MOB_MAX_TRADES_PER_DAY:
                break
            if mob_consecutive_losses >= MOB_CONSECUTIVE_LOSS_STOP:
                break
            if symbol in instrument_traded_today:
                break

            partial_df = df.iloc[:bar_idx + 1].copy()
            spot = float(partial_df.iloc[-1]["close"])
            om = OptionsMetrics()
            signal = strategy.evaluate(partial_df, om, spot)

            if signal is None:
                continue

            opt_type = signal.option_type.value

            # v2: Direction alignment — don't take opposing trades across instruments
            if settings.mob_direction_alignment and day_direction is not None:
                if opt_type != day_direction:
                    continue  # Skip: would be NIFTY CE + SENSEX PE (opposing)

            strike = round(spot / instrument.strike_interval) * instrument.strike_interval

            # Compute expiry and trading symbol (holiday-aware)
            trade_date = datetime.strptime(dt, "%Y-%m-%d").date() if isinstance(dt, str) else dt
            expiry_date = compute_weekly_expiry(trade_date, instrument.expiry_weekday)
            expiry_str = instrument.format_expiry(expiry_date)
            trading_symbol = instrument.build_option_symbol(expiry_str, strike, opt_type)

            opt_key = (symbol, dt, float(strike), opt_type)
            opt_df = option_cache.get(opt_key)

            if opt_df is None or len(opt_df) < 10:
                continue

            # Entry at NEXT candle open after signal bar
            bar_ts = df.index[bar_idx]
            entry_opt_idx = None
            for oidx in range(len(opt_df)):
                if opt_df.index[oidx] >= bar_ts:
                    if oidx + 1 < len(opt_df):
                        entry_opt_idx = oidx + 1
                    else:
                        entry_opt_idx = oidx
                    break

            if entry_opt_idx is None:
                continue

            raw_entry = float(opt_df.iloc[entry_opt_idx]["open"])
            if raw_entry <= 0:
                raw_entry = float(opt_df.iloc[entry_opt_idx]["close"])
            if raw_entry <= 0:
                continue

            # Slippage
            entry_price = round(raw_entry * (1 + MOB_SLIPPAGE_PCT / 100), 2)

            # Min premium filter
            if entry_price < settings.mob_min_premium:
                continue

            # SL & Targets
            one_r = entry_price * MOB_SL_PCT
            sl = round(entry_price - one_r, 2)
            sl = max(sl, 1.0)
            t1 = round(entry_price + one_r, 2)
            t2 = round(entry_price + 2 * one_r, 2)

            # Position sizing — score-based with v2 thresholds
            mob_score = signal.details.get("mob_score", 2)
            high_risk = settings.mob_high_score_risk_pct / 100
            low_risk = settings.mob_low_score_risk_pct / 100
            risk_per_trade = capital * (high_risk if mob_score >= settings.mob_high_score_threshold else low_risk)
            risk_per_lot = (entry_price - sl) * instrument.lot_size
            if risk_per_lot <= 0:
                continue
            num_lots = max(1, int(risk_per_trade / risk_per_lot))
            # v2: score < 4 gets half qty (was < 3)
            if mob_score < 4 and num_lots > 1:
                num_lots = max(1, num_lots // 2)

            # Run exit engine (v2 returns partial_exit info)
            exit_price, exit_reason, exit_ts, partial_exit = _run_mob_exit(
                opt_df, entry_opt_idx, entry_price, sl, t1, t2, one_r
            )

            # PnL with partial exit at T1
            brokerage = MOB_BROKERAGE_PER_LOT * num_lots
            if partial_exit is not None and num_lots > 1:
                # Partial: 50% lots exit at T1 price, rest at final exit
                partial_lots = max(1, int(num_lots * T1_PARTIAL_PCT))
                remaining_lots = num_lots - partial_lots
                partial_pnl = (partial_exit["price"] - entry_price) * instrument.lot_size * partial_lots
                remaining_pnl = (exit_price - entry_price) * instrument.lot_size * remaining_lots
                pnl = partial_pnl + remaining_pnl - brokerage
            else:
                pnl_per_unit = exit_price - entry_price
                pnl = (pnl_per_unit * instrument.lot_size * num_lots) - brokerage
            pnl_pct = (pnl / (entry_price * instrument.lot_size * num_lots) * 100) if entry_price > 0 else 0

            entry_dt = opt_df.index[entry_opt_idx]
            exit_time_str = exit_ts.strftime("%H:%M") if hasattr(exit_ts, "strftime") else ""
            result_str = "WIN" if pnl > 0 else "LOSS"

            trade = {
                "Date": dt,
                "Instrument": symbol,
                "Symbol": trading_symbol,
                "Strategy": "MOMENTUM_OPTION_BUYING",
                "Direction": opt_type,
                "Score": mob_score,
                "Entry Time": entry_dt.strftime("%H:%M"),
                "Exit Time": exit_time_str,
                "Strike": strike,
                "Entry Price": round(entry_price, 2),
                "Exit Price": round(exit_price, 2),
                "Stop Loss": round(sl, 2),
                "Target 1": round(t1, 2),
                "Target 2": round(t2, 2),
                "Lots": num_lots,
                "Lot Size": instrument.lot_size,
                "PnL": round(pnl, 2),
                "PnL %": round(pnl_pct, 2),
                "Exit Reason": exit_reason,
                "Result": result_str,
                "Risk %": settings.mob_high_score_risk_pct if mob_score >= settings.mob_high_score_threshold else settings.mob_low_score_risk_pct,
                "Data Source": "real",
                "Momentum Ratio": round(signal.details.get("momentum_ratio", 0), 1),
                "EMA Aligned": signal.details.get("ema_aligned", None),
                "RSI": signal.details.get("rsi", None),
                "Vol Confirmed": signal.details.get("vol_confirmed", None),
                "Partial Exit": "Yes" if partial_exit else "No",
            }

            day_trades.append(trade)
            mob_trades_today += 1
            instrument_traded_today.add(symbol)
            capital += pnl

            # v2: Lock direction for alignment across instruments
            if day_direction is None:
                day_direction = opt_type

            if pnl < 0:
                mob_consecutive_losses += 1
            else:
                mob_consecutive_losses = 0

            # Flag stoploss hit for daily SL stop
            if exit_reason == "stoploss":
                sl_hit_today = True

    return day_trades, capital


# ── ORB+VWAP Exit Engine ─────────────────────────────────────────────
# Structural SL at opposite ORB end, 2:1 R:R target, VWAP cross exit,
# trail after target.
# ─────────────────────────────────────────────────────────────────────

ORB_EXIT_SLIPPAGE_PCT = settings.orb_exit_slippage_pct / 100
ORB_EOD_EXIT = dtime(settings.orb_eod_exit_hour, settings.orb_eod_exit_minute)
ORB_EOD_EXIT_LOSER = dtime(15, 5)


def _run_orb_exit(opt_df, entry_opt_idx, entry_price, sl, target, index_df,
                  entry_bar_ts, opt_type, vwap_exit_enabled=True):
    """ORB exit engine: structural SL, 2:1 target, VWAP cross exit, trailing.

    Args:
        opt_df: option candle DataFrame
        entry_opt_idx: index in opt_df where we entered
        entry_price: option entry price
        sl: stop loss price (option price)
        target: target price (option price)
        index_df: index candle DataFrame (for VWAP cross check on spot)
        entry_bar_ts: timestamp of entry on index chart
        opt_type: "CE" or "PE" — needed for VWAP cross direction
        vwap_exit_enabled: whether to exit on VWAP cross

    Returns (exit_price, exit_reason, exit_time).
    """
    target_hit = False
    recent_lows = []
    one_r = entry_price - sl  # Risk amount
    vwap_cross_count = 0  # Consecutive bars with VWAP cross
    VWAP_EXIT_MIN_BARS = 30  # Don't check VWAP cross for first N bars
    VWAP_EXIT_CONFIRM = 3    # Need N consecutive VWAP cross bars

    for idx in range(entry_opt_idx + 1, len(opt_df)):
        bars_held = idx - entry_opt_idx
        bar = opt_df.iloc[idx]
        bar_time = opt_df.index[idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        bt = bar_time.time() if hasattr(bar_time, "time") else bar_time.to_pydatetime().time()

        # Smart EOD: exit losers early
        is_losing = bar_close < entry_price
        if is_losing and bt >= ORB_EOD_EXIT_LOSER:
            exit_p = bar_close * (1 - ORB_EXIT_SLIPPAGE_PCT)
            return exit_p, "eod_close", bar_time

        if bt >= ORB_EOD_EXIT:
            exit_p = bar_close * (1 - ORB_EXIT_SLIPPAGE_PCT)
            return exit_p, "eod_close", bar_time

        # Stop loss check
        if bar_low <= sl:
            reason = "trailing_sl" if target_hit else "stoploss"
            exit_p = sl * (1 - ORB_EXIT_SLIPPAGE_PCT)
            return exit_p, reason, bar_time

        # VWAP cross exit: only when trade is losing + after delay + confirmed
        if vwap_exit_enabled and not target_hit and bars_held >= VWAP_EXIT_MIN_BARS and bar_close < entry_price:
            idx_bar = _find_index_bar(index_df, bar_time)
            if idx_bar is not None:
                spot_close = float(idx_bar["close"])
                spot_vwap = idx_bar.get("vwap")
                if spot_vwap is not None and not pd.isna(spot_vwap):
                    spot_vwap = float(spot_vwap)
                    cross_now = False
                    # CE trade: exit if spot drops below VWAP
                    if opt_type == "CE" and spot_close < spot_vwap:
                        cross_now = True
                    # PE trade: exit if spot rises above VWAP
                    if opt_type == "PE" and spot_close > spot_vwap:
                        cross_now = True

                    if cross_now:
                        vwap_cross_count += 1
                        if vwap_cross_count >= VWAP_EXIT_CONFIRM:
                            exit_p = bar_close * (1 - ORB_EXIT_SLIPPAGE_PCT)
                            return exit_p, "vwap_cross", bar_time
                    else:
                        vwap_cross_count = 0

        # Target hit → move SL to breakeven + trail
        if not target_hit and bar_high >= target:
            target_hit = True
            sl = max(sl, entry_price + 0.01 * entry_price)  # Move SL to cost + 1%

        # Trail after target with 2-candle low
        if target_hit and len(recent_lows) >= 2:
            trail_level = min(recent_lows[-2:])
            if trail_level > sl:
                sl = trail_level

        recent_lows.append(bar_low)

    # Last bar
    exit_p = float(opt_df.iloc[-1]["close"]) * (1 - ORB_EXIT_SLIPPAGE_PCT)
    return exit_p, "eod_close", opt_df.index[-1]


def _find_index_bar(index_df, bar_time):
    """Find the matching index candle for a given timestamp."""
    if index_df is None or index_df.empty:
        return None
    # Find closest bar at or before bar_time
    mask = index_df.index <= bar_time
    if mask.any():
        return index_df.loc[mask].iloc[-1]
    return None


# ── ORB Day Simulation ────────────────────────────────────────────────

def _simulate_orb_day(dt, data, option_cache, instruments_cfg,
                      instrument_symbols, fe, strategy, capital):
    """Simulate one day of ORB+VWAP trading. Returns (day_trades, updated_capital)."""
    day_trades = []
    orb_trades_today = 0
    direction_locked = None  # First breakout locks direction for the day

    for symbol in instrument_symbols:
        instrument = instruments_cfg[symbol]
        if dt not in data.get(symbol, {}):
            continue

        if orb_trades_today >= settings.orb_max_trades_per_day:
            break

        df = data[symbol][dt].copy()
        df = fe.compute_indicators(df)

        if len(df) < 20:
            continue

        for bar_idx in range(15, len(df)):  # Start after ORB window
            if orb_trades_today >= settings.orb_max_trades_per_day:
                break

            partial_df = df.iloc[:bar_idx + 1].copy()
            spot = float(partial_df.iloc[-1]["close"])
            om = OptionsMetrics()
            signal = strategy.evaluate(partial_df, om, spot)

            if signal is None:
                continue

            opt_type = signal.option_type.value
            details = signal.details

            # Direction lock: first breakout locks for the day
            if settings.orb_direction_lock:
                if direction_locked is not None and opt_type != direction_locked:
                    continue
                if direction_locked is None:
                    direction_locked = opt_type

            strike = round(spot / instrument.strike_interval) * instrument.strike_interval

            # Expiry
            from datetime import datetime as dt_cls
            trade_date = dt_cls.strptime(dt, "%Y-%m-%d").date() if isinstance(dt, str) else dt
            expiry_date = compute_weekly_expiry(trade_date, instrument.expiry_weekday)
            expiry_str = instrument.format_expiry(expiry_date)
            trading_symbol = instrument.build_option_symbol(expiry_str, strike, opt_type)

            opt_key = (symbol, dt, float(strike), opt_type)
            opt_df = option_cache.get(opt_key)

            if opt_df is None or len(opt_df) < 10:
                continue

            # Entry at NEXT candle open after signal bar
            bar_ts = df.index[bar_idx]
            entry_opt_idx = None
            for oidx in range(len(opt_df)):
                if opt_df.index[oidx] >= bar_ts:
                    if oidx + 1 < len(opt_df):
                        entry_opt_idx = oidx + 1
                    else:
                        entry_opt_idx = oidx
                    break

            if entry_opt_idx is None:
                continue

            raw_entry = float(opt_df.iloc[entry_opt_idx]["open"])
            if raw_entry <= 0:
                raw_entry = float(opt_df.iloc[entry_opt_idx]["close"])
            if raw_entry <= 0:
                continue

            # Slippage
            entry_price = round(raw_entry * (1 + settings.orb_slippage_pct / 100), 2)

            # Min premium filter
            if entry_price < settings.orb_min_premium:
                continue

            # ── Structural SL calculation ──
            # SL is based on opposite end of ORB range mapped to option price
            orb_range = details.get("orb_range", 0)
            breakout_level = details.get("breakout_level", 0)
            structural_sl_level = details.get("structural_sl_level", 0)

            # The ORB range in spot points → approximate as fraction of entry price
            # For ATM options, delta ~0.5, so option moves ~50% of spot move
            option_delta = 0.5  # Conservative ATM delta estimate
            sl_points_option = orb_range * option_delta
            sl_price = round(entry_price - sl_points_option, 2)

            # Fallback: if structural SL is too wide (> config SL%), use config SL
            max_sl = round(entry_price * (1 - settings.orb_sl_pct), 2)
            sl_price = max(sl_price, max_sl, 1.0)

            # Target at R:R ratio
            one_r = entry_price - sl_price
            if one_r <= 0:
                continue
            target = round(entry_price + settings.orb_rr_ratio * one_r, 2)

            # Position sizing
            risk_per_trade = capital * (settings.orb_risk_pct / 100)
            risk_per_lot = one_r * instrument.lot_size
            if risk_per_lot <= 0:
                continue
            num_lots = max(1, int(risk_per_trade / risk_per_lot))

            # Run exit engine
            exit_price, exit_reason, exit_ts = _run_orb_exit(
                opt_df, entry_opt_idx, entry_price, sl_price, target,
                df, bar_ts, opt_type,
                vwap_exit_enabled=settings.orb_vwap_exit_enabled,
            )

            # PnL calculation
            brokerage = settings.orb_brokerage_per_lot * num_lots
            pnl_per_unit = exit_price - entry_price
            pnl = (pnl_per_unit * instrument.lot_size * num_lots) - brokerage
            pnl_pct = (pnl / (entry_price * instrument.lot_size * num_lots) * 100) if entry_price > 0 else 0

            entry_dt = opt_df.index[entry_opt_idx]
            exit_time_str = exit_ts.strftime("%H:%M") if hasattr(exit_ts, "strftime") else ""
            result_str = "WIN" if pnl > 0 else "LOSS"

            trade = {
                "Date": dt,
                "Instrument": symbol,
                "Symbol": trading_symbol,
                "Strategy": "ORB_VWAP",
                "Direction": opt_type,
                "Score": 0,  # ORB doesn't use scoring
                "Entry Time": entry_dt.strftime("%H:%M"),
                "Exit Time": exit_time_str,
                "Strike": strike,
                "Entry Price": round(entry_price, 2),
                "Exit Price": round(exit_price, 2),
                "Stop Loss": round(sl_price, 2),
                "Target 1": round(target, 2),
                "Target 2": 0,
                "Lots": num_lots,
                "Lot Size": instrument.lot_size,
                "PnL": round(pnl, 2),
                "PnL %": round(pnl_pct, 2),
                "Exit Reason": exit_reason,
                "Result": result_str,
                "Risk %": settings.orb_risk_pct,
                "Data Source": "real",
                "Momentum Ratio": round(details.get("orb_range_pct", 0), 3),
                "EMA Aligned": details.get("ema_aligned", None),
                "RSI": details.get("rsi", None),
                "Vol Confirmed": None,
                "Partial Exit": "No",
                "ORB High": details.get("orh", 0),
                "ORB Low": details.get("orl", 0),
                "ORB Range %": details.get("orb_range_pct", 0),
                "VWAP": details.get("vwap", 0),
            }

            day_trades.append(trade)
            orb_trades_today += 1
            capital += pnl

            break  # One trade per instrument per day

    return day_trades, capital


# ── RANGE_BREAKOUT helpers ────────────────────────────────────────────

# Per-strategy, per-instrument allowed time windows (backtest-proven edges)
_STRATEGY_WINDOWS = {
    "RANGE_BREAKOUT": {
        "NIFTY":  {"09:45-10:15"},
        "SENSEX": {"09:45-10:15"},
    },
    "EMA_BREAKOUT": {
        "NIFTY":  {"11:00-12:00"},
        # SENSEX excluded (PF 0.70)
    },
    "MOMENTUM_BREAKOUT": {
        "SENSEX": {"09:45-10:15"},
        # NIFTY excluded (PF 0.77)
    },
}

# Legacy alias
RB_ALLOWED_WINDOWS = {"09:45-10:15", "11:00-12:00"}
RB_STARTING_CAPITAL = 100_000
ANALYSIS_STARTING_CAPITAL = 100_000


def _build_synthetic_index(option_cache, instrument, dates):
    """Build synthetic 1-min index candles from ATM CE+PE via put-call parity."""
    synthetic = {}
    for dt in dates:
        dt_keys = [k for k in option_cache if k[0] == instrument and k[1] == dt]
        if not dt_keys:
            continue
        strikes = sorted(set(k[2] for k in dt_keys))
        if len(strikes) < 3:
            continue
        atm = strikes[len(strikes) // 2]
        ce_df = option_cache.get((instrument, dt, atm, "CE"))
        pe_df = option_cache.get((instrument, dt, atm, "PE"))
        if ce_df is None or pe_df is None or len(ce_df) < 30 or len(pe_df) < 30:
            continue
        merged = ce_df.join(pe_df, lsuffix="_ce", rsuffix="_pe", how="inner")
        if len(merged) < 30:
            continue
        syn_open = atm + merged["open_ce"] - merged["open_pe"]
        syn_close = atm + merged["close_ce"] - merged["close_pe"]
        syn_high = atm + merged["high_ce"] - merged["low_pe"]
        syn_low = atm + merged["low_ce"] - merged["high_pe"]
        df = pd.DataFrame({
            "open": syn_open, "close": syn_close,
            "high": np.maximum(syn_high, np.maximum(syn_open, syn_close)),
            "low": np.minimum(syn_low, np.minimum(syn_open, syn_close)),
            "volume": merged["volume_ce"].fillna(0).astype(int) + merged["volume_pe"].fillna(0).astype(int),
        }, index=merged.index)
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        if len(df) >= 30:
            synthetic[dt] = df
    return synthetic


def _build_prev_closes(index_data):
    """Build {symbol: {date: prev_close}} from index data."""
    result = {}
    for sym, dates_data in index_data.items():
        dates = sorted(dates_data.keys())
        pc = {}
        for i in range(1, len(dates)):
            prev_df = dates_data[dates[i - 1]]
            pc[dates[i]] = float(prev_df.iloc[-1]["close"])
        result[sym] = pc
    return result


# ── Main runner ──────────────────────────────────────────────────────

async def run_backtest(
    start_date: str,
    end_date: str,
    instruments: list[str] | None = None,
    strategies: list[str] | None = None,
) -> str:
    """Launch a backtest asynchronously. Returns a job_id for tracking."""
    job_id = str(uuid.uuid4())[:8]
    progress = BacktestProgress(job_id=job_id)
    _active_jobs[job_id] = progress

    # Determine which strategy to run (default: RANGE_BREAKOUT)
    strategy_name = "RANGE_BREAKOUT"
    if strategies and len(strategies) > 0:
        s = strategies[0].upper().strip()
        if s in ("RB", "RANGE_BREAKOUT"):
            strategy_name = "RANGE_BREAKOUT"
        elif s in ("EMA", "EMA_BREAKOUT"):
            strategy_name = "EMA_BREAKOUT"
        elif s in ("MB", "MOMENTUM_BREAKOUT"):
            strategy_name = "MOMENTUM_BREAKOUT"
        elif s in ("ALL", "ALL_THREE", "COMBINED"):
            strategy_name = "ALL_THREE"
        else:
            strategy_name = "RANGE_BREAKOUT"

    asyncio.create_task(_run_backtest_task(job_id, start_date, end_date, strategy_name))
    return job_id


async def _run_backtest_task(
    job_id: str,
    start_date: str,
    end_date: str,
    strategy_name: str = "ORB_VWAP",
) -> None:
    """Run backtest with progress tracking. Supports MOB, ORB_VWAP, and RANGE_BREAKOUT."""
    progress = _active_jobs[job_id]

    if strategy_name in ("RANGE_BREAKOUT", "EMA_BREAKOUT", "MOMENTUM_BREAKOUT", "ALL_THREE"):
        await _run_analysis_backtest_task(job_id, start_date, end_date, strategy_name)
        return

    is_orb = strategy_name == "ORB_VWAP"
    try:
        progress.status = "loading"
        progress.message = f"Initializing {strategy_name} backtest..."

        db_url = str(settings.database_url)
        engine = create_async_engine(db_url, echo=False)
        fe = FeatureEngine()

        if is_orb:
            strategy = ORBVWAPStrategy(
                min_range_pct=settings.orb_min_range_pct,
                max_range_atr_mult=settings.orb_max_range_atr_mult,
                entry_deadline=dtime(settings.orb_entry_window_end_hour,
                                    settings.orb_entry_window_end_min),
            )
            starting_capital = settings.orb_starting_capital
        else:
            strategy = MomentumOptionBuyingStrategy(
                afternoon_enabled=settings.mob_afternoon_enabled
            )
            starting_capital = STARTING_CAPITAL

        instruments_cfg = {ic.symbol: ic for ic in get_enabled_instruments()}
        instrument_symbols = list(instruments_cfg.keys())

        # Load 90 days before for indicator warmup
        eval_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)
        ).strftime("%Y-%m-%d")

        # Load data
        data = {}
        option_cache = {}

        for symbol in instrument_symbols:
            progress.message = f"Loading {symbol} index candles..."
            idx = await _load_index_candles(engine, symbol, eval_start, end_date)
            data[symbol] = idx

            dates_list = sorted(idx.keys())
            progress.message = f"Loading {symbol} option candles..."
            oc = await _load_option_candles_batch(engine, symbol, dates_list)
            option_cache.update(oc)

        await engine.dispose()

        # Trading dates within requested range
        trading_dates = sorted(set(
            d for sym in instrument_symbols
            for d in data.get(sym, {}).keys()
            if d >= start_date and d <= end_date
        ))

        progress.total_days = len(trading_dates)
        progress.status = "running"
        progress.message = f"Simulating {len(trading_dates)} trading days ({strategy_name})..."

        # Day-by-day simulation
        all_trades = []
        equity_curve = []
        capital = starting_capital
        consecutive_loss_days = 0
        cooldown_remaining = 0

        for day_idx, dt in enumerate(trading_dates):
            progress.processed_days = day_idx + 1
            progress.current_date = dt
            progress.message = f"Day {day_idx + 1}/{len(trading_dates)}: {dt}"

            # Consecutive losing day cooldown (MOB only)
            if not is_orb and cooldown_remaining > 0:
                cooldown_remaining -= 1
                equity_curve.append({
                    "Date": dt,
                    "PnL": 0,
                    "Capital": round(capital, 2),
                    "Trades": 0,
                })
                continue

            if is_orb:
                day_trades, capital = _simulate_orb_day(
                    dt, data, option_cache, instruments_cfg,
                    instrument_symbols, fe, strategy, capital
                )
            else:
                day_trades, capital = _simulate_mob_day(
                    dt, data, option_cache, instruments_cfg,
                    instrument_symbols, fe, strategy, capital
                )

            day_pnl = sum(t["PnL"] for t in day_trades)

            # Track consecutive loss days for cooldown (MOB only)
            if not is_orb and day_trades:
                any_win = any(t["PnL"] > 0 for t in day_trades)
                if any_win:
                    consecutive_loss_days = 0
                else:
                    consecutive_loss_days += 1
                    if consecutive_loss_days >= settings.mob_cooldown_loss_days:
                        cooldown_remaining = settings.mob_cooldown_skip_days
                        consecutive_loss_days = 0

            equity_curve.append({
                "Date": dt,
                "PnL": round(day_pnl, 2),
                "Capital": round(capital, 2),
                "Trades": len(day_trades),
            })

            all_trades.extend(day_trades)

            # Yield control to event loop
            if day_idx % 3 == 0:
                await asyncio.sleep(0)

        # Build result
        _build_result(job_id, all_trades, equity_curve, start_date, end_date,
                      instrument_symbols, capital, progress, strategy_name, starting_capital)

    except Exception as e:
        logger.exception("Backtest %s failed", job_id)
        progress.status = "failed"
        progress.error = str(e)


# ── RANGE_BREAKOUT backtest task ──────────────────────────────────────

async def _load_option_dates(engine, symbol, start_date, end_date):
    """Get distinct dates with option candle data."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text(
                "SELECT DISTINCT date FROM option_candles "
                "WHERE instrument = :inst AND date >= :s AND date <= :e "
                "ORDER BY date"
            ),
            {"inst": symbol, "s": start_date, "e": end_date},
        )
        return [str(row[0]) for row in result.fetchall()]


async def _run_analysis_backtest_task(
    job_id: str, start_date: str, end_date: str, strategy_name: str,
) -> None:
    """Run any analysis-engine strategy backtest with per-instrument window filtering.

    Uses _STRATEGY_WINDOWS to determine which instruments and time windows
    are allowed for the given strategy.  strategy_name="ALL_THREE" runs
    RANGE_BREAKOUT + EMA_BREAKOUT + MOMENTUM_BREAKOUT as a combined portfolio.
    """
    progress = _active_jobs[job_id]
    try:
        progress.status = "loading"
        display_name = "ALL 3 STRATEGIES" if strategy_name == "ALL_THREE" else strategy_name
        progress.message = f"Initializing {display_name} backtest..."

        db_url = str(settings.database_url)
        engine = create_async_engine(db_url, echo=False)
        fe = FeatureEngine()
        classifier = EnhancedDayClassifier()

        # For ALL_THREE: run each strategy with its own tester and window config
        if strategy_name == "ALL_THREE":
            strategy_configs = [
                ("RANGE_BREAKOUT", _STRATEGY_WINDOWS["RANGE_BREAKOUT"]),
                ("EMA_BREAKOUT", _STRATEGY_WINDOWS["EMA_BREAKOUT"]),
                ("MOMENTUM_BREAKOUT", _STRATEGY_WINDOWS["MOMENTUM_BREAKOUT"]),
            ]
        else:
            strat_windows = _STRATEGY_WINDOWS.get(strategy_name, {})
            strategy_configs = [(strategy_name, strat_windows)]

        # Build testers per strategy
        testers = {}
        for sname, _ in strategy_configs:
            testers[sname] = StrategyTester(fe, strategy_filter=[sname])

        # Collect all instruments needed across all strategies
        all_instrument_syms = set()
        for _, sw in strategy_configs:
            all_instrument_syms.update(sw.keys())
        instruments = [get_instrument(sym) for sym in sorted(all_instrument_syms)]
        starting_capital = ANALYSIS_STARTING_CAPITAL

        index_data = {}
        option_cache = {}

        for inst in instruments:
            sym = inst.symbol
            progress.message = f"Loading {sym} data..."

            option_dates = await _load_option_dates(engine, sym, start_date, end_date)
            idx = await _load_index_candles(engine, sym, start_date, end_date)

            all_dates = sorted(set(option_dates) | set(idx.keys()))
            oc = await _load_option_candles_batch(engine, sym, all_dates)
            option_cache.update(oc)

            missing = [d for d in option_dates if d not in idx]
            if missing:
                synthetic = _build_synthetic_index(option_cache, sym, missing)
                idx.update(synthetic)

            index_data[sym] = idx

        prev_closes = _build_prev_closes(index_data)
        await engine.dispose()

        trading_dates = sorted(set(
            d for inst in instruments
            for d in index_data.get(inst.symbol, {}).keys()
            if d >= start_date and d <= end_date
        ))

        progress.total_days = len(trading_dates)
        progress.status = "running"
        progress.message = f"Simulating {len(trading_dates)} days ({display_name})..."

        all_trades = []
        equity_curve = []
        capital = starting_capital
        DAILY_LOSS_CAP = -1500  # Stop adding trades once day PnL crosses this
        GAP_SKIP_PCT = 1.0     # Skip day if any instrument's opening gap > this %

        for day_idx, dt in enumerate(trading_dates):
            progress.processed_days = day_idx + 1
            progress.current_date = dt
            progress.message = f"Day {day_idx + 1}/{len(trading_dates)}: {dt}"

            day_pnl = 0.0
            day_trade_count = 0
            day_capped = False

            # Pre-trade gap filter: skip entire day if opening gap is too large
            gap_too_large = False
            for inst in instruments:
                sym = inst.symbol
                if dt not in index_data.get(sym, {}):
                    continue
                df = index_data[sym][dt]
                prev = prev_closes.get(sym, {}).get(dt)
                if prev and len(df) > 0:
                    open_price = float(df.iloc[0]["open"] if "open" in df.columns else df.iloc[0]["close"])
                    gap_pct = abs(open_price - prev) / prev * 100
                    if gap_pct > GAP_SKIP_PCT:
                        gap_too_large = True
                        break

            if gap_too_large:
                equity_curve.append({
                    "Date": dt, "PnL": 0, "Capital": round(capital, 2), "Trades": 0,
                })
                continue

            for inst in instruments:
                if day_capped:
                    break
                sym = inst.symbol
                if dt not in index_data.get(sym, {}):
                    continue

                df = index_data[sym][dt]
                prev = prev_closes.get(sym, {}).get(dt)
                hs = classifier.classify_hindsight(df, prev)

                # Run each strategy that is allowed on this instrument
                for sname, strat_windows in strategy_configs:
                    allowed_windows = strat_windows.get(sym, set())
                    if not allowed_windows:
                        continue

                    trades = testers[sname].test_day(
                        df=df,
                        instrument_symbol=sym,
                        strike_interval=inst.strike_interval,
                        lot_size=inst.lot_size,
                        option_cache=option_cache,
                        oi_snapshots=[],
                        day_type=hs.value,
                        day_type_hindsight=hs.value,
                        date_str=dt,
                    )

                    for t in trades:
                        if t.time_window not in allowed_windows:
                            continue

                        result_str = "WIN" if t.pnl > 0 else "LOSS"
                        trade = {
                            "Date": t.date,
                            "Instrument": t.instrument,
                            "Symbol": f"{t.instrument}{int(t.strike)}{t.direction}",
                            "Strategy": sname,
                            "Direction": t.direction,
                            "Score": 0,
                            "Entry Time": t.entry_time,
                            "Exit Time": t.exit_time,
                            "Strike": t.strike,
                            "Entry Price": round(t.entry_price, 2),
                            "Exit Price": round(t.exit_price, 2),
                            "Stop Loss": round(t.entry_price * 0.80, 2),
                            "Target 1": round(t.entry_price * 1.20, 2),
                            "Target 2": round(t.entry_price * 1.40, 2),
                            "Lots": 1,
                            "Lot Size": t.lot_size,
                            "PnL": round(t.pnl, 2),
                            "PnL %": round(t.pnl_pct, 2),
                            "Exit Reason": t.exit_reason,
                            "Result": result_str,
                            "Risk %": 20.0,
                            "Data Source": "real",
                            "R Multiple": round(t.r_multiple, 2),
                            "Time Window": t.time_window,
                            "Day Type": t.day_type_hindsight,
                            "Hold Minutes": t.hold_minutes,
                        }
                        all_trades.append(trade)
                        capital += t.pnl
                        day_pnl += t.pnl
                        day_trade_count += 1

                        # Daily loss cap: stop trading for the day once breached
                        if day_pnl <= DAILY_LOSS_CAP:
                            day_capped = True
                            break
                    if day_capped:
                        break

            equity_curve.append({
                "Date": dt,
                "PnL": round(day_pnl, 2),
                "Capital": round(capital, 2),
                "Trades": day_trade_count,
            })

            if day_idx % 3 == 0:
                await asyncio.sleep(0)

        _build_result(job_id, all_trades, equity_curve, start_date, end_date,
                      [i.symbol for i in instruments], capital, progress,
                      strategy_name, starting_capital)

    except Exception as e:
        logger.exception("Backtest %s failed (%s)", job_id, strategy_name)
        progress.status = "failed"
        progress.error = str(e)


# Legacy alias for backward compat
async def _run_rb_backtest_task(job_id: str, start_date: str, end_date: str) -> None:
    await _run_analysis_backtest_task(job_id, start_date, end_date, "RANGE_BREAKOUT")


def _build_result(job_id, all_trades, equity_curve, start_date, end_date,
                  instruments, capital, progress, strategy_name="MOB", starting_capital=None):
    """Build BacktestResult from simulation output."""
    if starting_capital is None:
        starting_capital = STARTING_CAPITAL
    total_trades = len(all_trades)
    winners = sum(1 for t in all_trades if t["PnL"] > 0)
    losers = total_trades - winners
    total_pnl = sum(t["PnL"] for t in all_trades)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0

    gp = sum(t["PnL"] for t in all_trades if t["PnL"] > 0)
    gl = abs(sum(t["PnL"] for t in all_trades if t["PnL"] <= 0)) or 1
    profit_factor = gp / gl

    peak = starting_capital
    max_dd = 0.0
    for row in equity_curve:
        cap = row["Capital"]
        if cap > peak:
            peak = cap
        dd = peak - cap
        if dd > max_dd:
            max_dd = dd

    daily_pnls = [r["PnL"] for r in equity_curve]
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = float(np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)
    else:
        sharpe = 0.0

    is_orb = strategy_name == "ORB_VWAP"
    is_analysis = strategy_name in ("RANGE_BREAKOUT", "EMA_BREAKOUT", "MOMENTUM_BREAKOUT", "ALL_THREE")
    if is_analysis:
        if strategy_name == "ALL_THREE":
            window_strs = []
            for sn in ("RANGE_BREAKOUT", "EMA_BREAKOUT", "MOMENTUM_BREAKOUT"):
                sw = _STRATEGY_WINDOWS.get(sn, {})
                for sym, wins in sw.items():
                    window_strs.append(f"{sn} → {sym}: {', '.join(sorted(wins))}")
            entry_cond = "RB: ADX<20 range breakout | EMA: EMA50 cross | MB: Donchian breakout"
        else:
            sw = _STRATEGY_WINDOWS.get(strategy_name, {})
            window_strs = [f"{sym}: {', '.join(sorted(wins))}" for sym, wins in sw.items()]
            entry_cond = {
                "RANGE_BREAKOUT": "ADX<20, range<0.80%, breakout + RSI/volume/body",
                "EMA_BREAKOUT": "Price crosses EMA50, EMA9>EMA20, RSI 50-70, body≥40%",
                "MOMENTUM_BREAKOUT": "Donchian 20-candle breakout, ADX>25, RSI>60, volume≥1.5×",
            }.get(strategy_name, "")
        config_used = {
            "strategy": strategy_name,
            "initial_capital": starting_capital,
            "sl_pct": "20%",
            "daily_loss_cap": "₹1,500",
            "gap_skip": ">1.0% opening gap → skip day",
            "entry_conditions": entry_cond,
            "time_windows": " | ".join(window_strs),
            "exit_engine": "20% SL (1R) | T1 +1R → SL→BE | T2 +2R → lock 1R | EOD 15:10",
            "instruments": instruments,
        }
    elif is_orb:
        config_used = {
            "strategy": "ORB_VWAP",
            "initial_capital": starting_capital,
            "sl_type": settings.orb_sl_type,
            "sl_pct": f"{settings.orb_sl_pct * 100:.0f}%",
            "rr_ratio": f"{settings.orb_rr_ratio:.1f}:1",
            "slippage_pct": f"{settings.orb_slippage_pct}% entry + {settings.orb_exit_slippage_pct}% exit",
            "max_trades_per_day": settings.orb_max_trades_per_day,
            "brokerage_per_lot": settings.orb_brokerage_per_lot,
            "entry_deadline": f"{settings.orb_entry_window_end_hour}:{settings.orb_entry_window_end_min:02d}",
            "eod_exit_time": f"{settings.orb_eod_exit_hour}:{settings.orb_eod_exit_minute:02d}",
            "vwap_exit": "Enabled" if settings.orb_vwap_exit_enabled else "Disabled",
            "direction_lock": "Enabled" if settings.orb_direction_lock else "Disabled",
            "instruments": instruments,
            "exit_engine": "Structural SL (ORB opposite) | 2:1 R:R target | VWAP cross exit | Trail after target",
        }
    else:
        config_used = {
            "strategy": "MOMENTUM_OPTION_BUYING",
            "initial_capital": starting_capital,
            "sl_pct": f"{MOB_SL_PCT * 100:.0f}%",
            "slippage_pct": f"{MOB_SLIPPAGE_PCT}%",
            "max_trades_per_day": MOB_MAX_TRADES_PER_DAY,
            "consecutive_loss_stop": MOB_CONSECUTIVE_LOSS_STOP,
            "brokerage_per_lot": MOB_BROKERAGE_PER_LOT,
            "eod_exit_time": str(EOD_EXIT_TIME),
            "instruments": instruments,
            "exit_engine": "T1 → partial 50% + cost+1% | T2 → lock 1R | 2-candle trail | 45-bar time exit",
        }

    result = BacktestResult(
        job_id=job_id,
        start_date=start_date,
        end_date=end_date,
        instruments=instruments,
        initial_capital=starting_capital,
        ending_capital=round(capital, 2),
        total_pnl=round(total_pnl, 2),
        return_pct=round((capital - starting_capital) / starting_capital * 100, 2),
        total_trades=total_trades,
        winners=winners,
        losers=losers,
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd / starting_capital * 100, 2),
        trades=all_trades,
        equity_curve=equity_curve,
        config_used=config_used,
    )
    _job_results[job_id] = result
    progress.status = "completed"
    progress.message = f"Done — {total_trades} trades, PnL: ₹{total_pnl:+,.0f}"


# ── Excel export ──────────────────────────────────────────────────────

def generate_excel(job_id: str) -> Optional[bytes]:
    """Generate Excel report for a completed backtest."""
    result = _job_results.get(job_id)
    if not result or not result.trades:
        return None

    tdf = pd.DataFrame(result.trades)
    edf = pd.DataFrame(result.equity_curve)

    strategy = result.config_used.get("strategy", "MOMENTUM_OPTION_BUYING")
    if strategy in ("RANGE_BREAKOUT", "EMA_BREAKOUT", "MOMENTUM_BREAKOUT"):
        display_cols = [
            "Date", "Instrument", "Symbol", "Strategy", "Direction",
            "Entry Time", "Exit Time", "Strike", "Entry Price", "Exit Price",
            "Stop Loss", "Target 1", "Target 2", "Lots", "Lot Size",
            "PnL", "PnL %", "Exit Reason", "Result",
            "R Multiple", "Time Window", "Day Type", "Hold Minutes",
        ]
    elif strategy == "ORB_VWAP":
        display_cols = [
            "Date", "Instrument", "Symbol", "Strategy", "Direction",
            "Entry Time", "Exit Time", "Strike", "Entry Price", "Exit Price",
            "Stop Loss", "Target 1", "Lots", "Lot Size",
            "PnL", "PnL %", "Exit Reason", "Result", "Data Source",
            "ORB High", "ORB Low", "ORB Range %", "VWAP", "RSI",
        ]
    else:
        display_cols = [
            "Date", "Instrument", "Symbol", "Strategy", "Direction", "Score",
            "Entry Time", "Exit Time", "Strike", "Entry Price", "Exit Price",
            "Stop Loss", "Target 1", "Target 2", "Lots", "Lot Size",
            "PnL", "PnL %", "Exit Reason", "Result", "Risk %", "Data Source",
            "Momentum Ratio", "EMA Aligned", "RSI", "Vol Confirmed", "Partial Exit",
        ]
    trades_display = tdf[[c for c in display_cols if c in tdf.columns]]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: All Trades
        trades_display.to_excel(writer, sheet_name="All Trades", index=False)

        # Sheet 2: Daily Summary
        daily = tdf.groupby("Date").agg(
            Trades=("PnL", "count"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
            Losers=("Result", lambda x: (x == "LOSS").sum()),
            Day_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
            Best_Trade=("PnL", "max"),
            Worst_Trade=("PnL", "min"),
        ).reset_index()
        daily["Win_Rate_%"] = (daily["Winners"] / daily["Trades"] * 100).round(1)
        daily = daily.merge(edf[["Date", "Capital"]], on="Date", how="left")
        daily.to_excel(writer, sheet_name="Daily Summary", index=False)

        # Sheet 3: By Instrument
        inst = tdf.groupby("Instrument").agg(
            Trades=("PnL", "count"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
            Losers=("Result", lambda x: (x == "LOSS").sum()),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
            Avg_Score=("Score", "mean"),
        ).reset_index()
        inst["Win_Rate_%"] = (inst["Winners"] / inst["Trades"] * 100).round(1)
        inst.to_excel(writer, sheet_name="By Instrument", index=False)

        # Sheet 4: By Exit Reason
        exit_s = tdf.groupby("Exit Reason").agg(
            Count=("PnL", "count"),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
        ).reset_index()
        exit_s["Win_Rate_%"] = (exit_s["Winners"] / exit_s["Count"] * 100).round(1)
        exit_s.to_excel(writer, sheet_name="By Exit Reason", index=False)

        # Sheet 5: By Direction
        dir_s = tdf.groupby("Direction").agg(
            Trades=("PnL", "count"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
        ).reset_index()
        dir_s["Win_Rate_%"] = (dir_s["Winners"] / dir_s["Trades"] * 100).round(1)
        dir_s.to_excel(writer, sheet_name="By Direction", index=False)

        # Sheet 6: Capital Curve
        edf.to_excel(writer, sheet_name="Capital Curve", index=False)

        # Sheet 7: Performance Summary
        avg_win = tdf[tdf["PnL"] > 0]["PnL"].mean() if result.winners > 0 else 0
        avg_loss = abs(tdf[tdf["PnL"] <= 0]["PnL"].mean()) if result.losers > 0 else 0
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        perf = pd.DataFrame([
            {"Metric": "Strategy", "Value": strategy},
            {"Metric": "Period", "Value": f"{result.start_date} to {result.end_date}"},
            {"Metric": "Trading Days", "Value": len(edf)},
            {"Metric": "Days with Trades", "Value": tdf["Date"].nunique()},
            {"Metric": "Starting Capital", "Value": f"₹{result.initial_capital:,.0f}"},
            {"Metric": "Ending Capital", "Value": f"₹{result.ending_capital:,.0f}"},
            {"Metric": "Total PnL", "Value": f"₹{result.total_pnl:,.2f}"},
            {"Metric": "Return %", "Value": f"{result.return_pct:.2f}%"},
            {"Metric": "Total Trades", "Value": result.total_trades},
            {"Metric": "Winners", "Value": result.winners},
            {"Metric": "Losers", "Value": result.losers},
            {"Metric": "Win Rate %", "Value": f"{result.win_rate:.1f}%"},
            {"Metric": "Profit Factor", "Value": f"{result.profit_factor:.2f}"},
            {"Metric": "Avg Win", "Value": f"₹{avg_win:,.2f}"},
            {"Metric": "Avg Loss", "Value": f"₹{avg_loss:,.2f}"},
            {"Metric": "Win/Loss Ratio", "Value": f"{wl_ratio:.2f}x"},
            {"Metric": "Sharpe Ratio (Ann.)", "Value": f"{result.sharpe_ratio:.2f}"},
            {"Metric": "Max Drawdown", "Value": f"₹{result.max_drawdown:,.2f}"},
            {"Metric": "Max Drawdown %", "Value": f"{result.max_drawdown_pct:.2f}%"},
            {"Metric": "Instruments", "Value": ", ".join(result.instruments)},
        ])
        perf.to_excel(writer, sheet_name="Performance Summary", index=False)

        # Sheet 8: Config
        cfg = pd.DataFrame([
            {"Key": k, "Value": str(v)} for k, v in result.config_used.items()
        ])
        cfg.to_excel(writer, sheet_name="Config", index=False)

    buf.seek(0)
    return buf.read()

