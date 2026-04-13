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
from app.core.instruments import get_enabled_instruments
from app.core.models import OptionsMetrics
from app.engine.feature_engine import FeatureEngine
from app.strategies.momentum_option_buying import MomentumOptionBuyingStrategy

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Only MOB strategy is active — all others disabled
ALL_STRATEGIES = {
    "MOMENTUM_OPTION_BUYING": MomentumOptionBuyingStrategy,
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


# ── MOB Exit Engine ───────────────────────────────────────────────────

def _run_mob_exit(opt_df, entry_opt_idx, entry_price, sl, t1, t2, one_r):
    """MOB exit: SL → cost+buffer at T1 → lock 1R at T2 → trail 3-candle low."""
    t1_hit = False
    t2_hit = False
    recent_lows = []

    for idx in range(entry_opt_idx + 1, len(opt_df)):
        bar = opt_df.iloc[idx]
        bar_time = opt_df.index[idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        bt = bar_time.time() if hasattr(bar_time, "time") else bar_time.to_pydatetime().time()

        # EOD close — takes priority
        if bt >= EOD_EXIT_TIME:
            return bar_close, "eod_close", bar_time

        # Stoploss check
        if bar_low <= sl:
            reason = "trailing_sl" if (t1_hit or t2_hit) else "stoploss"
            return sl, reason, bar_time

        # T2 (+2R) → lock 1R profit
        if not t2_hit and bar_high >= t2:
            t2_hit = True
            t1_hit = True
            sl = max(sl, entry_price + one_r)

        # T1 (+1R) → move SL to cost + 0.5% buffer
        if not t1_hit and bar_high >= t1:
            t1_hit = True
            sl = max(sl, entry_price * 1.005)

        # Trail after T2 using 3-candle low
        if t2_hit and len(recent_lows) >= 3:
            trail_level = min(recent_lows[-3:])
            if trail_level > sl:
                sl = trail_level

        recent_lows.append(bar_low)

    # Last bar
    return float(opt_df.iloc[-1]["close"]), "eod_close", opt_df.index[-1]


# ── Day Simulation ────────────────────────────────────────────────────

def _simulate_mob_day(dt, data, option_cache, instruments_cfg,
                      instrument_symbols, fe, strategy, capital):
    """Simulate one day of MOB trading. Returns (day_trades, updated_capital)."""
    day_trades = []
    mob_trades_today = 0
    mob_consecutive_losses = 0
    instrument_traded_today = set()

    for symbol in instrument_symbols:
        instrument = instruments_cfg[symbol]
        if dt not in data.get(symbol, {}):
            continue

        df = data[symbol][dt].copy()
        df = fe.compute_indicators(df)

        if len(df) < 15:
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
            strike = round(spot / instrument.strike_interval) * instrument.strike_interval

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

            # SL & Targets
            one_r = entry_price * MOB_SL_PCT
            sl = round(entry_price - one_r, 2)
            sl = max(sl, 1.0)
            t1 = round(entry_price + one_r, 2)
            t2 = round(entry_price + 2 * one_r, 2)

            # Position sizing
            mob_score = signal.details.get("mob_score", 2)
            high_risk = settings.mob_high_score_risk_pct / 100
            low_risk = settings.mob_low_score_risk_pct / 100
            risk_per_trade = capital * (high_risk if mob_score >= settings.mob_high_score_threshold else low_risk)
            risk_per_lot = (entry_price - sl) * instrument.lot_size
            if risk_per_lot <= 0:
                continue
            num_lots = max(1, int(risk_per_trade / risk_per_lot))
            if mob_score < 3 and num_lots > 1:
                num_lots = max(1, num_lots // 2)

            # Run exit engine
            exit_price, exit_reason, exit_ts = _run_mob_exit(
                opt_df, entry_opt_idx, entry_price, sl, t1, t2, one_r
            )

            # PnL
            brokerage = MOB_BROKERAGE_PER_LOT * num_lots
            pnl_per_unit = exit_price - entry_price
            pnl = (pnl_per_unit * instrument.lot_size * num_lots) - brokerage
            pnl_pct = (pnl_per_unit / entry_price * 100) if entry_price > 0 else 0

            entry_dt = opt_df.index[entry_opt_idx]
            exit_time_str = exit_ts.strftime("%H:%M") if hasattr(exit_ts, "strftime") else ""
            result_str = "WIN" if pnl > 0 else "LOSS"

            trade = {
                "Date": dt,
                "Instrument": symbol,
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
            }

            day_trades.append(trade)
            mob_trades_today += 1
            instrument_traded_today.add(symbol)
            capital += pnl

            if pnl < 0:
                mob_consecutive_losses += 1
            else:
                mob_consecutive_losses = 0

    return day_trades, capital


# ── Main runner ──────────────────────────────────────────────────────

async def run_backtest(
    start_date: str,
    end_date: str,
    instruments: list[str] | None = None,
    strategies: list[str] | None = None,
) -> str:
    """Launch a MOB backtest asynchronously. Returns a job_id for tracking."""
    job_id = str(uuid.uuid4())[:8]
    progress = BacktestProgress(job_id=job_id)
    _active_jobs[job_id] = progress

    asyncio.create_task(_run_backtest_task(job_id, start_date, end_date))
    return job_id


async def _run_backtest_task(
    job_id: str,
    start_date: str,
    end_date: str,
) -> None:
    """Run MOB-only backtest with progress tracking."""
    progress = _active_jobs[job_id]
    try:
        progress.status = "loading"
        progress.message = "Initializing MOB backtest..."

        db_url = str(settings.database_url)
        engine = create_async_engine(db_url, echo=False)
        fe = FeatureEngine()
        strategy = MomentumOptionBuyingStrategy()

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
        progress.message = f"Simulating {len(trading_dates)} trading days..."

        # Day-by-day MOB simulation
        all_trades = []
        equity_curve = []
        capital = STARTING_CAPITAL

        for day_idx, dt in enumerate(trading_dates):
            progress.processed_days = day_idx + 1
            progress.current_date = dt
            progress.message = f"Day {day_idx + 1}/{len(trading_dates)}: {dt}"

            day_trades, capital = _simulate_mob_day(
                dt, data, option_cache, instruments_cfg,
                instrument_symbols, fe, strategy, capital
            )

            day_pnl = sum(t["PnL"] for t in day_trades)
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
                      instrument_symbols, capital, progress)

    except Exception as e:
        logger.exception("Backtest %s failed", job_id)
        progress.status = "failed"
        progress.error = str(e)


def _build_result(job_id, all_trades, equity_curve, start_date, end_date,
                  instruments, capital, progress):
    """Build BacktestResult from simulation output."""
    total_trades = len(all_trades)
    winners = sum(1 for t in all_trades if t["PnL"] > 0)
    losers = total_trades - winners
    total_pnl = sum(t["PnL"] for t in all_trades)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0

    gp = sum(t["PnL"] for t in all_trades if t["PnL"] > 0)
    gl = abs(sum(t["PnL"] for t in all_trades if t["PnL"] <= 0)) or 1
    profit_factor = gp / gl

    peak = STARTING_CAPITAL
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

    config_used = {
        "strategy": "MOMENTUM_OPTION_BUYING",
        "initial_capital": STARTING_CAPITAL,
        "sl_pct": f"{MOB_SL_PCT * 100:.0f}%",
        "slippage_pct": f"{MOB_SLIPPAGE_PCT}%",
        "max_trades_per_day": MOB_MAX_TRADES_PER_DAY,
        "consecutive_loss_stop": MOB_CONSECUTIVE_LOSS_STOP,
        "brokerage_per_lot": MOB_BROKERAGE_PER_LOT,
        "eod_exit_time": str(EOD_EXIT_TIME),
        "instruments": instruments,
        "exit_engine": "T1 → cost+0.5% | T2 → lock 1R | 3-candle trail",
    }

    result = BacktestResult(
        job_id=job_id,
        start_date=start_date,
        end_date=end_date,
        instruments=instruments,
        initial_capital=STARTING_CAPITAL,
        ending_capital=round(capital, 2),
        total_pnl=round(total_pnl, 2),
        return_pct=round((capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
        total_trades=total_trades,
        winners=winners,
        losers=losers,
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd / STARTING_CAPITAL * 100, 2),
        trades=all_trades,
        equity_curve=equity_curve,
        config_used=config_used,
    )
    _job_results[job_id] = result
    progress.status = "completed"
    progress.message = f"Done — {total_trades} trades, PnL: ₹{total_pnl:+,.0f}"


# ── Excel export ──────────────────────────────────────────────────────

def generate_excel(job_id: str) -> Optional[bytes]:
    """Generate Excel report for a completed MOB backtest."""
    result = _job_results.get(job_id)
    if not result or not result.trades:
        return None

    tdf = pd.DataFrame(result.trades)
    edf = pd.DataFrame(result.equity_curve)

    display_cols = [
        "Date", "Instrument", "Strategy", "Direction", "Score",
        "Entry Time", "Exit Time", "Strike", "Entry Price", "Exit Price",
        "Stop Loss", "Target 1", "Target 2", "Lots", "Lot Size",
        "PnL", "PnL %", "Exit Reason", "Result", "Risk %", "Data Source",
        "Momentum Ratio",
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
            {"Metric": "Strategy", "Value": "MOMENTUM_OPTION_BUYING"},
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

