"""Production-exact backtest runner for UI API.

Wraps the ExactProductionBacktest engine from backtest_exact.py for use
by the FastAPI backtest endpoints. Provides async job management,
progress tracking, and Excel export.

Same simulation engine used by CLI (backtest_exact.py) and UI (this module).
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from app.core.config import settings
from app.core.instruments import get_enabled_instruments, get_instrument

logger = logging.getLogger(__name__)
_IST = pytz.timezone("Asia/Kolkata")

# Re-export ALL_STRATEGIES for routes.py /api/backtest/config
from app.strategies.trend_pullback import TrendPullbackStrategy
from app.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.strategies.orb import ORBStrategy
from app.strategies.range_breakout import RangeBreakoutStrategy
from app.strategies.vwap_reclaim import VWAPReclaimStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy

ALL_STRATEGIES = {
    "TREND_PULLBACK": TrendPullbackStrategy,
    "MOMENTUM_BREAKOUT": MomentumBreakoutStrategy,
    "ORB": ORBStrategy,
    "VWAP_RECLAIM": VWAPReclaimStrategy,
    "RANGE_BREAKOUT": RangeBreakoutStrategy,
    "LIQUIDITY_SWEEP": LiquiditySweepStrategy,
}


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

    asyncio.create_task(_run_backtest_task(job_id, start_date, end_date, instruments, strategies))
    return job_id


async def _run_backtest_task(
    job_id: str,
    start_date: str,
    end_date: str,
    instruments: list[str] | None,
    strategies: list[str] | None,
) -> None:
    """Run the full production-exact backtest simulation."""
    progress = _active_jobs[job_id]
    try:
        progress.status = "loading"
        progress.message = "Importing simulation engine..."

        # Import the production-exact simulation engine
        # backtest_exact.py lives at /app/backtest_exact.py inside the container
        import backtest_exact as bt_engine

        progress.message = "Initializing simulation..."

        # Create the simulation instance
        db_url = str(settings.database_url)
        sim = bt_engine.ExactProductionBacktest(db_url, start_date, end_date)

        # Override instruments if specified
        if instruments:
            sim.instruments = instruments
            sim.instrument_configs = {
                ic.symbol: ic for ic in get_enabled_instruments()
                if ic.symbol in instruments
            }

        # Run the full async simulation with progress tracking
        await _run_simulation_with_progress(sim, progress, bt_engine)

        # Build result from simulation output
        _build_result(job_id, sim, start_date, end_date, progress)

    except Exception as e:
        logger.exception("Backtest %s failed", job_id)
        progress.status = "failed"
        progress.error = str(e)


async def _run_simulation_with_progress(sim, progress, bt_engine):
    """Run the ExactProductionBacktest with progress reporting."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

    engine = create_async_engine(sim.db_url, echo=False)

    # We need 90 days before start_date for strategy evaluation
    eval_start = (
        datetime.strptime(sim.start_date, "%Y-%m-%d") - timedelta(days=140)
    ).strftime("%Y-%m-%d")

    # Load all data upfront
    progress.message = "Loading index candles..."
    data = {}
    option_cache = {}
    oi_data = {}

    for symbol in sim.instruments:
        instrument = sim.instrument_configs[symbol]
        progress.message = f"Loading {symbol} index candles..."
        idx = await bt_engine.load_index_candles(engine, symbol, eval_start, sim.end_date)
        data[symbol] = idx

        dates_list = sorted(idx.keys())
        progress.message = f"Loading {symbol} option candles..."
        oc = await bt_engine.load_option_candles_batch(engine, symbol, dates_list)
        option_cache.update(oc)

        progress.message = f"Loading {symbol} OI data..."
        oi = await bt_engine.load_oi_data(engine, symbol, dates_list)
        oi_data[symbol] = oi

    await engine.dispose()

    # Get trading dates
    all_dates_all = sorted(set(
        d for sym in sim.instruments
        for d in data.get(sym, {}).keys()
    ))
    trading_dates = [
        d for d in all_dates_all
        if d >= sim.start_date and d <= sim.end_date
    ]

    progress.total_days = len(trading_dates)
    progress.message = f"Pre-computing strategy evaluator ({len(trading_dates)} trading days)..."
    progress.status = "running"

    # Pre-compute strategy evaluator results (this is the slow part)
    instruments_list = [sim.instrument_configs[s] for s in sim.instruments]

    # Run pre-computation in a thread to avoid blocking the event loop
    await asyncio.to_thread(
        sim.eval_engine.precompute,
        data, option_cache, oi_data, all_dates_all, instruments_list,
    )

    progress.message = "Simulating trading days..."

    # Day-by-day simulation
    prev_closes = {}

    for day_idx, dt in enumerate(trading_dates):
        progress.processed_days = day_idx + 1
        progress.current_date = dt

        # Strategy evaluation for this day
        strat_picks = {}
        inst_win_rates = {}
        for symbol in sim.instruments:
            ranked = sim.eval_engine.evaluate(
                symbol, all_dates_all,
                end_date_str=sim._prev_date(dt, all_dates_all),
                lookback=90,
            )
            top3 = []
            for sname, score, met in ranked[:3]:
                if met.get("total_trades", 0) >= 3 and score >= 25.0:
                    top3.append(sname)
            if not top3:
                top3 = ["TREND_PULLBACK", "MOMENTUM_BREAKOUT", "ORB"]
            strat_picks[symbol] = top3

            best_wr = 0
            for sname, score, met in ranked:
                if sname in top3 and met.get("win_rate", 0) > best_wr:
                    best_wr = met.get("win_rate", 0)
            inst_win_rates[symbol] = max(best_wr, 10)

        instrument_capital = sim._compute_capital_allocation(inst_win_rates)

        # Simulate day (run in thread since it's CPU-bound)
        day_trades = await asyncio.to_thread(
            sim._simulate_day,
            dt, data, option_cache, oi_data, prev_closes,
            strat_picks, instrument_capital,
        )

        day_pnl = sum(t["PnL"] for t in day_trades)
        sim.capital += day_pnl
        sim.equity_curve.append({
            "Date": dt,
            "PnL": round(day_pnl, 2),
            "Capital": round(sim.capital, 2),
            "Trades": len(day_trades),
        })

        for symbol in sim.instruments:
            if dt in data.get(symbol, {}):
                prev_closes[symbol] = float(data[symbol][dt].iloc[-1]["close"])

        sim.all_trades.extend(day_trades)

        # Yield control to event loop periodically
        if day_idx % 5 == 0:
            await asyncio.sleep(0)


def _build_result(job_id, sim, start_date, end_date, progress):
    """Build BacktestResult from simulation output."""
    total_trades = len(sim.all_trades)
    winners = sum(1 for t in sim.all_trades if t["PnL"] > 0)
    losers = total_trades - winners
    total_pnl = sum(t["PnL"] for t in sim.all_trades)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0

    gp = sum(t["PnL"] for t in sim.all_trades if t["PnL"] > 0)
    gl = abs(sum(t["PnL"] for t in sim.all_trades if t["PnL"] <= 0)) or 1
    profit_factor = gp / gl

    peak = sim.starting_capital
    max_dd = 0.0
    for row in sim.equity_curve:
        cap = row["Capital"]
        if cap > peak:
            peak = cap
        dd = peak - cap
        if dd > max_dd:
            max_dd = dd

    edf_pnls = [r["PnL"] for r in sim.equity_curve]
    if len(edf_pnls) > 1 and np.std(edf_pnls) > 0:
        sharpe = float(np.mean(edf_pnls) / np.std(edf_pnls)) * np.sqrt(252)
    else:
        sharpe = 0.0

    from backtest_exact import SCORE_GATE, NO_NEW_ENTRY_AFTER

    config_used = {
        "initial_capital": sim.starting_capital,
        "score_gate": SCORE_GATE,
        "entry_cutoff": str(NO_NEW_ENTRY_AFTER),
        "instruments": sim.instruments,
        "strategies": "all 6 — selected via 90-day rolling evaluation",
        "max_trades_per_day": settings.max_trades_per_day,
        "max_concurrent_positions": settings.max_concurrent_positions,
        "max_concurrent_per_instrument": settings.max_concurrent_per_instrument,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "consecutive_loss_limit": settings.consecutive_loss_limit,
        "risk_per_trade_pct": settings.risk_per_trade_pct,
        "v2_max_hold_minutes": settings.v2_max_hold_minutes,
    }

    result = BacktestResult(
        job_id=job_id,
        start_date=start_date,
        end_date=end_date,
        instruments=sim.instruments,
        initial_capital=sim.starting_capital,
        ending_capital=round(sim.capital, 2),
        total_pnl=round(total_pnl, 2),
        return_pct=round((sim.capital - sim.starting_capital) / sim.starting_capital * 100, 2),
        total_trades=total_trades,
        winners=winners,
        losers=losers,
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd / sim.starting_capital * 100, 2),
        trades=sim.all_trades,
        equity_curve=sim.equity_curve,
        config_used=config_used,
    )
    _job_results[job_id] = result
    progress.status = "completed"
    progress.message = f"Done — {total_trades} trades, PnL: ₹{total_pnl:+,.0f}"


# ── Excel export ──────────────────────────────────────────────────────

def generate_excel(job_id: str) -> Optional[bytes]:
    """Generate Excel report for a completed backtest. Returns bytes."""
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
        "Day Type", "Allocated Capital",
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

        # Sheet 3: Strategy Summary
        strat = tdf.groupby("Strategy").agg(
            Trades=("PnL", "count"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
            Losers=("Result", lambda x: (x == "LOSS").sum()),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
            Avg_Score=("Score", "mean"),
            Best_Trade=("PnL", "max"),
            Worst_Trade=("PnL", "min"),
        ).reset_index()
        strat["Win_Rate_%"] = (strat["Winners"] / strat["Trades"] * 100).round(1)
        gross_profit = tdf[tdf["PnL"] > 0].groupby("Strategy")["PnL"].sum()
        gross_loss = tdf[tdf["PnL"] <= 0].groupby("Strategy")["PnL"].sum().abs()
        strat["Profit_Factor"] = (gross_profit / gross_loss.replace(0, 1)).round(2)
        strat = strat.fillna(0)
        strat.to_excel(writer, sheet_name="By Strategy", index=False)

        # Sheet 4: Instrument Summary
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

        # Sheet 5: By Exit Reason
        exit_s = tdf.groupby("Exit Reason").agg(
            Count=("PnL", "count"),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
        ).reset_index()
        exit_s["Win_Rate_%"] = (exit_s["Winners"] / exit_s["Count"] * 100).round(1)
        exit_s.to_excel(writer, sheet_name="By Exit Reason", index=False)

        # Sheet 6: By Direction
        dir_s = tdf.groupby("Direction").agg(
            Trades=("PnL", "count"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
        ).reset_index()
        dir_s["Win_Rate_%"] = (dir_s["Winners"] / dir_s["Trades"] * 100).round(1)
        dir_s.to_excel(writer, sheet_name="By Direction", index=False)

        # Sheet 7: Monthly Summary
        tdf_copy = tdf.copy()
        tdf_copy["Month"] = pd.to_datetime(tdf_copy["Date"]).dt.to_period("M").astype(str)
        monthly = tdf_copy.groupby("Month").agg(
            Trading_Days=("Date", "nunique"),
            Trades=("PnL", "count"),
            Winners=("Result", lambda x: (x == "WIN").sum()),
            Total_PnL=("PnL", "sum"),
            Avg_PnL=("PnL", "mean"),
        ).reset_index()
        monthly["Win_Rate_%"] = (monthly["Winners"] / monthly["Trades"] * 100).round(1)
        monthly.to_excel(writer, sheet_name="Monthly Summary", index=False)

        # Sheet 8: Capital Curve
        edf.to_excel(writer, sheet_name="Capital Curve", index=False)

        # Sheet 9: Performance Summary
        perf = pd.DataFrame([
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
            {"Metric": "Sharpe Ratio (Ann.)", "Value": f"{result.sharpe_ratio:.2f}"},
            {"Metric": "Max Drawdown", "Value": f"₹{result.max_drawdown:,.2f}"},
            {"Metric": "Max Drawdown %", "Value": f"{result.max_drawdown_pct:.2f}%"},
            {"Metric": "Instruments", "Value": ", ".join(result.instruments)},
        ])
        perf.to_excel(writer, sheet_name="Performance Summary", index=False)

        # Sheet 10: Config
        cfg = pd.DataFrame([
            {"Key": k, "Value": str(v)} for k, v in result.config_used.items()
        ])
        cfg.to_excel(writer, sheet_name="Config", index=False)

    buf.seek(0)
    return buf.read()
