"""Run ALL strategies backtest over 6 months with time-window analysis.

Tests all 10 strategies in the analysis engine against NIFTY data
with uniform 20% SL exit framework. Outputs per-strategy + per-window results.

Usage:
    python run_all_strategies_bt.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import asyncpg
import pandas as pd
from datetime import date, timedelta
from collections import defaultdict

from app.analysis.engine import StrategyTester, TimeWindow, STRATEGIES
from app.engine.feature_engine import FeatureEngine

DB_DSN = None  # Use explicit params below (password has @)

# 6-month window (Oct 2025 - Mar 2026)
START_DATE = date(2025, 10, 1)
END_DATE = date(2026, 3, 31)

NIFTY_STRIKE_INTERVAL = 50
NIFTY_LOT_SIZE = 65

SENSEX_STRIKE_INTERVAL = 100
SENSEX_LOT_SIZE = 20


async def load_index_candles(conn, instrument, start, end):
    """Load 1-min candles grouped by date."""
    rows = await conn.fetch(
        """SELECT date, timestamp, open, high, low, close, volume
           FROM index_candles
           WHERE instrument = $1 AND date >= $2 AND date <= $3
           ORDER BY timestamp""",
        instrument, str(start), str(end),
    )
    if not rows:
        return {}

    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    result = {}
    for d, candles in by_date.items():
        df = pd.DataFrame([dict(r) for r in candles])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        result[d] = df

    return result


async def load_option_candles(conn, instrument, start, end):
    """Load option candles into cache dict."""
    rows = await conn.fetch(
        """SELECT date, timestamp, strike, option_type, open, high, low, close, volume
           FROM option_candles
           WHERE instrument = $1 AND date >= $2 AND date <= $3
           ORDER BY timestamp""",
        instrument, str(start), str(end),
    )
    cache = {}
    by_key = defaultdict(list)
    for r in rows:
        key = (instrument, r["date"], float(r["strike"]), r["option_type"])
        by_key[key].append(r)

    for key, candles in by_key.items():
        df = pd.DataFrame([dict(r) for r in candles])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        cache[key] = df

    return cache


async def run():
    print(f"Connecting to DB...")
    conn = await asyncpg.connect(
        user="tradeai", password="TradeAI@6724",
        host="localhost", port=5432, database="tradeai",
    )

    for instrument, strike_int, lot_size in [
        ("NIFTY", NIFTY_STRIKE_INTERVAL, NIFTY_LOT_SIZE),
        ("SENSEX", SENSEX_STRIKE_INTERVAL, SENSEX_LOT_SIZE),
    ]:
        print(f"\n{'='*80}")
        print(f"  INSTRUMENT: {instrument} | {START_DATE} to {END_DATE}")
        print(f"{'='*80}")

        print(f"Loading {instrument} index candles...")
        daily_candles = await load_index_candles(conn, instrument, START_DATE, END_DATE)
        print(f"  {len(daily_candles)} trading days")

        print(f"Loading {instrument} option candles...")
        option_cache = await load_option_candles(conn, instrument, START_DATE, END_DATE)
        print(f"  {len(option_cache)} option series")

        if not daily_candles:
            print(f"  No data for {instrument}, skipping")
            continue

        # Test each strategy individually
        fe = FeatureEngine()
        strategy_names = list(STRATEGIES.keys())

        # Collect all trades across strategies
        all_trades = []

        for strat_name in strategy_names:
            tester = StrategyTester(fe, strategy_filter=[strat_name])
            strat_trades = []

            dates_sorted = sorted(daily_candles.keys())
            for d in dates_sorted:
                df = daily_candles[d]
                results = tester.test_day(
                    df, instrument, strike_int, lot_size,
                    option_cache, date_str=d,
                )
                strat_trades.extend(results)

            all_trades.extend(strat_trades)

            # Quick summary for this strategy
            if strat_trades:
                wins = [t for t in strat_trades if t.pnl > 0]
                losses = [t for t in strat_trades if t.pnl <= 0]
                total_pnl = sum(t.pnl * t.lot_size for t in strat_trades)
                gross_win = sum(t.pnl * t.lot_size for t in wins) if wins else 0
                gross_loss = abs(sum(t.pnl * t.lot_size for t in losses)) if losses else 0
                pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
                wr = len(wins) / len(strat_trades) * 100
                avg_r = sum(t.r_multiple for t in strat_trades) / len(strat_trades)

                print(f"\n  {strat_name}: {len(strat_trades)} trades | "
                      f"PnL: ₹{total_pnl:+,.0f} | PF: {pf:.2f} | WR: {wr:.1f}% | Avg R: {avg_r:.2f}")
            else:
                print(f"\n  {strat_name}: 0 trades")

        # ── Detailed per-strategy per-window analysis ────────────────

        print(f"\n\n{'='*80}")
        print(f"  DETAILED: STRATEGY × TIME WINDOW ({instrument})")
        print(f"{'='*80}")

        # Group trades
        strat_window = defaultdict(list)
        for t in all_trades:
            strat_window[(t.strategy, t.time_window)].append(t)

        # Print header
        windows = TimeWindow.ALL
        header = f"{'Strategy':<25}"
        for w in windows:
            header += f" | {w:>12}"
        header += f" | {'TOTAL':>12}"
        print(f"\n{header}")
        print("-" * len(header))

        # For each strategy, print PnL per window
        for sname in strategy_names:
            strat_all = [t for t in all_trades if t.strategy == sname]
            if not strat_all:
                continue

            row_trades = f"{sname:<25}"
            row_pnl = f"{'  PnL':<25}"
            row_pf = f"{'  PF':<25}"
            row_wr = f"{'  WR%':<25}"
            row_avgr = f"{'  AvgR':<25}"

            total_count = 0
            total_pnl = 0
            total_win_pnl = 0
            total_loss_pnl = 0
            total_wins = 0

            for w in windows:
                trades_in = strat_window.get((sname, w), [])
                n = len(trades_in)
                total_count += n

                if n == 0:
                    row_trades += f" | {'--':>12}"
                    row_pnl += f" | {'--':>12}"
                    row_pf += f" | {'--':>12}"
                    row_wr += f" | {'--':>12}"
                    row_avgr += f" | {'--':>12}"
                    continue

                ws = [t for t in trades_in if t.pnl > 0]
                ls = [t for t in trades_in if t.pnl <= 0]
                pnl = sum(t.pnl * t.lot_size for t in trades_in)
                gw = sum(t.pnl * t.lot_size for t in ws) if ws else 0
                gl = abs(sum(t.pnl * t.lot_size for t in ls)) if ls else 0
                pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0)
                wr = len(ws) / n * 100
                ar = sum(t.r_multiple for t in trades_in) / n

                total_pnl += pnl
                total_win_pnl += gw
                total_loss_pnl += gl
                total_wins += len(ws)

                row_trades += f" | {n:>12}"
                row_pnl += f" | {pnl:>+12,.0f}"
                pf_str = f"{pf:.2f}" if pf < 99 else "INF"
                row_pf += f" | {pf_str:>12}"
                row_wr += f" | {wr:>11.1f}%"
                row_avgr += f" | {ar:>+12.2f}"

            # Totals
            t_pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else (99.0 if total_win_pnl > 0 else 0)
            t_wr = total_wins / total_count * 100 if total_count > 0 else 0
            t_ar = sum(t.r_multiple for t in strat_all) / len(strat_all) if strat_all else 0

            row_trades += f" | {total_count:>12}"
            row_pnl += f" | {total_pnl:>+12,.0f}"
            t_pf_str = f"{t_pf:.2f}" if t_pf < 99 else "INF"
            row_pf += f" | {t_pf_str:>12}"
            row_wr += f" | {t_wr:>11.1f}%"
            row_avgr += f" | {t_ar:>+12.2f}"

            print(row_trades)
            print(row_pnl)
            print(row_pf)
            print(row_wr)
            print(row_avgr)
            print()

        # ── PROFITABLE WINDOWS SUMMARY ────────────────────────────────
        print(f"\n{'='*80}")
        print(f"  PROFITABLE WINDOWS (PF > 1.2 AND trades >= 10) — {instrument}")
        print(f"{'='*80}")
        print(f"{'Strategy':<25} {'Window':<15} {'Trades':>7} {'PnL':>12} {'PF':>7} {'WR%':>7} {'AvgR':>7}")
        print("-" * 85)

        profitable = []
        for (sname, window), trades_in in sorted(strat_window.items()):
            if len(trades_in) < 10:
                continue
            ws = [t for t in trades_in if t.pnl > 0]
            ls = [t for t in trades_in if t.pnl <= 0]
            pnl = sum(t.pnl * t.lot_size for t in trades_in)
            gw = sum(t.pnl * t.lot_size for t in ws) if ws else 0
            gl = abs(sum(t.pnl * t.lot_size for t in ls)) if ls else 0
            pf = gw / gl if gl > 0 else 99.0
            wr = len(ws) / len(trades_in) * 100
            ar = sum(t.r_multiple for t in trades_in) / len(trades_in)

            if pf > 1.2:
                profitable.append((sname, window, len(trades_in), pnl, pf, wr, ar))
                print(f"{sname:<25} {window:<15} {len(trades_in):>7} {pnl:>+12,.0f} {pf:>7.2f} {wr:>6.1f}% {ar:>+7.2f}")

        if not profitable:
            print("  (none found)")

        # ── WIN/LOSS STREAKS ──────────────────────────────────────────
        print(f"\n{'='*80}")
        print(f"  MAX DRAWDOWN PER STRATEGY — {instrument}")
        print(f"{'='*80}")

        for sname in strategy_names:
            strat_all = sorted(
                [t for t in all_trades if t.strategy == sname],
                key=lambda t: (t.date, t.entry_time),
            )
            if not strat_all:
                continue

            equity = 100000.0
            peak = equity
            max_dd = 0
            max_dd_pct = 0

            for t in strat_all:
                equity += t.pnl * t.lot_size
                peak = max(peak, equity)
                dd = peak - equity
                dd_pct = dd / peak * 100
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

            final_pnl = equity - 100000
            print(f"  {sname:<25} Final PnL: ₹{final_pnl:>+10,.0f} | Max DD: ₹{max_dd:>8,.0f} ({max_dd_pct:.1f}%)")

    await conn.close()
    print(f"\n\nDone!")


if __name__ == "__main__":
    asyncio.run(run())
