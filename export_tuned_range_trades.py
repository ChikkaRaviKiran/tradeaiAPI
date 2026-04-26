"""Export all trades for the final tuned RANGE_BREAKOUT configuration.

Outputs detailed trade-level data (entry, exit, prices, pnl, R multiple, etc.)
for NIFTY over 2025-10-01 to 2026-03-31.
"""

import os
import sys
import asyncio
from dataclasses import asdict
from datetime import date
from pathlib import Path

import asyncpg
import pandas as pd

# Tuned RB config (applied before importing strategy modules)
os.environ.setdefault("RB_WINDOW_START", "09:45")
os.environ.setdefault("RB_WINDOW_END", "10:30")
os.environ.setdefault("RB_CALL_RSI_MIN", "58")
os.environ.setdefault("RB_PUT_RSI_MAX", "42")
os.environ.setdefault("RB_ADX_THRESHOLD", "20")
os.environ.setdefault("RB_RANGE_PCT_THRESHOLD", "0.80")
os.environ.setdefault("RB_MIN_BODY_RATIO", "0.45")

sys.path.insert(0, os.path.dirname(__file__))

from run_all_strategies_bt import load_index_candles, load_option_candles  # noqa: E402
from app.analysis.engine import StrategyTester  # noqa: E402
from app.engine.feature_engine import FeatureEngine  # noqa: E402

PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_USER = os.getenv("PGUSER", "tradeai")
PG_PASSWORD = os.getenv("PGPASSWORD", "TradeAI@6724")
PG_DATABASE = os.getenv("PGDATABASE", "tradeai")

START_DATE = date(2025, 10, 1)
END_DATE = date(2026, 3, 31)

INSTRUMENT = "NIFTY"
STRIKE_INTERVAL = 50
LOT_SIZE = 65


async def main() -> int:
    conn = await asyncpg.connect(
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
    )

    try:
        daily_candles = await load_index_candles(conn, INSTRUMENT, START_DATE, END_DATE)
        option_cache = await load_option_candles(conn, INSTRUMENT, START_DATE, END_DATE)

        if not daily_candles:
            print("No daily candles found.")
            return 1

        fe = FeatureEngine()
        tester = StrategyTester(fe, strategy_filter=["RANGE_BREAKOUT"])

        trades = []
        for d in sorted(daily_candles.keys()):
            df = daily_candles[d]
            day_trades = tester.test_day(
                df,
                INSTRUMENT,
                STRIKE_INTERVAL,
                LOT_SIZE,
                option_cache,
                date_str=d,
            )
            trades.extend(day_trades)

        rows = [asdict(t) for t in trades]
        trade_df = pd.DataFrame(rows)

        if trade_df.empty:
            print("No trades generated for tuned configuration.")
            return 2

        trade_df["entry_timestamp"] = trade_df["date"].astype(str) + " " + trade_df["entry_time"].astype(str)
        trade_df["exit_timestamp"] = trade_df["date"].astype(str) + " " + trade_df["exit_time"].astype(str)
        trade_df = trade_df[
            [
                "date",
                "instrument",
                "strategy",
                "direction",
                "strike",
                "entry_time",
                "exit_time",
                "entry_timestamp",
                "exit_timestamp",
                "entry_price",
                "exit_price",
                "pnl",
                "pnl_pct",
                "r_multiple",
                "exit_reason",
                "hold_minutes",
                "time_window",
                "day_type",
                "day_type_hindsight",
                "lot_size",
            ]
        ].sort_values(["date", "entry_time"]).reset_index(drop=True)

        summary = pd.DataFrame(
            [
                {
                    "Trades": int(len(trade_df)),
                    "WinRatePct": round((trade_df["pnl"] > 0).mean() * 100, 2),
                    "ProfitFactor": round(
                        (trade_df.loc[trade_df["pnl"] > 0, "pnl"].sum())
                        / max(abs(trade_df.loc[trade_df["pnl"] <= 0, "pnl"].sum()), 1e-9),
                        4,
                    ),
                    "TotalPnL": round(float(trade_df["pnl"].sum()), 2),
                    "AvgR": round(float(trade_df["r_multiple"].mean()), 4),
                }
            ]
        )

        settings = pd.DataFrame(
            [
                {"Param": "RB_WINDOW_START", "Value": os.getenv("RB_WINDOW_START")},
                {"Param": "RB_WINDOW_END", "Value": os.getenv("RB_WINDOW_END")},
                {"Param": "RB_CALL_RSI_MIN", "Value": os.getenv("RB_CALL_RSI_MIN")},
                {"Param": "RB_PUT_RSI_MAX", "Value": os.getenv("RB_PUT_RSI_MAX")},
                {"Param": "RB_ADX_THRESHOLD", "Value": os.getenv("RB_ADX_THRESHOLD")},
                {"Param": "RB_RANGE_PCT_THRESHOLD", "Value": os.getenv("RB_RANGE_PCT_THRESHOLD")},
                {"Param": "RB_MIN_BODY_RATIO", "Value": os.getenv("RB_MIN_BODY_RATIO")},
                {"Param": "Period", "Value": f"{START_DATE} to {END_DATE}"},
                {"Param": "Instrument", "Value": INSTRUMENT},
                {"Param": "LotSize", "Value": LOT_SIZE},
            ]
        )

        out_xlsx = Path("/app/tuned_range_breakout_all_trades.xlsx")
        out_csv = Path("/app/tuned_range_breakout_all_trades.csv")

        trade_df.to_csv(out_csv, index=False)

        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                trade_df.to_excel(writer, sheet_name="All_Trades", index=False)
                summary.to_excel(writer, sheet_name="Summary", index=False)
                settings.to_excel(writer, sheet_name="Tuned_Settings", index=False)
            print(f"XLSX={out_xlsx}")
        except Exception as ex:
            print(f"XLSX_ERROR={ex}")

        print(f"CSV={out_csv}")
        print(f"TRADES={len(trade_df)}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
