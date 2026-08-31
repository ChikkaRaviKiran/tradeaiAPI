"""SQLite persistence for option-chain snapshots.

Deliberately separate from the trading engine's PostgreSQL store. The Market
Story page recomputes its entire timeline from raw snapshots on every request,
which is a local read-heavy workload over a single table, and keeping it in its
own file means the feature can be copied, backed up or thrown away without a
migration.

Point ``POSITIONING_DB_PATH`` at an existing database to read snapshots that
were captured elsewhere.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]

DB_PATH = Path(
    os.environ.get("POSITIONING_DB_PATH")
    or (BASE_DIR / "data" / "positioning.db")
).expanduser()

_LOCK = threading.Lock()

# `source` records HOW the row was obtained: 'live' is a poll of the current
# chain, 'rolling' is a session rebuilt afterwards from the expired-options
# API. The two are not interchangeable - a rebuild has bar closes where a poll
# has last traded price, and a narrower ladder - so a reader that cannot tell
# them apart would eventually compare one against the other and believe it.
SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    symbol TEXT NOT NULL,
    expiry TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    strike REAL NOT NULL,
    spot REAL NOT NULL,
    ce_oi REAL, ce_prev_oi REAL, ce_volume REAL, ce_ltp REAL, ce_iv REAL,
    pe_oi REAL, pe_prev_oi REAL, pe_volume REAL, pe_ltp REAL, pe_iv REAL,
    source TEXT NOT NULL DEFAULT 'live',
    PRIMARY KEY (symbol, captured_at, strike)
);
CREATE INDEX IF NOT EXISTS idx_chain_snap_day
    ON chain_snapshots(symbol, captured_at);
"""

_SNAP_COLS = ("symbol", "expiry", "captured_at", "strike", "spot",
              "ce_oi", "ce_prev_oi", "ce_volume", "ce_ltp", "ce_iv",
              "pe_oi", "pe_prev_oi", "pe_volume", "pe_ltp", "pe_iv",
              "source")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _LOCK, _connect() as conn:
        conn.executescript(SCHEMA)
        # CREATE TABLE IF NOT EXISTS does nothing to a table that already
        # exists, so a database written by an older build keeps the old shape
        # until it is widened here.
        have = {r["name"] for r in conn.execute("PRAGMA table_info(chain_snapshots)")}
        if have and "source" not in have:
            conn.execute("ALTER TABLE chain_snapshots ADD COLUMN source"
                         " TEXT NOT NULL DEFAULT 'live'")


def save_chain_snapshot(snapshot: dict[str, Any]) -> int:
    """Persist one normalised chain reading. Returns rows written.

    INSERT OR REPLACE rather than INSERT: polling the same five-minute bucket
    twice is a correction, not a second observation, and a monitor that
    double-counts a bucket would report open interest changing when only the
    poller hiccuped.
    """
    source = snapshot.get("source") or "live"
    rows = [
        (snapshot["symbol"], snapshot["expiry"], snapshot["captured_at"],
         float(r["strike"]), float(snapshot["spot"]),
         r["ce"]["oi"], r["ce"]["prev_oi"], r["ce"]["volume"],
         r["ce"]["ltp"], r["ce"]["iv"],
         r["pe"]["oi"], r["pe"]["prev_oi"], r["pe"]["volume"],
         r["pe"]["ltp"], r["pe"]["iv"], source)
        for r in snapshot["strikes"]
    ]
    if not rows:
        return 0
    placeholders = ",".join("?" * len(_SNAP_COLS))
    with _LOCK, closing(_connect()) as conn, conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO chain_snapshots ({','.join(_SNAP_COLS)})"
            f" VALUES ({placeholders})",
            rows,
        )
    return len(rows)


def load_chain_day(session_date: str, symbol: str = "NIFTY") -> list[dict[str, Any]]:
    """Every strike of every bucket for one session, in time then strike order.

    The whole day rather than a sliding tail, because the detectors compare the
    current bucket against several earlier ones and against the session open.
    """
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM chain_snapshots WHERE symbol=? AND captured_at LIKE ?"
            " ORDER BY captured_at, strike",
            (symbol, f"{session_date}%"),
        ).fetchall()
    return [dict(r) for r in rows]


def chain_day_sources(session_date: str, symbol: str = "NIFTY") -> dict[str, int]:
    """How each bucket of a session was obtained -> bucket count per source.

    Buckets, not rows, because a session rebuilt at a narrower ladder has fewer
    rows per bucket and a row count would read as thinner coverage.
    """
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT source, COUNT(DISTINCT captured_at) AS n FROM chain_snapshots"
            " WHERE symbol=? AND captured_at LIKE ? GROUP BY source",
            (symbol, f"{session_date}%"),
        ).fetchall()
    return {r["source"]: r["n"] for r in rows}


def delete_chain_day(session_date: str, symbol: str = "NIFTY",
                     source: str = "rolling") -> int:
    """Drop one session's rows for one source. Returns rows removed.

    Only ever used to re-run a rebuild. Live polls are the one thing in this
    schema that cannot be recreated, so the source filter has no default that
    would delete them.
    """
    with _LOCK, closing(_connect()) as conn, conn:
        cur = conn.execute(
            "DELETE FROM chain_snapshots WHERE symbol=? AND captured_at LIKE ?"
            " AND source=?",
            (symbol, f"{session_date}%", source),
        )
    return cur.rowcount


def latest_chain_bucket(symbol: str = "NIFTY") -> str | None:
    """Timestamp of the most recent snapshot, or None if nothing is stored."""
    with _LOCK, closing(_connect()) as conn:
        row = conn.execute(
            "SELECT MAX(captured_at) AS at FROM chain_snapshots WHERE symbol=?",
            (symbol,),
        ).fetchone()
    return row["at"] if row and row["at"] else None


def chain_session_dates(symbol: str = "NIFTY", limit: int = 30) -> list[str]:
    """Sessions that have at least one snapshot, newest first."""
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT DISTINCT substr(captured_at, 1, 10) AS d FROM chain_snapshots"
            " WHERE symbol=? ORDER BY d DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return [r["d"] for r in rows]
