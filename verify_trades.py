"""Verify backtest trades against actual DB option candle data.

For each trade: show the DB candle at entry/exit time, compare OHLC with backtest prices.
Deduplicates by taking the candle with highest volume per minute.
Shows a window of candles around entry and exit (5 before, 5 after).
"""
import asyncio
import asyncpg
from collections import defaultdict


# Trades from the backtest result (screenshot)
TRADES = [
    {"date": "2026-04-02", "inst": "NIFTY",  "strike": 22250, "opt": "CE", "entry_time": "10:43", "exit_time": "13:57", "bt_entry": 300.88, "bt_exit": 465.75},
    {"date": "2026-04-02", "inst": "SENSEX", "strike": 71700, "opt": "PE", "entry_time": "10:03", "exit_time": "10:56", "bt_entry": 283.1,  "bt_exit": 226.48},
    {"date": "2026-04-06", "inst": "NIFTY",  "strike": 22550, "opt": "PE", "entry_time": "11:14", "exit_time": "11:21", "bt_entry": 204.42, "bt_exit": 163.54},
    {"date": "2026-04-06", "inst": "SENSEX", "strike": 73000, "opt": "CE", "entry_time": "11:24", "exit_time": "12:48", "bt_entry": 921.78, "bt_exit": 1148.7},
    {"date": "2026-04-07", "inst": "NIFTY",  "strike": 22900, "opt": "PE", "entry_time": "10:56", "exit_time": "11:05", "bt_entry": 102.57, "bt_exit": 103.08},
    {"date": "2026-04-07", "inst": "SENSEX", "strike": 73900, "opt": "CE", "entry_time": "10:04", "exit_time": "14:43", "bt_entry": 839.01, "bt_exit": 1029.75},
    {"date": "2026-04-08", "inst": "NIFTY",  "strike": 24000, "opt": "CE", "entry_time": "10:57", "exit_time": "15:10", "bt_entry": 213.26, "bt_exit": 221.85},
    {"date": "2026-04-08", "inst": "SENSEX", "strike": 77600, "opt": "CE", "entry_time": "14:45", "exit_time": "15:10", "bt_entry": 412.99, "bt_exit": 400.0},
    {"date": "2026-04-09", "inst": "NIFTY",  "strike": 23850, "opt": "CE", "entry_time": "11:31", "exit_time": "13:21", "bt_entry": 220.48, "bt_exit": 176.38},
    {"date": "2026-04-09", "inst": "SENSEX", "strike": 76800, "opt": "PE", "entry_time": "10:42", "exit_time": "10:47", "bt_entry": 210.28, "bt_exit": 211.33},
    {"date": "2026-04-10", "inst": "NIFTY",  "strike": 24000, "opt": "CE", "entry_time": "14:52", "exit_time": "15:10", "bt_entry": 149.48, "bt_exit": 179.0},
    {"date": "2026-04-10", "inst": "SENSEX", "strike": 77400, "opt": "CE", "entry_time": "14:51", "exit_time": "15:10", "bt_entry": 721.64, "bt_exit": 853.55},
]

WINDOW = 3  # candles before/after entry and exit to show


def dedup(rows):
    """Keep highest-volume candle per minute."""
    by_min = defaultdict(list)
    for r in rows:
        hhmm = r["timestamp"].strftime("%H:%M")
        by_min[hhmm].append(r)
    result = []
    for hhmm in sorted(by_min):
        best = max(by_min[hhmm], key=lambda x: x["volume"])
        result.append(best)
    return result


async def main():
    conn = await asyncpg.connect(
        host="localhost", port=15432, user="tradeai",
        password="TradeAI@6724", database="tradeai",
    )

    for t in TRADES:
        print("=" * 110)
        print(f"TRADE: {t['date']} | {t['inst']} {t['strike']} {t['opt']} | "
              f"Entry {t['entry_time']} @{t['bt_entry']} | Exit {t['exit_time']} @{t['bt_exit']}")
        print("-" * 110)

        rows = await conn.fetch("""
            SELECT timestamp, open, high, low, close, volume, expiry
            FROM option_candles
            WHERE instrument = $1
              AND date = $2
              AND strike = $3
              AND option_type = $4
            ORDER BY timestamp
        """, t["inst"], t["date"], float(t["strike"]), t["opt"])

        if not rows:
            print("  *** NO DATA IN DB ***\n")
            continue

        candles = dedup(rows)
        expiry_label = candles[0]["expiry"]
        print(f"  DB Expiry: {expiry_label}  |  Candles: {len(candles)} (raw: {len(rows)})")

        # Build index
        idx = {}
        for i, c in enumerate(candles):
            idx[c["timestamp"].strftime("%H:%M")] = i

        entry_i = idx.get(t["entry_time"])
        exit_i = idx.get(t["exit_time"])

        # Determine which candles to show
        show = set()
        if entry_i is not None:
            for j in range(max(0, entry_i - WINDOW), min(len(candles), entry_i + WINDOW + 1)):
                show.add(j)
        if exit_i is not None:
            for j in range(max(0, exit_i - WINDOW), min(len(candles), exit_i + WINDOW + 1)):
                show.add(j)

        if not show:
            print(f"  *** Entry {t['entry_time']} / Exit {t['exit_time']} NOT FOUND ***")
            # Show first/last 3
            for j in list(range(min(3, len(candles)))) + list(range(max(0, len(candles)-3), len(candles))):
                show.add(j)

        print(f"\n  {'Time':<8} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>10}  Note")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*30}")

        prev_j = -2
        for j in sorted(show):
            if j > prev_j + 1:
                print(f"  {'...':^8}")
            prev_j = j
            c = candles[j]
            hhmm = c["timestamp"].strftime("%H:%M")
            marker = ""
            if hhmm == t["entry_time"]:
                marker = f"<-- ENTRY (bt open={t['bt_entry']})"
            if hhmm == t["exit_time"]:
                marker = f"<-- EXIT  (bt={t['bt_exit']})"
            print(f"  {hhmm:<8} {c['open']:>10.2f} {c['high']:>10.2f} {c['low']:>10.2f} {c['close']:>10.2f} {c['volume']:>10}  {marker}")

        # Comparison
        print()
        if entry_i is not None:
            ec = candles[entry_i]
            db_o = ec["open"]
            diff = abs(db_o - t["bt_entry"])
            sym = "=" if diff < 0.01 else ("~" if diff < 1.0 else "X")
            print(f"  ENTRY: DB open={db_o:.2f}  BT={t['bt_entry']:.2f}  diff={diff:.2f}  [{sym}]")
        else:
            print(f"  ENTRY: *** {t['entry_time']} NOT IN DB ***")

        if exit_i is not None:
            xc = candles[exit_i]
            # Check which OHLC field the backtest exit matches
            fields = {"open": xc["open"], "high": xc["high"], "low": xc["low"], "close": xc["close"]}
            best_field = min(fields, key=lambda f: abs(fields[f] - t["bt_exit"]))
            best_val = fields[best_field]
            diff = abs(best_val - t["bt_exit"])
            sym = "=" if diff < 0.01 else ("~" if diff < 1.0 else "X")
            print(f"  EXIT:  DB {best_field}={best_val:.2f}  BT={t['bt_exit']:.2f}  diff={diff:.2f}  [{sym}]")
        else:
            print(f"  EXIT:  *** {t['exit_time']} NOT IN DB ***")

        print()

    await conn.close()


asyncio.run(main())
