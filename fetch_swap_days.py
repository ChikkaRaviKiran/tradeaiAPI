"""Fetch missing intraday bars: SENSEX on Tue, NIFTY on Thu, for the swap scenario."""
import json, time
from datetime import datetime, timedelta, date
from pathlib import Path
from app.data.angelone_client import AngelOneClient

INTRADAY_PATH = Path(__file__).parent / "intraday_full_day_cache.json"
NIFTY_TOKEN, SENSEX_TOKEN = "99926000", "99919000"

# Date window same as cache
START = date(2026, 1, 1)
END   = date(2026, 5, 22)

cache = json.loads(INTRADAY_PATH.read_text())

needed = []  # (date_str, sym)
d = START
while d <= END:
    wd = d.weekday()
    d_str = d.strftime("%Y-%m-%d")
    if wd == 1:  # Tue -> SENSEX
        if d_str not in cache.get("SENSEX", {}) or len(cache["SENSEX"].get(d_str, [])) < 30:
            needed.append((d_str, "SENSEX"))
    elif wd == 3:  # Thu -> NIFTY
        if d_str not in cache.get("NIFTY", {}) or len(cache["NIFTY"].get(d_str, [])) < 30:
            needed.append((d_str, "NIFTY"))
    d += timedelta(days=1)

print(f"Need to fetch {len(needed)} day(s)")
client = AngelOneClient()
for i, (d_str, sym) in enumerate(needed, 1):
    token = NIFTY_TOKEN if sym == "NIFTY" else SENSEX_TOKEN
    exch  = "NSE" if sym == "NIFTY" else "BSE"
    d0 = datetime.strptime(d_str, "%Y-%m-%d")
    f = d0.replace(hour=9, minute=15).strftime("%Y-%m-%d %H:%M")
    t = d0.replace(hour=15, minute=30).strftime("%Y-%m-%d %H:%M")
    try:
        bars = client.get_candle_data(
            symbol_token=token, exchange=exch,
            interval="FIVE_MINUTE", from_date=f, to_date=t) or []
    except Exception as e:
        print(f"  [{i}/{len(needed)}] {d_str} {sym}: ERR {e}")
        bars = []
        time.sleep(1.0)
    rows = [[c.timestamp.strftime("%H:%M"), c.open, c.high, c.low, c.close] for c in bars]
    cache.setdefault(sym, {})[d_str] = rows
    print(f"  [{i}/{len(needed)}] {d_str} {sym}: {len(rows)} bars")
    if i % 10 == 0:
        INTRADAY_PATH.write_text(json.dumps(cache))
    time.sleep(0.4)
INTRADAY_PATH.write_text(json.dumps(cache))
print("Saved.")
