"""
Verify the actual expiry day for NIFTY and SENSEX weekly options
from Jan 1, 2026 onwards by querying Dhan's authoritative expiry-list API.

This bypasses any potential local-DB staleness and asks Dhan directly:
"What are the valid expiry dates for this underlying?"
"""
import os
import sys
import json
import time
from datetime import date, datetime
from pathlib import Path

import requests

# Load .env
ENV = Path(__file__).parent / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

CLIENT_ID = os.environ["DHAN_CLIENT_ID"]
TOKEN = os.environ["DHAN_ACCESS_TOKEN"]

# Underlying security IDs (NSE IDX / BSE IDX) per Dhan docs
UNDERLYINGS = [
    ("NIFTY",  13,    "IDX_I"),   # NSE NIFTY 50 index
    ("SENSEX", 51,    "IDX_I"),   # BSE SENSEX index
]

HEADERS = {
    "access-token": TOKEN,
    "client-id": CLIENT_ID,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

URL = "https://api.dhan.co/v2/optionchain/expirylist"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

print("=" * 78)
print("DHAN AUTHORITATIVE EXPIRY-LIST VERIFICATION")
print("Source: api.dhan.co/v2/optionchain/expirylist  (live broker API)")
print("=" * 78)

CUTOFF = date(2026, 1, 1)

for sym, sec_id, seg in UNDERLYINGS:
    body = {"UnderlyingScrip": sec_id, "UnderlyingSeg": seg}
    print(f"\n--- {sym}  (security_id={sec_id}, segment={seg}) ---")
    try:
        r = requests.post(URL, headers=HEADERS, json=body, timeout=15)
    except Exception as e:
        print(f"  ERROR: request failed: {e}")
        continue

    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:300]}")
        continue

    js = r.json()
    expiries = js.get("data") or js.get("expirylist") or js
    if not isinstance(expiries, list):
        print(f"  Unexpected response: {json.dumps(js)[:300]}")
        continue

    # Parse and filter from Jan 1, 2026
    parsed = []
    for e in expiries:
        try:
            d = datetime.strptime(str(e), "%Y-%m-%d").date()
        except Exception:
            continue
        if d >= CUTOFF:
            parsed.append(d)
    parsed.sort()

    if not parsed:
        print("  No expiries on or after 2026-01-01 returned.")
        continue

    # Group by day-of-week and show all
    by_dow = {}
    for d in parsed:
        by_dow.setdefault(d.weekday(), []).append(d)

    print(f"  Total expiries from 2026-01-01 onwards: {len(parsed)}")
    print("  Day-of-week distribution:")
    for dow in sorted(by_dow.keys()):
        print(f"    {DAY_NAMES[dow]:5s}: {len(by_dow[dow]):3d} contracts")

    print("\n  Full expiry list (first 25):")
    for d in parsed[:25]:
        print(f"    {d.strftime('%Y-%m-%d')}  {DAY_NAMES[d.weekday()]}")

    if len(parsed) > 25:
        print(f"    ... and {len(parsed)-25} more")

    # Pause between calls to be polite
    time.sleep(1)

print("\n" + "=" * 78)
print("DONE.  If day-of-week above does NOT match our DB option_candles, our DB")
print("ingestion is stale and post-regime analysis needs re-running.")
print("=" * 78)
