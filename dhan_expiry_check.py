#!/usr/bin/env python3
"""Verify NIFTY/SENSEX weekly expiry days from Dhan instrument master."""
import csv, sys
from datetime import datetime, date
from collections import defaultdict

path = "/tmp/dhan_master.csv"
buckets = defaultdict(set)   # symbol -> set of (date, weekday-name)
with open(path) as f:
    rdr = csv.reader(f)
    next(rdr)
    for row in rdr:
        if len(row) < 11: continue
        seg, _, _, instr, _, tsym, _, custom, expiry, strike, opt, *_ = row
        if instr != "OPTIDX": continue
        if opt != "CE": continue
        # Pull base symbol
        sym = tsym.split("-")[0].upper()
        if sym not in ("NIFTY","SENSEX"): continue
        try:
            d = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < date(2025, 9, 1) or d > date(2026, 7, 31): continue
        buckets[sym].add(d)

for sym in ("NIFTY","SENSEX"):
    print(f"\n=== {sym}  weekly+monthly expiries Sep 2025 → Jul 2026 ===")
    days_by_dow = defaultdict(int)
    for d in sorted(buckets[sym]):
        dow = d.strftime("%a")
        days_by_dow[dow] += 1
        print(f"  {d.isoformat()}  {dow}")
    print(f"  by DOW: {dict(days_by_dow)}")
