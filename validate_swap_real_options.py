#!/usr/bin/env python3
"""Run inside ubuntu host. Validates the SWAP schedule against REAL option_candles DB.

For each day in swap_schedule.csv:
  1. Pick nearest expiry (>= date) for that instrument
  2. Get spot at entry_time from index_candles -> round to ATM strike
  3. Get CE+PE close at entry_time -> credit
  4. Get CE+PE close at exit_time -> debit (or last available before exit)
  5. Net P&L per lot in points; rupees = pnl_pts * lot * num_lots
"""
import csv, subprocess, json
from datetime import datetime

NIFTY_LOT, SENSEX_LOT = 75, 20
NUM_LOTS = 3

PSQL = ["docker", "exec", "-i", "tradeai-postgres",
        "psql", "-U", "tradeai", "-d", "tradeai", "-A", "-F", "|", "-t", "-c"]


def q(sql):
    r = subprocess.run(PSQL + [sql], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return [line.split("|") for line in r.stdout.strip().split("\n") if line.strip()]


def get_spot_at(sym, date, hhmm):
    # index_candles 1-min bar
    sql = (f"SELECT close FROM index_candles WHERE instrument='{sym}' "
           f"AND date='{date}' AND to_char(timestamp,'HH24:MI')='{hhmm}' "
           f"ORDER BY timestamp LIMIT 1;")
    r = q(sql)
    if not r:
        # fall back: nearest bar in 15 min window
        sql = (f"SELECT close FROM index_candles WHERE instrument='{sym}' "
               f"AND date='{date}' AND to_char(timestamp,'HH24:MI') >= '{hhmm}' "
               f"ORDER BY timestamp LIMIT 1;")
        r = q(sql)
    return float(r[0][0]) if r else None


def get_expiry(sym, date):
    # nearest expiry whose actual date >= our date.
    # expiry stored as DDMMMYY like '28APR26'
    sql = (f"SELECT DISTINCT expiry FROM option_candles "
           f"WHERE instrument='{sym}' AND date='{date}';")
    r = q(sql)
    if not r:
        return None
    # parse each to datetime, pick min >= date
    target = datetime.strptime(date, "%Y-%m-%d").date()
    parsed = []
    for (e,) in r:
        try:
            d = datetime.strptime(e, "%d%b%y").date()
            parsed.append((d, e))
        except ValueError:
            continue
    parsed.sort()
    for d, e in parsed:
        if d >= target:
            return e
    return parsed[0][1] if parsed else None


def get_premium(sym, date, expiry, strike, otype, hhmm):
    sql = (f"SELECT close FROM option_candles WHERE instrument='{sym}' "
           f"AND date='{date}' AND expiry='{expiry}' AND strike={strike} "
           f"AND option_type='{otype}' AND to_char(timestamp,'HH24:MI')='{hhmm}' "
           f"LIMIT 1;")
    r = q(sql)
    if not r:
        # take nearest <= hhmm to handle exit-time, or nearest >= for entry
        sql = (f"SELECT close FROM option_candles WHERE instrument='{sym}' "
               f"AND date='{date}' AND expiry='{expiry}' AND strike={strike} "
               f"AND option_type='{otype}' AND to_char(timestamp,'HH24:MI') <= '{hhmm}' "
               f"ORDER BY timestamp DESC LIMIT 1;")
        r = q(sql)
    return float(r[0][0]) if r else None


def round_strike(spot, sym):
    step = 50 if sym == "NIFTY" else 100
    return int(round(spot / step) * step)


def main():
    rows = list(csv.DictReader(open("swap_schedule.csv")))
    print(f"Validating {len(rows)} days against real option_candles DB\n")
    results = []
    skipped = []
    for i, r in enumerate(rows, 1):
        d, sym, entry, exit_t = r["date"], r["sym"], r["entry"], r["exit"]
        spot = get_spot_at(sym, d, entry)
        if not spot:
            skipped.append((d, sym, "no spot"))
            continue
        strike = round_strike(spot, sym)
        expiry = get_expiry(sym, d)
        if not expiry:
            skipped.append((d, sym, "no expiry"))
            continue
        ce_in = get_premium(sym, d, expiry, strike, "CE", entry)
        pe_in = get_premium(sym, d, expiry, strike, "PE", entry)
        ce_out = get_premium(sym, d, expiry, strike, "CE", exit_t)
        pe_out = get_premium(sym, d, expiry, strike, "PE", exit_t)
        if None in (ce_in, pe_in, ce_out, pe_out):
            skipped.append((d, sym, f"missing legs strike={strike} expiry={expiry}"))
            continue
        credit = ce_in + pe_in
        debit  = ce_out + pe_out
        pnl_pts = credit - debit
        lot = NIFTY_LOT if sym == "NIFTY" else SENSEX_LOT
        rupees = pnl_pts * lot * NUM_LOTS
        pnl_pct = pnl_pts / spot * 100
        results.append({
            "date": d, "sym": sym, "spot": spot, "strike": strike,
            "expiry": expiry, "credit": credit, "debit": debit,
            "pnl_pts": pnl_pts, "pnl_pct": pnl_pct, "rupees": rupees,
        })
        print(f"[{i:>2}/{len(rows)}] {d} {sym:6} K={strike} exp={expiry}  "
              f"credit={credit:7.2f} debit={debit:7.2f} "
              f"pnl={pnl_pts:+7.2f} pts ({pnl_pct:+.3f}%)  ₹{rupees:+,.0f}")

    # Summary
    n = len(results)
    if not n:
        print("No results"); return
    wins = sum(1 for r in results if r["rupees"] > 0)
    total_rup = sum(r["rupees"] for r in results)
    total_pct = sum(r["pnl_pct"] for r in results)
    print(f"\n=== REAL OPTION-DATA RESULT ===")
    print(f"Days validated: {n} (skipped: {len(skipped)})")
    print(f"Wins: {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"Cumulative P&L: {total_pct:+.2f}% / ₹{total_rup:+,.0f}")
    print(f"Avg per day: {total_pct/n:+.3f}% / ₹{total_rup/n:+,.0f}")

    # Monthly
    from collections import defaultdict
    mo = defaultdict(lambda: {"n": 0, "w": 0, "r": 0.0})
    for x in results:
        k = x["date"][:7]
        mo[k]["n"] += 1
        mo[k]["r"] += x["rupees"]
        if x["rupees"] > 0:
            mo[k]["w"] += 1
    print(f"\nMonthly:")
    for k in sorted(mo):
        s = mo[k]
        print(f"  {k}  n={s['n']:>2}  wins={s['w']:>2}  total=₹{s['r']:+,.0f}")

    # Skipped
    if skipped:
        print(f"\nSkipped {len(skipped)} days:")
        for d, sym, reason in skipped:
            print(f"  {d} {sym}: {reason}")

    # Worst
    results.sort(key=lambda r: r["rupees"])
    print(f"\n5 worst days:")
    for x in results[:5]:
        print(f"  {x['date']} {x['sym']}  pnl={x['pnl_pct']:+.3f}%  ₹{x['rupees']:+,.0f}")
    print(f"\n5 best days:")
    for x in results[-5:][::-1]:
        print(f"  {x['date']} {x['sym']}  pnl={x['pnl_pct']:+.3f}%  ₹{x['rupees']:+,.0f}")

    # Save
    with open("swap_real_validation.json", "w") as f:
        json.dump({"results": results, "skipped": skipped,
                   "summary": {"n": n, "wins": wins, "total_rup": total_rup,
                               "total_pct": total_pct}}, f, indent=2)


if __name__ == "__main__":
    main()
