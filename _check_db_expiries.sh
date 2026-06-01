#!/bin/bash
echo "=== SENSEX expiries Jan-May 2026 with day-of-week derived from MAX(date) bar ==="
echo "SELECT instrument, expiry, MAX(date) AS last_traded, TO_CHAR(MAX(date)::date,'Dy') AS dow, COUNT(*) AS bars, COUNT(DISTINCT date) AS trading_days FROM option_candles WHERE instrument='SENSEX' AND date BETWEEN '2026-01-01' AND '2026-05-31' GROUP BY instrument, expiry ORDER BY MAX(date)::date;" | docker exec -i tradeai-postgres psql -U tradeai -d tradeai
