#!/bin/bash
echo "=== Most recent option_candles dates ==="
echo "SELECT instrument, date, COUNT(DISTINCT strike) AS strikes, COUNT(*) AS bars FROM option_candles WHERE instrument IN ('NIFTY','SENSEX') AND date >= '2026-05-20' GROUP BY instrument, date ORDER BY date DESC, instrument LIMIT 30;" | docker exec -i tradeai-postgres psql -U tradeai -d tradeai
