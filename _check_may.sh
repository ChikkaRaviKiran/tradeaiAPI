#!/bin/bash
SQL="SELECT instrument, date, COUNT(DISTINCT strike) AS strikes, COUNT(*) AS bars FROM option_candles WHERE date >= '2026-05-01' AND date <= '2026-05-29' GROUP BY instrument, date ORDER BY date, instrument;"
docker exec tradeai-postgres psql -U tradeai -d tradeai -c "$SQL"
