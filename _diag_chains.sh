#!/bin/bash
# Diagnose heavy chains and check psycopg2 availability on server.
python3 -c 'import psycopg2; print("psycopg2", psycopg2.__version__)' 2>&1 || echo "NO_PSYCOPG2"
echo "--- top 10 (date,instrument) by row-count Apr-May 2026 ---"
docker exec tradeai-postgres psql -U tradeai -d tradeai -t -A -F"|" -c \
  "select date, instrument, count(*) as rows from option_candles where date between '2026-04-15' and '2026-05-26' group by date, instrument order by rows desc limit 10;"
echo "--- total rows per month ---"
docker exec tradeai-postgres psql -U tradeai -d tradeai -t -A -F"|" -c \
  "select substr(date,1,7) ym, count(*) from option_candles where date >= '2025-12-01' group by ym order by ym;"
