#!/bin/bash
TOKEN=$(docker exec tradeai-postgres psql -U tradeai -d tradeai -t -A -c "SELECT value FROM broker_credentials WHERE broker='dhan' AND key='access_token' ORDER BY id DESC LIMIT 1;")
if [ -z "$TOKEN" ]; then
  echo "ERROR: no dhan access_token in broker_credentials"
  exit 1
fi
echo "Token length: ${#TOKEN}"
echo ""
echo "=== Re-fetching last 30 days (forces re-download via --refetch) ==="
docker exec -e DHAN_ACCESS_TOKEN="$TOKEN" tradeai-backend python -m app.data.dhan_option_fetcher --refetch --batch --days 30 2>&1 | tail -40
