#!/bin/bash
for i in 1 2 3 4 5 6 7 8; do
  age=$(sudo docker ps --format '{{.Names}} {{.Status}}' | grep tradeai-backend)
  echo "[try $i] $age"
  if echo "$age" | grep -qE 'Up (Less than a|[1-9]) (second|minute)'; then
    echo 'NEW CONTAINER'
    break
  fi
  sleep 25
done
echo ---
sudo docker exec tradeai-backend grep -c 'BFO and expiry' /app/app/core/instruments.py
