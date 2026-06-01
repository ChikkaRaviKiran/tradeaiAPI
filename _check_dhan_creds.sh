#!/bin/bash
echo "\d broker_credentials" | docker exec -i tradeai-postgres psql -U tradeai -d tradeai
echo "---"
echo "SELECT * FROM broker_credentials LIMIT 3;" | docker exec -i tradeai-postgres psql -U tradeai -d tradeai
