-- Migration: Add ema200 column to market_snapshots table
-- Run on the server: docker compose exec postgres psql -U postgres -d tradeai -f /dev/stdin < scripts/migrate_ema200.sql

ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS ema200 FLOAT;
