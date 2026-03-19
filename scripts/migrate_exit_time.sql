-- Add exit_time column to trades table if it doesn't exist
-- Run on the server: docker compose exec postgres psql -U postgres -d tradeai -f /dev/stdin < scripts/migrate_exit_time.sql

ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_time VARCHAR(8);
