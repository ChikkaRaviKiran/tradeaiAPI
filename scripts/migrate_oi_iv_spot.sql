-- Migration: Add oi, iv, spot columns to option_candles table
-- These fields are available from DhanHQ rollingoption API but were not previously fetched

ALTER TABLE option_candles ADD COLUMN IF NOT EXISTS oi INTEGER;
ALTER TABLE option_candles ADD COLUMN IF NOT EXISTS iv FLOAT;
ALTER TABLE option_candles ADD COLUMN IF NOT EXISTS spot FLOAT;

-- Verify
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'option_candles' ORDER BY ordinal_position;
