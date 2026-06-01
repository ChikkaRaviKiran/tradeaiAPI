SELECT instrument, MIN(date), MAX(date), COUNT(DISTINCT date) as days, COUNT(*) as rows
FROM option_candles GROUP BY instrument ORDER BY instrument;
