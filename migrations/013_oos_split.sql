-- Migration: 013_oos_split
-- Add in-sample/out-of-sample split support for parameter tuning

-- Add OOS configuration to tune requests
ALTER TABLE backtest_tunes
ADD COLUMN IF NOT EXISTS oos_ratio DOUBLE PRECISION;

-- Constraint: oos_ratio must be in (0, 0.5] when set
ALTER TABLE backtest_tunes
ADD CONSTRAINT chk_oos_ratio_range
CHECK (oos_ratio IS NULL OR (oos_ratio > 0 AND oos_ratio <= 0.5));

-- Add IS/OOS scores to tune runs
ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS score_is DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS score_oos DOUBLE PRECISION;

COMMENT ON COLUMN backtest_tunes.oos_ratio IS 'OOS split ratio (0-0.5). When set, score=score_oos and winner chosen by OOS.';
COMMENT ON COLUMN backtest_tune_runs.score_is IS 'In-sample score (fitting window)';
COMMENT ON COLUMN backtest_tune_runs.score_oos IS 'Out-of-sample score (validation window)';
