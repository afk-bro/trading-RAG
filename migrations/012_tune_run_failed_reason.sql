-- Migration: 012_tune_run_failed_reason
-- Add failed_reason column for timeout/error tracking and timing columns

-- Add failed_reason to track why a trial failed
ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS failed_reason TEXT;

-- Add timing columns for run duration tracking
ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;

-- Add index for finding failed trials by reason
CREATE INDEX IF NOT EXISTS idx_tune_runs_failed_reason
ON backtest_tune_runs(tune_id, failed_reason)
WHERE failed_reason IS NOT NULL;

COMMENT ON COLUMN backtest_tune_runs.failed_reason IS 'Reason for failure: timeout, error, canceled, etc.';
COMMENT ON COLUMN backtest_tune_runs.started_at IS 'When trial execution started';
COMMENT ON COLUMN backtest_tune_runs.finished_at IS 'When trial execution completed';
