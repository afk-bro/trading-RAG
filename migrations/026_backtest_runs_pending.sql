-- Migration: 026_backtest_runs_pending
-- Add 'pending' status to backtest_runs for pre-creation of variant rows

ALTER TABLE backtest_runs DROP CONSTRAINT IF EXISTS backtest_runs_status_check;
ALTER TABLE backtest_runs ADD CONSTRAINT backtest_runs_status_check
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped'));

COMMENT ON COLUMN backtest_runs.status IS 'Lifecycle: pending -> running -> completed|failed|skipped';
