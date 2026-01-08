-- Migration: 014_tune_runs_metrics_jsonb
-- Add metrics JSONB columns for IS/OOS summary stats

BEGIN;

ALTER TABLE backtest_tune_runs
  ADD COLUMN IF NOT EXISTS metrics_is JSONB,
  ADD COLUMN IF NOT EXISTS metrics_oos JSONB;

-- Skip GIN indexes for now to keep writes cheap.
-- Add later if querying JSON heavily.

COMMENT ON COLUMN backtest_tune_runs.metrics_is IS 'Summary metrics computed on in-sample window.';
COMMENT ON COLUMN backtest_tune_runs.metrics_oos IS 'Summary metrics computed on out-of-sample window.';

COMMIT;
