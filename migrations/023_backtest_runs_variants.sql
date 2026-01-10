-- Migration: 023_backtest_runs_variants
-- Add variant tracking columns to backtest_runs

-- New columns for run plan linkage
ALTER TABLE backtest_runs
ADD COLUMN IF NOT EXISTS run_plan_id UUID NULL REFERENCES run_plans(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS variant_index INT NULL,
ADD COLUMN IF NOT EXISTS variant_fingerprint TEXT NULL,
ADD COLUMN IF NOT EXISTS run_kind TEXT NOT NULL DEFAULT 'backtest'
    CHECK (run_kind IN ('backtest', 'tune_variant', 'test_variant')),
ADD COLUMN IF NOT EXISTS objective_score DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS skip_reason TEXT NULL;

-- Artifact metadata escape hatches
ALTER TABLE backtest_runs
ADD COLUMN IF NOT EXISTS has_equity_curve BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS has_trades BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS equity_points INT NULL,
ADD COLUMN IF NOT EXISTS trade_count INT NULL,
ADD COLUMN IF NOT EXISTS artifacts_ref JSONB NULL;

-- Update status check to include 'skipped'
-- First drop existing constraint, then add new one
ALTER TABLE backtest_runs DROP CONSTRAINT IF EXISTS backtest_runs_status_check;
ALTER TABLE backtest_runs ADD CONSTRAINT backtest_runs_status_check
    CHECK (status IN ('running', 'completed', 'failed', 'skipped'));

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_backtest_runs_plan_variant
    ON backtest_runs(run_plan_id, variant_index);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_plan_score
    ON backtest_runs(run_plan_id, objective_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_kind_created
    ON backtest_runs(run_kind, created_at DESC);

COMMENT ON COLUMN backtest_runs.run_plan_id IS 'Links to parent run_plan (NULL for standalone backtests)';
COMMENT ON COLUMN backtest_runs.variant_index IS '0..N-1 ordering within a run plan';
COMMENT ON COLUMN backtest_runs.variant_fingerprint IS 'hash(canonical(params)) for verification';
COMMENT ON COLUMN backtest_runs.run_kind IS 'Distinguishes standalone vs plan variants';
COMMENT ON COLUMN backtest_runs.objective_score IS 'Extracted for fast ORDER BY queries';
COMMENT ON COLUMN backtest_runs.artifacts_ref IS 'Future S3 refs: {"equity_curve": "s3://..."}';
