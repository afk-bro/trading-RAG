-- Migration 015: Add composite objective configuration
-- Allows configuring objective function type and parameters per tune

-- Add objective configuration to tunes
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS objective_type VARCHAR(50) DEFAULT 'sharpe',
  ADD COLUMN IF NOT EXISTS objective_params JSONB;

-- Add computed objective_score to tune_runs (for persistence/ordering)
ALTER TABLE backtest_tune_runs
  ADD COLUMN IF NOT EXISTS objective_score DOUBLE PRECISION;

COMMENT ON COLUMN backtest_tunes.objective_type IS 'Objective function type: sharpe, sharpe_dd_penalty, return, calmar';
COMMENT ON COLUMN backtest_tunes.objective_params IS 'Objective function parameters, e.g., {"dd_lambda": 0.02}';
COMMENT ON COLUMN backtest_tune_runs.objective_score IS 'Computed objective score (may differ from raw metric score when using composite objectives)';
