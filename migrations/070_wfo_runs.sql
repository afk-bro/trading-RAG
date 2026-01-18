-- Migration 070: WFO runs table
-- Walk-Forward Optimization sessions tracking

-- WFO status enum values: pending, running, completed, partial, failed, canceled

CREATE TABLE IF NOT EXISTS wfo_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID NOT NULL,

    -- WFO configuration
    wfo_config JSONB NOT NULL,  -- train_days, test_days, step_days, min_folds, etc.
    param_space JSONB,          -- Parameter search space
    data_source JSONB,          -- exchange_id, symbol, timeframe

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending',
    n_folds INT NOT NULL DEFAULT 0,
    folds_completed INT NOT NULL DEFAULT 0,
    folds_failed INT NOT NULL DEFAULT 0,

    -- Results (populated on completion)
    best_params JSONB,
    best_candidate JSONB,       -- Full WFOCandidateMetrics of winner
    candidates JSONB,           -- Top eligible candidates
    child_tune_ids UUID[] DEFAULT '{}',

    -- Job linkage
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,

    -- Audit
    warnings JSONB DEFAULT '[]',
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_wfo_runs_workspace ON wfo_runs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_wfo_runs_strategy ON wfo_runs(strategy_entity_id);
CREATE INDEX IF NOT EXISTS idx_wfo_runs_status ON wfo_runs(status);
CREATE INDEX IF NOT EXISTS idx_wfo_runs_job ON wfo_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_wfo_runs_created ON wfo_runs(created_at DESC);

-- Composite index for common list query
CREATE INDEX IF NOT EXISTS idx_wfo_runs_workspace_status
    ON wfo_runs(workspace_id, status, created_at DESC);

COMMENT ON TABLE wfo_runs IS 'Walk-Forward Optimization sessions';
COMMENT ON COLUMN wfo_runs.wfo_config IS 'WFO configuration: train_days, test_days, step_days, min_folds, leaderboard_top_k, allow_partial';
COMMENT ON COLUMN wfo_runs.best_candidate IS 'Full WFOCandidateMetrics for the winning parameter set';
COMMENT ON COLUMN wfo_runs.candidates IS 'Top eligible candidates meeting coverage threshold, sorted by selection score';
