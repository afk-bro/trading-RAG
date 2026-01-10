-- Migration: 022_run_plans
-- Create run_plans table for orchestration grouping

CREATE TABLE IF NOT EXISTS run_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    objective_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    n_variants INT NOT NULL DEFAULT 0,
    n_completed INT NOT NULL DEFAULT 0,
    n_failed INT NOT NULL DEFAULT 0,
    n_skipped INT NOT NULL DEFAULT 0,
    best_backtest_run_id UUID NULL,
    best_objective_score DOUBLE PRECISION NULL,
    error_summary TEXT NULL,
    plan JSONB NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_run_plans_workspace
    ON run_plans(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_plans_status
    ON run_plans(status, created_at DESC);

COMMENT ON TABLE run_plans IS 'Grouping container for test/tune orchestration runs';
COMMENT ON COLUMN run_plans.plan IS 'Immutable JSON: inputs, resolved variants, provenance';
