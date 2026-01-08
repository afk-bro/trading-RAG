-- Migration: 009_backtest_tunes
-- Add tables for parameter tuning sessions and trial runs

-- Main tuning session table
CREATE TABLE IF NOT EXISTS backtest_tunes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Workspace scoping
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Strategy linkage
    strategy_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,
    strategy_spec_id UUID REFERENCES kb_strategy_specs(id) ON DELETE SET NULL,

    -- Search configuration
    search_type TEXT NOT NULL CHECK (search_type IN ('grid', 'random')),
    n_trials INTEGER NOT NULL CHECK (n_trials > 0 AND n_trials <= 200),
    seed INTEGER,
    param_space JSONB NOT NULL,

    -- Objective configuration
    objective_metric TEXT NOT NULL DEFAULT 'sharpe',
    min_trades INTEGER NOT NULL DEFAULT 5,

    -- Execution state
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    trials_completed INTEGER DEFAULT 0,

    -- Results (cached, derived from tune_runs)
    best_run_id UUID REFERENCES backtest_runs(id) ON DELETE SET NULL,
    leaderboard JSONB,

    -- Timing & errors
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT
);

-- Trial runs join table
CREATE TABLE IF NOT EXISTS backtest_tune_runs (
    tune_id UUID NOT NULL REFERENCES backtest_tunes(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    trial_index INTEGER NOT NULL,
    params JSONB NOT NULL,
    score DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'skipped')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (tune_id, run_id),
    UNIQUE (tune_id, trial_index)
);

-- Indexes for backtest_tunes
CREATE INDEX IF NOT EXISTS idx_tunes_workspace ON backtest_tunes(workspace_id);
CREATE INDEX IF NOT EXISTS idx_tunes_strategy ON backtest_tunes(strategy_entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tunes_status ON backtest_tunes(status);

-- Indexes for backtest_tune_runs
CREATE INDEX IF NOT EXISTS idx_tune_runs_tune_id ON backtest_tune_runs(tune_id);
CREATE INDEX IF NOT EXISTS idx_tune_runs_score ON backtest_tune_runs(tune_id, score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_tune_runs_status ON backtest_tune_runs(tune_id, status);
