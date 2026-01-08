-- Migration: 008_backtest_runs
-- Store backtest execution results with metrics and trade history

CREATE TABLE IF NOT EXISTS backtest_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Workspace scoping
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Strategy linkage
    strategy_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,
    strategy_spec_id UUID REFERENCES kb_strategy_specs(id) ON DELETE SET NULL,
    spec_version INTEGER,

    -- Execution status
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),

    -- Run configuration
    params JSONB NOT NULL DEFAULT '{}',
    engine TEXT NOT NULL DEFAULT 'backtesting.py',

    -- Dataset metadata (don't store raw CSV)
    dataset_meta JSONB NOT NULL DEFAULT '{}',
    -- Example: {"filename": "AAPL_1h.csv", "row_count": 5000, "date_min": "2024-01-01", "date_max": "2024-12-31", "symbol": "AAPL"}

    -- Results (populated on completion)
    summary JSONB,
    -- Example: {"return_pct": 12.4, "max_drawdown_pct": -6.1, "sharpe": 1.32, "trades": 84, "win_rate": 0.58}

    equity_curve JSONB,
    -- Example: [{"t": "2024-01-01T00:00:00Z", "equity": 10000}, ...]

    trades JSONB,
    -- Example: [{"t_entry": "...", "t_exit": "...", "side": "long", "pnl": 42.1, "return_pct": 0.42}, ...]

    -- Warnings/errors
    warnings JSONB DEFAULT '[]',
    error TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_backtest_runs_workspace ON backtest_runs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_entity_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created ON backtest_runs(created_at DESC);

COMMENT ON TABLE backtest_runs IS 'Backtest execution results with metrics, equity curve, and trade history';
