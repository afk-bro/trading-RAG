-- Paper Equity Snapshots: Time series of equity state for drawdown computation
-- Migration: 081_paper_equity_snapshots.sql
--
-- NOTE: Retention policy (e.g., 90d) intentionally deferred until volume warrants it.
-- See PaperEquityRepository.delete_old_snapshots() for the cleanup method when ready.

-- =============================================================================
-- paper_equity_snapshots: Append-only time series of paper broker equity state
-- =============================================================================
--
-- Design decisions:
-- - Workspace-level equity (matches paper_broker.py's workspace-keyed state)
-- - strategy_version_id nullable for v1 (workspace-wide); can be populated later
-- - Drawdown computed at query-time, not stored (Option A in spec)
-- - inputs_hash for dedupe to prevent spam if snapshot loop runs frequently

CREATE TABLE paper_equity_snapshots (
    -- Identity + scoping
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_version_id UUID REFERENCES strategy_versions(id) ON DELETE SET NULL,

    -- Time axes
    snapshot_ts TIMESTAMPTZ NOT NULL,        -- Market time (bar close or evaluation time)
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- Wall clock when snapshot was recorded

    -- Equity components
    equity DOUBLE PRECISION NOT NULL,        -- Total equity: cash + positions_value
    cash DOUBLE PRECISION NOT NULL,          -- Available cash
    positions_value DOUBLE PRECISION NOT NULL,  -- Sum of position values
    realized_pnl DOUBLE PRECISION NOT NULL,  -- Cumulative realized P&L

    -- Dedupe (to avoid spam from frequent snapshot loops)
    inputs_hash CHAR(64),                    -- SHA256 of (workspace_id, cash, positions_value, realized_pnl)

    -- Constraints
    CONSTRAINT equity_non_negative CHECK (equity >= 0),
    CONSTRAINT cash_check CHECK (cash IS NOT NULL),  -- Explicit (not NULL + type ensures this)
    CONSTRAINT positions_value_non_negative CHECK (positions_value >= 0)
);

-- =============================================================================
-- Indexes for efficient querying
-- =============================================================================

-- Primary query: window queries for a workspace (newest first)
CREATE INDEX idx_paper_equity_workspace_time
    ON paper_equity_snapshots(workspace_id, snapshot_ts DESC);

-- Version-specific queries (for future per-version tracking)
CREATE INDEX idx_paper_equity_version_time
    ON paper_equity_snapshots(strategy_version_id, snapshot_ts DESC)
    WHERE strategy_version_id IS NOT NULL;

-- Computed time (for monitoring, cleanup)
CREATE INDEX idx_paper_equity_computed
    ON paper_equity_snapshots(computed_at DESC);

-- Dedupe lookup (prevent duplicate snapshots with same inputs)
CREATE INDEX idx_paper_equity_dedupe
    ON paper_equity_snapshots(workspace_id, inputs_hash)
    WHERE inputs_hash IS NOT NULL;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE paper_equity_snapshots IS 'Append-only time series of paper broker equity for drawdown computation';
COMMENT ON COLUMN paper_equity_snapshots.snapshot_ts IS 'Market time this snapshot refers to (e.g., bar close, trade time)';
COMMENT ON COLUMN paper_equity_snapshots.computed_at IS 'Wall clock time when snapshot was recorded';
COMMENT ON COLUMN paper_equity_snapshots.equity IS 'Total equity: cash + positions_value';
COMMENT ON COLUMN paper_equity_snapshots.cash IS 'Available cash (from paper broker state)';
COMMENT ON COLUMN paper_equity_snapshots.positions_value IS 'Market value of all open positions';
COMMENT ON COLUMN paper_equity_snapshots.realized_pnl IS 'Cumulative realized P&L from closed trades';
COMMENT ON COLUMN paper_equity_snapshots.inputs_hash IS 'SHA256 of state for deduplication';
COMMENT ON COLUMN paper_equity_snapshots.strategy_version_id IS 'Optional: version association (NULL for workspace-wide in v1)';
