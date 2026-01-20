-- Strategy Lifecycle v1.5: Intelligence snapshots for regime + confidence tracking
-- Migration: 080_strategy_intel_snapshots.sql

-- =============================================================================
-- strategy_intel_snapshots: Append-only time series of computed intel
-- =============================================================================

CREATE TABLE strategy_intel_snapshots (
    -- Identity + scoping
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_version_id UUID NOT NULL REFERENCES strategy_versions(id) ON DELETE CASCADE,

    -- Time axes
    as_of_ts TIMESTAMPTZ NOT NULL,      -- Market time the intel refers to
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When we computed it

    -- Core intel
    regime TEXT NOT NULL,               -- Start as TEXT; enum later once taxonomy stabilizes
    confidence_score DOUBLE PRECISION NOT NULL,
    confidence_components JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Optional payloads (future extensibility)
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    explain JSONB NOT NULL DEFAULT '{}'::jsonb,  -- Human-friendly breakdown

    -- Provenance
    engine_version TEXT,                -- Version of computation engine
    inputs_hash CHAR(64),               -- SHA256 of upstream inputs (for dedupe if needed)
    run_id UUID,                        -- Optional link to job/run that produced it

    -- Constraints
    CONSTRAINT confidence_score_range CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- =============================================================================
-- Indexes for efficient querying
-- =============================================================================

-- Timeline for a version (newest first) - primary query pattern
CREATE INDEX idx_intel_snapshots_version_time
    ON strategy_intel_snapshots(strategy_version_id, as_of_ts DESC);

-- Workspace-level scanning (for dashboards, alerts)
CREATE INDEX idx_intel_snapshots_workspace_time
    ON strategy_intel_snapshots(workspace_id, as_of_ts DESC);

-- Computed time (for "what was just calculated" queries)
CREATE INDEX idx_intel_snapshots_computed
    ON strategy_intel_snapshots(computed_at DESC);

-- Optional: dedupe by inputs_hash if recompute spam becomes an issue
-- CREATE UNIQUE INDEX ux_intel_snapshots_version_asof
--     ON strategy_intel_snapshots(strategy_version_id, as_of_ts);
-- Decision: Allow duplicates for now; layer dedupe later if needed

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE strategy_intel_snapshots IS 'Append-only time series of regime + confidence intel per strategy version';
COMMENT ON COLUMN strategy_intel_snapshots.as_of_ts IS 'Market time this intel refers to (e.g., bar close time)';
COMMENT ON COLUMN strategy_intel_snapshots.computed_at IS 'Wall clock time when intel was computed';
COMMENT ON COLUMN strategy_intel_snapshots.regime IS 'Current regime classification (TEXT for flexibility; enum later)';
COMMENT ON COLUMN strategy_intel_snapshots.confidence_score IS 'Aggregated confidence [0,1] for strategy in current conditions';
COMMENT ON COLUMN strategy_intel_snapshots.confidence_components IS 'Breakdown of confidence factors (e.g., {"regime_fit": 0.8, "backtest_oos": 0.9})';
COMMENT ON COLUMN strategy_intel_snapshots.features IS 'Optional: raw feature values used for computation';
COMMENT ON COLUMN strategy_intel_snapshots.explain IS 'Optional: human-readable explanation of confidence';
COMMENT ON COLUMN strategy_intel_snapshots.engine_version IS 'Version of the computation engine that produced this snapshot';
COMMENT ON COLUMN strategy_intel_snapshots.inputs_hash IS 'SHA256 of inputs for deduplication (optional)';
COMMENT ON COLUMN strategy_intel_snapshots.run_id IS 'Optional link to job/workflow run that produced this snapshot';
