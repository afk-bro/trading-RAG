-- Migration 027: KB Status Model for Trial Ingestion
-- Implements Phase 1 of the trial ingestion design:
--   - Status columns on backtest_tune_runs and backtest_runs
--   - Workspace circuit breaker state
--   - kb_status_history audit table
--   - kb_trial_index idempotency table

-- ============================================================================
-- 1. KB Status columns on backtest_tune_runs
--    Default: 'promoted' (tune runs are curated by design)
-- ============================================================================

ALTER TABLE backtest_tune_runs
    ADD COLUMN IF NOT EXISTS kb_status TEXT DEFAULT 'promoted',
    ADD COLUMN IF NOT EXISTS kb_status_changed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS kb_status_changed_by TEXT,
    ADD COLUMN IF NOT EXISTS kb_status_reason TEXT,
    ADD COLUMN IF NOT EXISTS kb_promoted_at TIMESTAMPTZ DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS kb_promoted_by TEXT;

-- Constraint for valid status values
ALTER TABLE backtest_tune_runs
    DROP CONSTRAINT IF EXISTS backtest_tune_runs_kb_status_check;
ALTER TABLE backtest_tune_runs
    ADD CONSTRAINT backtest_tune_runs_kb_status_check
    CHECK (kb_status IN ('excluded', 'candidate', 'promoted', 'rejected'));

-- ============================================================================
-- 2. KB Status + Regime columns on backtest_runs
--    Default: 'excluded' (test variants are noisy by default)
-- ============================================================================

ALTER TABLE backtest_runs
    ADD COLUMN IF NOT EXISTS kb_status TEXT DEFAULT 'excluded',
    ADD COLUMN IF NOT EXISTS kb_status_changed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS kb_status_changed_by TEXT,
    ADD COLUMN IF NOT EXISTS kb_status_reason TEXT,
    ADD COLUMN IF NOT EXISTS kb_promoted_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS kb_promoted_by TEXT,
    ADD COLUMN IF NOT EXISTS auto_candidate_gate TEXT,
    ADD COLUMN IF NOT EXISTS auto_candidate_breaker TEXT,
    -- Regime data for candidacy evaluation
    ADD COLUMN IF NOT EXISTS regime_is JSONB,
    ADD COLUMN IF NOT EXISTS regime_oos JSONB,
    ADD COLUMN IF NOT EXISTS regime_schema_version TEXT;

-- Constraint for valid status values
ALTER TABLE backtest_runs
    DROP CONSTRAINT IF EXISTS backtest_runs_kb_status_check;
ALTER TABLE backtest_runs
    ADD CONSTRAINT backtest_runs_kb_status_check
    CHECK (kb_status IN ('excluded', 'candidate', 'promoted', 'rejected'));

-- Index for querying eligible trials
CREATE INDEX IF NOT EXISTS idx_backtest_runs_kb_status
    ON backtest_runs(workspace_id, kb_status)
    WHERE kb_status IN ('candidate', 'promoted');

-- ============================================================================
-- 3. Workspace circuit breaker state columns
-- ============================================================================

ALTER TABLE workspaces
    ADD COLUMN IF NOT EXISTS kb_auto_candidacy_state TEXT DEFAULT 'enabled',
    ADD COLUMN IF NOT EXISTS kb_auto_candidacy_disabled_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS kb_auto_candidacy_trip_reason TEXT,
    ADD COLUMN IF NOT EXISTS kb_auto_candidacy_tripped_at TIMESTAMPTZ;

-- Constraint for valid breaker states
ALTER TABLE workspaces
    DROP CONSTRAINT IF EXISTS workspaces_kb_auto_candidacy_state_check;
ALTER TABLE workspaces
    ADD CONSTRAINT workspaces_kb_auto_candidacy_state_check
    CHECK (kb_auto_candidacy_state IN ('enabled', 'degraded', 'disabled'));

-- ============================================================================
-- 4. KB Status History (audit log)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_status_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_id UUID NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Index for looking up history by source
CREATE INDEX IF NOT EXISTS idx_kb_status_history_source
    ON kb_status_history(source_type, source_id);

-- Index for workspace-level audit queries (recent first)
CREATE INDEX IF NOT EXISTS idx_kb_status_history_workspace_created
    ON kb_status_history(workspace_id, created_at DESC);

-- Constraint for valid status values in history
ALTER TABLE kb_status_history
    DROP CONSTRAINT IF EXISTS kb_status_history_status_check;
ALTER TABLE kb_status_history
    ADD CONSTRAINT kb_status_history_status_check
    CHECK (
        from_status IN ('excluded', 'candidate', 'promoted', 'rejected')
        AND to_status IN ('excluded', 'candidate', 'promoted', 'rejected')
    );

-- Constraint for valid actor types
ALTER TABLE kb_status_history
    DROP CONSTRAINT IF EXISTS kb_status_history_actor_type_check;
ALTER TABLE kb_status_history
    ADD CONSTRAINT kb_status_history_actor_type_check
    CHECK (actor_type IN ('auto', 'admin'));

-- ============================================================================
-- 5. KB Trial Index (idempotency + Qdrant mapping)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_trial_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_id UUID NOT NULL,
    qdrant_point_id UUID NOT NULL,
    content_hash TEXT NOT NULL,
    content_hash_algo TEXT NOT NULL DEFAULT 'sha256_v1',
    regime_schema_version TEXT,
    embed_model TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    archived_at TIMESTAMPTZ,
    archived_reason TEXT,
    archived_by TEXT,

    -- Each source can only have one active index entry per workspace
    UNIQUE (workspace_id, source_type, source_id)
);

-- Partial index for fast lookups of non-archived entries
CREATE INDEX IF NOT EXISTS idx_kb_trial_index_lookup
    ON kb_trial_index(workspace_id, source_type, source_id)
    WHERE archived_at IS NULL;

-- Index for Qdrant point ID lookups (for deletion/updates)
CREATE INDEX IF NOT EXISTS idx_kb_trial_index_qdrant
    ON kb_trial_index(qdrant_point_id);

-- Index for collection-based queries
CREATE INDEX IF NOT EXISTS idx_kb_trial_index_collection
    ON kb_trial_index(collection_name, workspace_id)
    WHERE archived_at IS NULL;

-- Constraint for valid source types
ALTER TABLE kb_trial_index
    DROP CONSTRAINT IF EXISTS kb_trial_index_source_type_check;
ALTER TABLE kb_trial_index
    ADD CONSTRAINT kb_trial_index_source_type_check
    CHECK (source_type IN ('tune_run', 'test_variant'));

-- ============================================================================
-- Comments for documentation
-- ============================================================================

COMMENT ON COLUMN backtest_tune_runs.kb_status IS 'KB eligibility status: excluded|candidate|promoted|rejected';
COMMENT ON COLUMN backtest_tune_runs.kb_promoted_at IS 'Timestamp when explicitly promoted (for tie-breaking)';
COMMENT ON COLUMN backtest_runs.kb_status IS 'KB eligibility status: excluded|candidate|promoted|rejected';
COMMENT ON COLUMN backtest_runs.auto_candidate_gate IS 'Gate decision reason from is_candidate()';
COMMENT ON COLUMN backtest_runs.auto_candidate_breaker IS 'Circuit breaker decision if gate passed';
COMMENT ON COLUMN backtest_runs.regime_is IS 'Regime snapshot from in-sample segment';
COMMENT ON COLUMN backtest_runs.regime_oos IS 'Regime snapshot from out-of-sample segment';
COMMENT ON COLUMN workspaces.kb_auto_candidacy_state IS 'Circuit breaker state: enabled|degraded|disabled';
COMMENT ON TABLE kb_status_history IS 'Audit log for all KB status transitions';
COMMENT ON TABLE kb_trial_index IS 'Maps trial sources to Qdrant points with content hash for idempotency';
