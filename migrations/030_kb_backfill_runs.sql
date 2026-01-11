-- Migration 030: KB Backfill Run Tracking
-- Tracks backfill operations for idempotency and resume semantics.
--
-- Enables:
--   - Safe reruns (no duplicate writes)
--   - Resume from last processed cursor on failure
--   - Audit trail for all backfill operations
--   - Admin visibility into backfill coverage

CREATE TABLE IF NOT EXISTS kb_backfill_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Backfill type: 'candidacy' | 'regime'
    backfill_type TEXT NOT NULL CHECK (backfill_type IN ('candidacy', 'regime')),

    -- Run status: 'running' | 'completed' | 'failed'
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),

    -- Timestamps
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Progress counters
    processed_count INT NOT NULL DEFAULT 0,
    skipped_count INT NOT NULL DEFAULT 0,
    error_count INT NOT NULL DEFAULT 0,

    -- Resume support: generic cursor (entity ID, timestamp, etc.)
    last_processed_cursor TEXT,

    -- Configuration snapshot for reproducibility
    -- e.g., {"since": "2025-01-01", "limit": 5000, "experiment_type": "sweep"}
    config JSONB NOT NULL DEFAULT '{}',

    -- Error message if status = 'failed'
    error TEXT,

    -- Dry-run flag (for audit without mutations)
    dry_run BOOLEAN NOT NULL DEFAULT FALSE
);

-- Index for finding resumable runs
CREATE INDEX IF NOT EXISTS idx_kb_backfill_runs_resume
    ON kb_backfill_runs(workspace_id, backfill_type, status)
    WHERE status IN ('running', 'failed');

-- Index for workspace history queries
CREATE INDEX IF NOT EXISTS idx_kb_backfill_runs_workspace_history
    ON kb_backfill_runs(workspace_id, started_at DESC);

-- Index for admin visibility (recent runs across all workspaces)
CREATE INDEX IF NOT EXISTS idx_kb_backfill_runs_recent
    ON kb_backfill_runs(started_at DESC);

COMMENT ON TABLE kb_backfill_runs IS 'Tracks KB backfill operations for idempotency and resume';
COMMENT ON COLUMN kb_backfill_runs.last_processed_cursor IS 'Generic cursor for resume (entity ID, timestamp, etc.)';
COMMENT ON COLUMN kb_backfill_runs.config IS 'Configuration snapshot: since, limit, experiment_type, etc.';
COMMENT ON COLUMN kb_backfill_runs.dry_run IS 'True if run was preview-only (no KB mutations)';
