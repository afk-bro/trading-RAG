-- Add triage workflow columns to match_runs
-- Migration: 051_match_runs_triage.sql

-- Coverage status for triage workflow
CREATE TYPE coverage_status AS ENUM ('open', 'acknowledged', 'resolved');

ALTER TABLE match_runs
    ADD COLUMN IF NOT EXISTS coverage_status coverage_status DEFAULT 'open',
    ADD COLUMN IF NOT EXISTS acknowledged_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS acknowledged_by TEXT,
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS resolved_by TEXT,
    ADD COLUMN IF NOT EXISTS resolution_note TEXT;

-- Index for filtering by status (common cockpit queries)
CREATE INDEX IF NOT EXISTS idx_match_runs_coverage_status
    ON match_runs (workspace_id, coverage_status, created_at DESC)
    WHERE weak_coverage = true;

-- Comments
COMMENT ON COLUMN match_runs.coverage_status IS 'Triage status: open, acknowledged, resolved';
COMMENT ON COLUMN match_runs.acknowledged_at IS 'When the coverage gap was acknowledged';
COMMENT ON COLUMN match_runs.acknowledged_by IS 'Who acknowledged (user/system identifier)';
COMMENT ON COLUMN match_runs.resolved_at IS 'When the coverage gap was resolved';
COMMENT ON COLUMN match_runs.resolved_by IS 'Who resolved (user/system identifier)';
COMMENT ON COLUMN match_runs.resolution_note IS 'Free-text note about resolution';
