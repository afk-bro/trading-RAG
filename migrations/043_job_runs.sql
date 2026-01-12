-- migrations/043_job_runs.sql
-- Job runs tracking table for retention/rollup jobs

CREATE TABLE job_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name TEXT NOT NULL,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_ms INTEGER,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    metrics JSONB NOT NULL DEFAULT '{}',
    error TEXT,
    triggered_by TEXT,
    correlation_id UUID,

    CONSTRAINT job_runs_status_check
        CHECK (status IN ('running', 'completed', 'failed', 'skipped'))
);

-- Admin UI: last N runs per job (global view)
CREATE INDEX idx_job_runs_job_started
    ON job_runs(job_name, started_at DESC);

-- Admin UI: filter by workspace
CREATE INDEX idx_job_runs_workspace_job_started
    ON job_runs(workspace_id, job_name, started_at DESC);

-- Stale detection
CREATE INDEX idx_job_runs_running
    ON job_runs(status, started_at) WHERE status = 'running';
