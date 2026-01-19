-- migrations/065_jobs.sql
-- Job queue table for backtest automation

CREATE TABLE IF NOT EXISTS jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type            TEXT NOT NULL,    -- 'data_sync', 'data_fetch', 'tune', 'wfo'
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'canceled')),
    priority        INT DEFAULT 100,  -- lower = higher priority
    payload         JSONB NOT NULL,
    result          JSONB,            -- populated on completion

    attempt         INT DEFAULT 0,
    max_attempts    INT DEFAULT 3,
    run_after       TIMESTAMPTZ DEFAULT now(),

    locked_at       TIMESTAMPTZ,
    locked_by       TEXT,             -- worker_id

    parent_job_id   UUID REFERENCES jobs(id),
    workspace_id    UUID,
    dedupe_key      TEXT,             -- idempotency

    created_at      TIMESTAMPTZ DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

-- Partial unique index for idempotency
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedupe_key
    ON jobs (dedupe_key) WHERE dedupe_key IS NOT NULL;

-- Index for claim query: pending jobs ordered by priority
CREATE INDEX IF NOT EXISTS idx_jobs_claimable
    ON jobs (priority, created_at) WHERE status = 'pending';

-- Index for parent job lookups (WFO child jobs)
CREATE INDEX IF NOT EXISTS idx_jobs_parent
    ON jobs (parent_job_id) WHERE parent_job_id IS NOT NULL;

-- Index for workspace filtering
CREATE INDEX IF NOT EXISTS idx_jobs_workspace
    ON jobs (workspace_id, created_at DESC) WHERE workspace_id IS NOT NULL;

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_jobs_status
    ON jobs (status, created_at DESC);

COMMENT ON TABLE jobs IS 'Job queue for backtest automation tasks';
COMMENT ON COLUMN jobs.type IS 'Job type: data_sync, data_fetch, tune, wfo';
COMMENT ON COLUMN jobs.dedupe_key IS 'Idempotency key to prevent duplicate jobs';
COMMENT ON COLUMN jobs.run_after IS 'Earliest time job can be claimed (for delayed/retry)';
