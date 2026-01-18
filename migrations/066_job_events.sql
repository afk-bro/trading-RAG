-- migrations/066_job_events.sql
-- Job events table for structured logging during job execution

CREATE TABLE IF NOT EXISTS job_events (
    id         BIGSERIAL PRIMARY KEY,
    job_id     UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts         TIMESTAMPTZ DEFAULT now(),
    level      TEXT NOT NULL CHECK (level IN ('info', 'warn', 'error')),
    message    TEXT NOT NULL,
    meta       JSONB
);

CREATE INDEX IF NOT EXISTS idx_job_events_job
    ON job_events (job_id, ts);

CREATE INDEX IF NOT EXISTS idx_job_events_level
    ON job_events (level, ts DESC) WHERE level IN ('warn', 'error');

COMMENT ON TABLE job_events IS 'Structured log events during job execution';
COMMENT ON COLUMN job_events.level IS 'Log level: info, warn, error';
COMMENT ON COLUMN job_events.meta IS 'Additional structured data for the event';
