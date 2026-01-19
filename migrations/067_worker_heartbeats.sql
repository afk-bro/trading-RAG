-- migrations/067_worker_heartbeats.sql
-- Worker heartbeat tracking for health monitoring

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id   TEXT PRIMARY KEY,
    version     TEXT,
    last_seen   TIMESTAMPTZ DEFAULT now(),
    started_at  TIMESTAMPTZ DEFAULT now(),
    jobs_completed INT DEFAULT 0,
    jobs_failed    INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_worker_heartbeats_last_seen
    ON worker_heartbeats (last_seen DESC);

COMMENT ON TABLE worker_heartbeats IS 'Worker process heartbeats for health monitoring';
COMMENT ON COLUMN worker_heartbeats.worker_id IS 'Unique worker identifier (hostname:pid)';
COMMENT ON COLUMN worker_heartbeats.version IS 'Application version running on worker';
