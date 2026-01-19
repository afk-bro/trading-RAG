-- migrations/071_ops_alerts.sql
-- Operational alerts for system health, coverage gaps, drift detection
-- Separate from strategy-focused alert_events (migration 045)

CREATE TABLE IF NOT EXISTS ops_alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID NOT NULL REFERENCES workspaces(id),

    -- Classification
    rule_type       TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'medium'
                    CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'resolved')),
    rule_version    TEXT NOT NULL DEFAULT 'v1',

    -- Deduplication (unique per workspace)
    dedupe_key      TEXT NOT NULL,

    -- Context
    payload         JSONB NOT NULL DEFAULT '{}',
    source          TEXT NOT NULL DEFAULT 'alert_evaluator',
    job_run_id      UUID,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,

    -- Acknowledgment (metadata, not status)
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,

    UNIQUE(workspace_id, dedupe_key)
);

-- Primary query: recent alerts for workspace
CREATE INDEX IF NOT EXISTS idx_ops_alerts_workspace_recent
    ON ops_alerts(workspace_id, created_at DESC);

-- Active alerts (ordered by last_seen for "still happening")
CREATE INDEX IF NOT EXISTS idx_ops_alerts_active
    ON ops_alerts(workspace_id, last_seen_at DESC) WHERE status = 'active';

-- Severity filtering
CREATE INDEX IF NOT EXISTS idx_ops_alerts_severity
    ON ops_alerts(workspace_id, severity, created_at DESC);

-- Job run linkage
CREATE INDEX IF NOT EXISTS idx_ops_alerts_job_run
    ON ops_alerts(job_run_id) WHERE job_run_id IS NOT NULL;

-- Rule type filtering
CREATE INDEX IF NOT EXISTS idx_ops_alerts_rule_type
    ON ops_alerts(workspace_id, rule_type, created_at DESC);

COMMENT ON TABLE ops_alerts IS 'Operational alerts for health/coverage/drift with deduplication';
COMMENT ON COLUMN ops_alerts.dedupe_key IS 'Unique per workspace: {rule_type}:{bucket_key}';
COMMENT ON COLUMN ops_alerts.last_seen_at IS 'Updated each time condition is still true (heartbeat)';
COMMENT ON COLUMN ops_alerts.rule_version IS 'Version of rule logic that created this event';
