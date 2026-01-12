-- Alert events (occurrences)
CREATE TABLE alert_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    rule_id UUID NOT NULL REFERENCES alert_rules(id),
    strategy_entity_id UUID NOT NULL,
    regime_key TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('drift_spike', 'confidence_drop', 'combo')),

    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved')),
    severity TEXT NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high')),

    acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,

    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,

    context_json JSONB NOT NULL DEFAULT '{}',
    fingerprint TEXT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_alert_event_key UNIQUE(workspace_id, strategy_entity_id, regime_key, timeframe, rule_type, fingerprint)
);

-- Consistency constraints
ALTER TABLE alert_events ADD CONSTRAINT chk_ack_consistency CHECK (
    (acknowledged = false AND acknowledged_at IS NULL AND acknowledged_by IS NULL)
    OR (acknowledged = true AND acknowledged_at IS NOT NULL)
);

ALTER TABLE alert_events ADD CONSTRAINT chk_resolved_consistency CHECK (
    (status = 'active' AND resolved_at IS NULL)
    OR (status = 'resolved' AND resolved_at IS NOT NULL)
);

-- Indexes
CREATE INDEX idx_alert_events_active ON alert_events(workspace_id, status) WHERE status = 'active';
CREATE INDEX idx_alert_events_list ON alert_events(workspace_id, last_seen DESC);
CREATE INDEX idx_alert_events_filtered ON alert_events(workspace_id, status, severity, last_seen DESC);
CREATE INDEX idx_alert_events_needs_attention ON alert_events(workspace_id, last_seen DESC)
    WHERE status = 'active' AND acknowledged = false;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_alert_events_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_alert_events_updated_at
    BEFORE UPDATE ON alert_events
    FOR EACH ROW
    EXECUTE FUNCTION update_alert_events_updated_at();
