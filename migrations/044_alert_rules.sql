-- Alert rules (definitions)
CREATE TABLE alert_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('drift_spike', 'confidence_drop', 'combo')),
    strategy_entity_id UUID,
    regime_key TEXT,
    timeframe TEXT,
    enabled BOOLEAN DEFAULT true,
    config JSONB NOT NULL DEFAULT '{}',
    cooldown_minutes INT DEFAULT 60,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alert_rules_workspace ON alert_rules(workspace_id, enabled);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_alert_rules_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_alert_rules_updated_at
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_alert_rules_updated_at();
