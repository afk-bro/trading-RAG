-- Strategy Lifecycle v0.5: state transition audit log
-- Migration: 077_strategy_version_transitions.sql

CREATE TABLE strategy_version_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES strategy_versions(id) ON DELETE CASCADE,
    from_state strategy_version_state,  -- NULL for initial creation
    to_state strategy_version_state NOT NULL,
    triggered_by TEXT NOT NULL,  -- Format: "admin:<token_name>", "system", "api"
    triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason TEXT  -- Optional: "Pausing for regime shift", "Emergency stop", etc.
);

-- Query transitions by version
CREATE INDEX idx_version_transitions_version
    ON strategy_version_transitions(version_id, triggered_at DESC);

-- Find recent transitions across all versions (audit dashboard)
CREATE INDEX idx_version_transitions_recent
    ON strategy_version_transitions(triggered_at DESC);

-- Comments
COMMENT ON TABLE strategy_version_transitions IS 'Audit log of all state changes for strategy versions';
COMMENT ON COLUMN strategy_version_transitions.from_state IS 'Previous state (NULL for initial draft creation)';
COMMENT ON COLUMN strategy_version_transitions.to_state IS 'New state after transition';
COMMENT ON COLUMN strategy_version_transitions.triggered_by IS 'Actor: admin:<token_name>, system, api';
COMMENT ON COLUMN strategy_version_transitions.reason IS 'Optional explanation for state change';
