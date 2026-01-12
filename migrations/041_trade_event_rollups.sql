-- migrations/041_trade_event_rollups.sql
-- Daily rollup table for historical analytics

CREATE TABLE IF NOT EXISTS trade_event_rollups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID REFERENCES kb_entities(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    rollup_date DATE NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    sample_correlation_ids TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_rollup_key UNIQUE (workspace_id, strategy_entity_id, event_type, rollup_date)
);

CREATE INDEX IF NOT EXISTS idx_rollups_workspace_date
    ON trade_event_rollups(workspace_id, rollup_date DESC);

CREATE INDEX IF NOT EXISTS idx_rollups_strategy
    ON trade_event_rollups(strategy_entity_id, rollup_date DESC)
    WHERE strategy_entity_id IS NOT NULL;
