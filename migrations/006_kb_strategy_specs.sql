-- Migration: 006_kb_strategy_specs
-- Strategy specification snapshots derived from verified claims

CREATE TABLE IF NOT EXISTS kb_strategy_specs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_entity_id UUID NOT NULL UNIQUE REFERENCES kb_entities(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- The compiled specification
    spec_json JSONB NOT NULL DEFAULT '{}',

    -- Governance status
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'deprecated')),

    -- Traceability: which claims were used to derive this spec
    derived_from_claim_ids JSONB DEFAULT '[]',

    -- Metadata
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    approved_by TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_kb_strategy_specs_workspace ON kb_strategy_specs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_kb_strategy_specs_status ON kb_strategy_specs(status);

COMMENT ON TABLE kb_strategy_specs IS 'Persisted strategy specifications derived from verified KB claims';
