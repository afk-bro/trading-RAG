-- Truth Store Schema
-- Migration: 004_truth_store_schema
-- Stores verified knowledge objects with provenance

-- ===========================================
-- 1) kb_entities: concepts, indicators, strategies, etc.
-- ===========================================
CREATE TABLE kb_entities (
    id UUID PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Entity classification
    type TEXT NOT NULL CHECK (type IN (
        'concept', 'indicator', 'strategy', 'equation',
        'test', 'metric', 'asset', 'pattern', 'parameter', 'other'
    )),

    -- Identity
    name TEXT NOT NULL,
    aliases JSONB DEFAULT '[]'::jsonb,  -- ["RSI", "Relative Strength Index"]

    -- Grounded description (must come from evidence)
    description TEXT,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for kb_entities
CREATE INDEX idx_kb_entities_workspace ON kb_entities(workspace_id);
CREATE INDEX idx_kb_entities_type ON kb_entities(type);
CREATE INDEX idx_kb_entities_name ON kb_entities(name);
CREATE INDEX idx_kb_entities_aliases ON kb_entities USING gin(aliases);

-- Unique constraint: no duplicate entity names within a workspace+type
CREATE UNIQUE INDEX idx_kb_entities_unique_name
ON kb_entities(workspace_id, type, lower(name));

-- ===========================================
-- 2) kb_claims: atomic verified truth statements
-- ===========================================
CREATE TABLE kb_claims (
    id UUID PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Optional link to entity (claim may be standalone)
    entity_id UUID REFERENCES kb_entities(id) ON DELETE SET NULL,

    -- Claim classification
    claim_type TEXT NOT NULL CHECK (claim_type IN (
        'definition', 'rule', 'assumption', 'warning',
        'parameter', 'equation', 'observation', 'recommendation', 'other'
    )),

    -- The claim itself (grounded statement)
    text TEXT NOT NULL,

    -- Verification status
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'verified', 'weak', 'rejected'
    )),

    -- Extraction metadata
    extraction_model TEXT,  -- Which LLM extracted this
    verification_model TEXT,  -- Which LLM verified this

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for kb_claims
CREATE INDEX idx_kb_claims_workspace ON kb_claims(workspace_id);
CREATE INDEX idx_kb_claims_entity ON kb_claims(entity_id);
CREATE INDEX idx_kb_claims_type ON kb_claims(claim_type);
CREATE INDEX idx_kb_claims_status ON kb_claims(status);
CREATE INDEX idx_kb_claims_confidence ON kb_claims(confidence DESC);

-- Composite index for common query: verified claims by workspace
CREATE INDEX idx_kb_claims_verified
ON kb_claims(workspace_id, status) WHERE status = 'verified';

-- ===========================================
-- 3) kb_evidence: links claims to source chunks
-- ===========================================
CREATE TABLE kb_evidence (
    id UUID PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),

    -- Link to claim
    claim_id UUID NOT NULL REFERENCES kb_claims(id) ON DELETE CASCADE,

    -- Link to source (document + chunk)
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,

    -- The supporting quote/excerpt
    quote TEXT NOT NULL,

    -- Position within chunk (optional, for highlighting)
    start_offset INT,
    end_offset INT,

    -- Evidence strength
    relevance_score REAL DEFAULT 1.0 CHECK (relevance_score >= 0 AND relevance_score <= 1),

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for kb_evidence
CREATE INDEX idx_kb_evidence_claim ON kb_evidence(claim_id);
CREATE INDEX idx_kb_evidence_doc ON kb_evidence(doc_id);
CREATE INDEX idx_kb_evidence_chunk ON kb_evidence(chunk_id);

-- Prevent duplicate evidence for same claim+chunk
CREATE UNIQUE INDEX idx_kb_evidence_unique
ON kb_evidence(claim_id, chunk_id);

-- ===========================================
-- 4) kb_relations: connect entities
-- ===========================================
CREATE TABLE kb_relations (
    id UUID PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Relationship
    from_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN (
        'uses', 'requires', 'derived_from', 'variant_of',
        'contradicts', 'supports', 'mentions', 'component_of',
        'input_to', 'output_of', 'precedes', 'follows'
    )),
    to_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,

    -- Optional supporting claim
    claim_id UUID REFERENCES kb_claims(id) ON DELETE SET NULL,

    -- Relationship strength/confidence
    weight REAL DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for kb_relations
CREATE INDEX idx_kb_relations_workspace ON kb_relations(workspace_id);
CREATE INDEX idx_kb_relations_from ON kb_relations(from_entity_id);
CREATE INDEX idx_kb_relations_to ON kb_relations(to_entity_id);
CREATE INDEX idx_kb_relations_type ON kb_relations(relation);

-- Prevent duplicate relations
CREATE UNIQUE INDEX idx_kb_relations_unique
ON kb_relations(from_entity_id, relation, to_entity_id);

-- ===========================================
-- Updated_at triggers
-- ===========================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_kb_entities_updated_at
    BEFORE UPDATE ON kb_entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_kb_claims_updated_at
    BEFORE UPDATE ON kb_claims
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ===========================================
-- Comments for documentation
-- ===========================================
COMMENT ON TABLE kb_entities IS 'Knowledge base entities: concepts, indicators, strategies, etc.';
COMMENT ON TABLE kb_claims IS 'Atomic truth statements with verification status';
COMMENT ON TABLE kb_evidence IS 'Links claims to source chunks with supporting quotes';
COMMENT ON TABLE kb_relations IS 'Relationships between entities (uses, requires, etc.)';

COMMENT ON COLUMN kb_claims.status IS 'pending=unverified, verified=confirmed, weak=low confidence, rejected=disproven';
COMMENT ON COLUMN kb_claims.confidence IS 'Confidence score 0-1, based on evidence strength';
COMMENT ON COLUMN kb_evidence.quote IS 'Verbatim excerpt from source that supports the claim';
