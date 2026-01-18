-- Migration: 057_strategy_scripts
-- Purpose: Track Pine script discovery state and auto-discovery metadata
-- Links source files to strategy registry with lifecycle workflow

-- Discovery status enum
CREATE TYPE script_discovery_status AS ENUM (
    'discovered',      -- Found and cataloged
    'spec_generated',  -- StrategySpec created
    'published',       -- Linked to strategies table
    'archived'         -- No longer tracked
);

CREATE TABLE strategy_scripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Source identity (canonical_url derived from source_type + rel_path)
    rel_path TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'local',  -- 'local' | 'git' (future)

    -- Content tracking
    sha256 TEXT NOT NULL,
    pine_version TEXT,
    script_type TEXT,  -- 'strategy' | 'indicator' | 'library'
    title TEXT,

    -- Discovery state
    status script_discovery_status NOT NULL DEFAULT 'discovered',

    -- Spec generation (strategies only)
    spec_json JSONB,
    spec_generated_at TIMESTAMPTZ,

    -- Lint info (for debugging / UI)
    lint_json JSONB,

    -- Link to strategy registry (when published)
    strategy_id UUID REFERENCES strategies(id) ON DELETE SET NULL,
    published_at TIMESTAMPTZ,

    -- Timestamps
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint on (workspace, source_type, rel_path)
    CONSTRAINT uq_strategy_scripts_workspace_path
        UNIQUE (workspace_id, source_type, rel_path)
);

-- Updated_at trigger (reuse existing function from 001_initial_schema)
CREATE TRIGGER set_strategy_scripts_updated_at
    BEFORE UPDATE ON strategy_scripts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Primary index: active scripts by last seen (for list views)
CREATE INDEX idx_strategy_scripts_active_seen
    ON strategy_scripts(workspace_id, last_seen_at DESC)
    WHERE status != 'archived';

-- Status filtering
CREATE INDEX idx_strategy_scripts_workspace_status
    ON strategy_scripts(workspace_id, status);

-- Fingerprint lookup for reconciliation
CREATE INDEX idx_strategy_scripts_sha256
    ON strategy_scripts(sha256);

-- Strategy registry linkage
CREATE INDEX idx_strategy_scripts_strategy_id
    ON strategy_scripts(strategy_id)
    WHERE strategy_id IS NOT NULL;

-- Table and column comments
COMMENT ON TABLE strategy_scripts IS 'Discovery state for Pine scripts. Links source files to strategy registry.';
COMMENT ON COLUMN strategy_scripts.rel_path IS 'Relative path from scan root, POSIX-normalized (no leading /, no .., no ./)';
COMMENT ON COLUMN strategy_scripts.source_type IS 'Source type: local (filesystem) or git (future)';
COMMENT ON COLUMN strategy_scripts.sha256 IS 'SHA256 hash of script content for change detection';
COMMENT ON COLUMN strategy_scripts.status IS 'Lifecycle: discovered → spec_generated → published → archived';
COMMENT ON COLUMN strategy_scripts.spec_json IS 'Generated StrategySpec JSON (strategies only)';
COMMENT ON COLUMN strategy_scripts.lint_json IS 'Lint findings for debugging and UI display';
COMMENT ON COLUMN strategy_scripts.last_seen_at IS 'Last time script was seen during discovery scan';
