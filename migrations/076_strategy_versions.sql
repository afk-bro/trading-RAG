-- Strategy Lifecycle v0.5: strategy_versions table with state machine
-- Migration: 076_strategy_versions.sql

-- =============================================================================
-- Part A: Add strategy_entity_id to strategies table (if missing)
-- =============================================================================

-- Ensure strategies table has mapping to legacy kb_entities
-- ON DELETE RESTRICT: prevent accidental deletion of kb_entities that have strategies
ALTER TABLE strategies
ADD COLUMN IF NOT EXISTS strategy_entity_id UUID REFERENCES kb_entities(id) ON DELETE RESTRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategies_workspace_entity
    ON strategies(workspace_id, strategy_entity_id) WHERE strategy_entity_id IS NOT NULL;

COMMENT ON COLUMN strategies.strategy_entity_id IS 'Link to kb_entities for legacy backtest/tune/WFO FK compatibility';

-- =============================================================================
-- Part B: Create strategy_versions table
-- =============================================================================

-- State enum for version lifecycle
CREATE TYPE strategy_version_state AS ENUM ('draft', 'active', 'paused', 'retired');

CREATE TABLE strategy_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,

    -- Legacy compatibility: denormalized join key for existing backtest/tune/WFO tables
    -- ON DELETE RESTRICT: preserve audit trail, don't cascade-delete versions
    strategy_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE RESTRICT,

    -- Version identity
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    version_tag TEXT,  -- Optional: "v1.0-beta"

    -- Immutable config snapshot (enforced by trigger)
    config_snapshot JSONB NOT NULL,
    config_hash CHAR(64) NOT NULL,  -- SHA256 hex, always 64 chars

    -- State machine
    state strategy_version_state NOT NULL DEFAULT 'draft',

    -- Live Intelligence hook (v1.5)
    regime_awareness JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Audit timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT,  -- Format: "admin:<token_name>" or "system"

    activated_at TIMESTAMPTZ,
    paused_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,

    -- Source linkage (optional)
    kb_strategy_spec_id UUID REFERENCES kb_strategy_specs(id) ON DELETE SET NULL,

    -- Constraints
    CONSTRAINT version_tag_not_empty CHECK (version_tag IS NULL OR length(trim(version_tag)) > 0)
);

-- NOTE: workspace_id omitted - derive via JOIN on strategy_id if needed
-- This avoids redundant data that could get out of sync

-- CRITICAL: One active version per strategy (partial unique index)
CREATE UNIQUE INDEX idx_strategy_versions_one_active
    ON strategy_versions(strategy_id) WHERE state = 'active';

-- Unique version number per strategy
CREATE UNIQUE INDEX idx_strategy_versions_number
    ON strategy_versions(strategy_id, version_number);

-- No duplicate configs per strategy (dedup by hash)
CREATE UNIQUE INDEX idx_strategy_versions_config_hash
    ON strategy_versions(strategy_id, config_hash);

-- List versions by strategy (newest first)
CREATE INDEX idx_strategy_versions_strategy
    ON strategy_versions(strategy_id, created_at DESC);

-- Comments
COMMENT ON TABLE strategy_versions IS 'Immutable strategy config versions with state machine lifecycle';
COMMENT ON COLUMN strategy_versions.strategy_entity_id IS 'Denormalized FK for legacy backtest/tune/WFO joins';
COMMENT ON COLUMN strategy_versions.config_snapshot IS 'Immutable strategy config captured at version creation';
COMMENT ON COLUMN strategy_versions.config_hash IS 'SHA256 hex of config_snapshot for deduplication';
COMMENT ON COLUMN strategy_versions.regime_awareness IS 'Live Intelligence: regime-specific behavior config (v1.5)';
COMMENT ON COLUMN strategy_versions.created_by IS 'Actor who created version: admin:<token_name> or system';

-- =============================================================================
-- Part C: Triggers for entity_id validation and config immutability
-- =============================================================================

-- Enforce: strategy_entity_id matches parent AND parent has mapping set
CREATE OR REPLACE FUNCTION enforce_version_insert_constraints()
RETURNS TRIGGER AS $$
DECLARE
    parent_entity_id UUID;
BEGIN
    SELECT strategy_entity_id INTO parent_entity_id
    FROM strategies
    WHERE id = NEW.strategy_id
    FOR SHARE;  -- Lock to prevent race conditions

    IF parent_entity_id IS NULL THEN
        RAISE EXCEPTION 'Cannot create version: parent strategy % has no strategy_entity_id mapping', NEW.strategy_id;
    END IF;

    IF NEW.strategy_entity_id != parent_entity_id THEN
        RAISE EXCEPTION 'strategy_entity_id mismatch: version has %, parent strategy % has %', NEW.strategy_entity_id, NEW.strategy_id, parent_entity_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_version_insert_constraints
    BEFORE INSERT ON strategy_versions
    FOR EACH ROW
    EXECUTE FUNCTION enforce_version_insert_constraints();

-- Enforce: immutable columns cannot be changed after creation
CREATE OR REPLACE FUNCTION enforce_version_immutability()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.strategy_id != OLD.strategy_id THEN
        RAISE EXCEPTION 'Cannot change strategy_id (immutable)';
    END IF;
    IF NEW.strategy_entity_id != OLD.strategy_entity_id THEN
        RAISE EXCEPTION 'Cannot change strategy_entity_id (immutable)';
    END IF;
    IF NEW.version_number != OLD.version_number THEN
        RAISE EXCEPTION 'Cannot change version_number (immutable)';
    END IF;
    IF NEW.config_hash != OLD.config_hash THEN
        RAISE EXCEPTION 'Cannot change config_hash (immutable)';
    END IF;
    IF NEW.config_snapshot != OLD.config_snapshot THEN
        RAISE EXCEPTION 'Cannot change config_snapshot (immutable)';
    END IF;
    IF NEW.created_at != OLD.created_at THEN
        RAISE EXCEPTION 'Cannot change created_at (immutable)';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_version_immutability
    BEFORE UPDATE OF strategy_id, strategy_entity_id, version_number, config_hash, config_snapshot, created_at
    ON strategy_versions
    FOR EACH ROW
    EXECUTE FUNCTION enforce_version_immutability();
