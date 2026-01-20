-- Strategy Lifecycle v0.5: denormalized active version pointer
-- Migration: 079_strategies_active_version.sql

-- Add active_version_id to strategies for fast lookup
-- ON DELETE SET NULL: if version is deleted, clear the pointer
ALTER TABLE strategies
ADD COLUMN IF NOT EXISTS active_version_id UUID REFERENCES strategy_versions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_strategies_active_version
    ON strategies(active_version_id) WHERE active_version_id IS NOT NULL;

COMMENT ON COLUMN strategies.active_version_id IS 'Denormalized pointer to currently active version (if any)';

-- Trigger to enforce active_version_id points to a version of THIS strategy in 'active' state
CREATE OR REPLACE FUNCTION enforce_active_version_consistency()
RETURNS TRIGGER AS $$
DECLARE
    version_strategy_id UUID;
    version_state strategy_version_state;
BEGIN
    IF NEW.active_version_id IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT strategy_id, state INTO version_strategy_id, version_state
    FROM strategy_versions
    WHERE id = NEW.active_version_id
    FOR SHARE;  -- Lock to prevent race with concurrent activation

    IF version_strategy_id IS NULL THEN
        RAISE EXCEPTION 'active_version_id % references non-existent version', NEW.active_version_id;
    END IF;

    IF version_strategy_id != NEW.id THEN
        RAISE EXCEPTION 'active_version_id % belongs to strategy %, not %', NEW.active_version_id, version_strategy_id, NEW.id;
    END IF;

    IF version_state != 'active' THEN
        RAISE EXCEPTION 'active_version_id % has state %, expected active', NEW.active_version_id, version_state;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_active_version_consistency
    BEFORE INSERT OR UPDATE OF active_version_id ON strategies
    FOR EACH ROW
    EXECUTE FUNCTION enforce_active_version_consistency();
