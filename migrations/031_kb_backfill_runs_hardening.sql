-- Migration 031: KB Backfill Runs Hardening
-- Adds config_hash for stable matching and concurrency guard.
--
-- Fixes:
--   1. Config equality gotchas (JSON key ordering, default values)
--   2. Concurrent backfills stomping each other

-- Add config_hash column for stable matching
-- sha256 of canonical JSON (sorted keys, no whitespace)
ALTER TABLE kb_backfill_runs
ADD COLUMN IF NOT EXISTS config_hash TEXT;

-- Backfill existing rows with hash (application will compute going forward)
-- For now, leave NULL - find_resumable will fall back to config equality

-- Index for config_hash lookups (faster than JSONB equality)
CREATE INDEX IF NOT EXISTS idx_kb_backfill_runs_config_hash
    ON kb_backfill_runs(workspace_id, backfill_type, config_hash)
    WHERE config_hash IS NOT NULL;

-- Concurrency guard: only one running backfill per workspace/type/config
-- Prevents two shells from stomping each other
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_backfill_runs_one_active
    ON kb_backfill_runs(workspace_id, backfill_type, config_hash)
    WHERE status = 'running' AND config_hash IS NOT NULL;

COMMENT ON COLUMN kb_backfill_runs.config_hash IS 'SHA256 of canonical JSON config for stable matching and concurrency guard';
