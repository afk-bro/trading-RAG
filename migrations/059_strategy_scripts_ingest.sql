-- Migration: 059_strategy_scripts_ingest
-- Purpose: Add ingest tracking fields to strategy_scripts table
-- Enables tracking KB ingest status separately from discovery status

-- Add ingest tracking columns
ALTER TABLE strategy_scripts
    ADD COLUMN IF NOT EXISTS doc_id UUID,
    ADD COLUMN IF NOT EXISTS last_ingested_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_ingested_sha TEXT,
    ADD COLUMN IF NOT EXISTS ingest_status TEXT,  -- 'ok' | 'error' | NULL
    ADD COLUMN IF NOT EXISTS ingest_error TEXT;

-- Index for finding scripts needing ingest (sha256 changed since last ingest)
CREATE INDEX IF NOT EXISTS idx_strategy_scripts_ingest_pending
    ON strategy_scripts(workspace_id, sha256)
    WHERE ingest_status IS NULL OR last_ingested_sha IS DISTINCT FROM sha256;

-- Index for joining with documents table
CREATE INDEX IF NOT EXISTS idx_strategy_scripts_doc_id
    ON strategy_scripts(doc_id)
    WHERE doc_id IS NOT NULL;

-- Comments
COMMENT ON COLUMN strategy_scripts.doc_id IS 'Document ID in documents table (from ingest)';
COMMENT ON COLUMN strategy_scripts.last_ingested_at IS 'Last successful ingest timestamp';
COMMENT ON COLUMN strategy_scripts.last_ingested_sha IS 'SHA256 of content at last ingest (for change detection)';
COMMENT ON COLUMN strategy_scripts.ingest_status IS 'Ingest status: ok (success), error (failed), NULL (never ingested)';
COMMENT ON COLUMN strategy_scripts.ingest_error IS 'Last ingest error message (truncated)';
