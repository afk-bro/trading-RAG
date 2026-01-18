-- Migration: 061_pine_repos_polling
-- Purpose: Add polling support columns for automated repo scanning
-- Enables scheduled scans with exponential backoff on failures

-- ===========================================
-- Add polling columns to pine_repos
-- ===========================================

ALTER TABLE pine_repos
    ADD COLUMN IF NOT EXISTS next_scan_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS failure_count INT NOT NULL DEFAULT 0;

-- Index for efficient poll selection query
-- Orders by next_scan_at to find due repos quickly
CREATE INDEX IF NOT EXISTS idx_pine_repos_poll_due
    ON pine_repos(next_scan_at NULLS FIRST, last_scan_at ASC)
    WHERE enabled = true;

-- Comments
COMMENT ON COLUMN pine_repos.next_scan_at IS 'Scheduled time for next poll scan (NULL = scan immediately)';
COMMENT ON COLUMN pine_repos.failure_count IS 'Consecutive scan failures for exponential backoff';
