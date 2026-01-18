-- Migration: 060_pine_repos
-- Purpose: GitHub repository registry for Pine script discovery
-- Enables cloning, incremental scanning, and tracking of remote repos

-- ===========================================
-- Pine Repository Registry Table
-- ===========================================

CREATE TABLE pine_repos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Identity
    repo_url TEXT NOT NULL,  -- https://github.com/owner/repo
    repo_slug TEXT NOT NULL, -- owner/repo (normalized, validated)

    -- Clone location (derived: DATA_DIR/repos/{owner}__{repo})
    clone_path TEXT,  -- Populated after first clone

    -- Git state
    branch TEXT NOT NULL DEFAULT 'main',  -- Target branch to track
    last_seen_commit TEXT,  -- SHA at last successful scan

    -- Scan state
    last_scan_at TIMESTAMPTZ,
    last_scan_ok BOOLEAN,
    last_scan_error TEXT,  -- Truncated to 1000 chars
    scripts_count INT NOT NULL DEFAULT 0,

    -- Clone state
    last_pull_at TIMESTAMPTZ,
    last_pull_ok BOOLEAN,
    pull_error TEXT,  -- Truncated to 1000 chars

    -- Config
    enabled BOOLEAN NOT NULL DEFAULT true,
    scan_globs TEXT[] DEFAULT ARRAY['**/*.pine'],  -- Glob patterns

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT uq_pine_repos_workspace_slug UNIQUE (workspace_id, repo_slug),
    CONSTRAINT chk_repo_slug_format CHECK (repo_slug ~ '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
);

-- Updated_at trigger
CREATE TRIGGER set_pine_repos_updated_at
    BEFORE UPDATE ON pine_repos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Indexes
CREATE INDEX idx_pine_repos_enabled ON pine_repos(workspace_id, enabled) WHERE enabled;
CREATE INDEX idx_pine_repos_slug ON pine_repos(repo_slug);

-- Table and column comments
COMMENT ON TABLE pine_repos IS 'GitHub repositories registered for Pine script discovery';
COMMENT ON COLUMN pine_repos.repo_url IS 'Full GitHub URL (https://github.com/owner/repo)';
COMMENT ON COLUMN pine_repos.repo_slug IS 'Normalized owner/repo identifier, validated format';
COMMENT ON COLUMN pine_repos.clone_path IS 'Local filesystem path to cloned repo';
COMMENT ON COLUMN pine_repos.branch IS 'Git branch to track (default: main)';
COMMENT ON COLUMN pine_repos.last_seen_commit IS 'SHA of origin/<branch> at last successful scan';
COMMENT ON COLUMN pine_repos.last_scan_at IS 'Timestamp of last scan attempt';
COMMENT ON COLUMN pine_repos.last_scan_ok IS 'Whether last scan succeeded';
COMMENT ON COLUMN pine_repos.last_scan_error IS 'Error message from last failed scan (truncated)';
COMMENT ON COLUMN pine_repos.scripts_count IS 'Number of scripts discovered in this repo';
COMMENT ON COLUMN pine_repos.scan_globs IS 'Glob patterns for matching Pine files (e.g., **/*.pine)';

-- ===========================================
-- Extend strategy_scripts for GitHub tracking
-- ===========================================

ALTER TABLE strategy_scripts
    ADD COLUMN IF NOT EXISTS repo_id UUID REFERENCES pine_repos(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS scan_commit TEXT,  -- Repo commit at time of scan
    ADD COLUMN IF NOT EXISTS source_url TEXT;   -- Commit-specific GitHub blob URL

-- Index for repo-based queries
CREATE INDEX IF NOT EXISTS idx_strategy_scripts_repo_id
    ON strategy_scripts(repo_id)
    WHERE repo_id IS NOT NULL;

-- Comments for new columns
COMMENT ON COLUMN strategy_scripts.repo_id IS 'Reference to pine_repos for GitHub-sourced scripts';
COMMENT ON COLUMN strategy_scripts.scan_commit IS 'Repo commit SHA when this script was last scanned';
COMMENT ON COLUMN strategy_scripts.source_url IS 'Commit-specific GitHub blob URL for linking';

-- ===========================================
-- Add deleted_at for soft delete support
-- ===========================================

ALTER TABLE strategy_scripts
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- Index for finding active (non-deleted) scripts
CREATE INDEX IF NOT EXISTS idx_strategy_scripts_deleted_at
    ON strategy_scripts(workspace_id, deleted_at)
    WHERE deleted_at IS NULL;

COMMENT ON COLUMN strategy_scripts.deleted_at IS 'Soft delete timestamp (NULL = active, set = deleted)';
