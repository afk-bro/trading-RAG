-- Migration: 053_idempotency_keys
-- Purpose: Generic idempotency support for API endpoints to prevent duplicate operations on client retries
-- Design: Separate table (not columns on individual tables) for reusability across endpoints

CREATE TABLE IF NOT EXISTS idempotency_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    idempotency_key TEXT NOT NULL CHECK (char_length(idempotency_key) <= 200),  -- Length-limited to prevent index bloat
    request_hash TEXT NOT NULL,       -- Full 64-char SHA256 hex of canonical request JSON
    endpoint TEXT NOT NULL,           -- e.g., 'backtests.tune', 'testing.run-plans'
    http_method TEXT NOT NULL DEFAULT 'POST',  -- Track method for safety
    api_version TEXT,                 -- Schema version to detect behavior changes
    resource_id UUID,                 -- e.g., tune_id after creation
    response_json JSONB,              -- Exact response for replay (success case)
    error_code TEXT,                  -- Error classification for failed requests
    error_json JSONB,                 -- Error details for failed requests (for deterministic replay)
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, completed, failed
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- For detecting stale pending keys
    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '7 days',  -- Long enough for slow jobs

    -- Include endpoint in constraint: same key can be used across different endpoints
    CONSTRAINT uq_idempotency_workspace_endpoint_key UNIQUE (workspace_id, endpoint, idempotency_key)
);

-- Index for expiry cleanup (retention job)
CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at);

-- Index for resource lookup (if needed to find idempotency record by resource)
CREATE INDEX IF NOT EXISTS idx_idempotency_resource ON idempotency_keys(resource_id) WHERE resource_id IS NOT NULL;

-- Index for finding stale pending keys
CREATE INDEX IF NOT EXISTS idx_idempotency_pending_created ON idempotency_keys(created_at) WHERE status = 'pending';

-- Comment explaining the idempotency contract
COMMENT ON TABLE idempotency_keys IS 'Generic idempotency support for API endpoints. Keys expire after 7 days. Request hashes use SHA256 of canonical JSON (sort_keys=True, separators=(",",":"), floats normalized to 10 decimals for known float fields only).';

COMMENT ON COLUMN idempotency_keys.idempotency_key IS 'Client-provided idempotency key. Max 200 chars to prevent index bloat.';
COMMENT ON COLUMN idempotency_keys.request_hash IS '64-char SHA256 hex of canonical request JSON. Used to detect key reuse with different payloads (409 Conflict).';
COMMENT ON COLUMN idempotency_keys.updated_at IS 'Last status change. Used to detect stale pending keys (worker crash recovery).';
COMMENT ON COLUMN idempotency_keys.error_code IS 'Classification of failure (e.g., validation_error, internal_error). For deterministic failed replay.';
