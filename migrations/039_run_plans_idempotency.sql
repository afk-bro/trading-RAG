-- Migration: 039_run_plans_idempotency
-- Description: Add idempotency support to run_plans table
-- Purpose: Prevent duplicate run plans on client retries (timeouts, network blips)
-- Dual-key approach:
--   - idempotency_key: client-provided key (optional, via X-Idempotency-Key header)
--   - request_hash: server-computed hash of request body (for automatic dedup)

ALTER TABLE run_plans
    ADD COLUMN IF NOT EXISTS idempotency_key TEXT,
    ADD COLUMN IF NOT EXISTS request_hash TEXT;

-- Unique constraint on idempotency_key (when provided)
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_plans_idempotency_key
    ON run_plans(idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Index for request_hash lookups
CREATE INDEX IF NOT EXISTS idx_run_plans_request_hash
    ON run_plans(request_hash)
    WHERE request_hash IS NOT NULL;
