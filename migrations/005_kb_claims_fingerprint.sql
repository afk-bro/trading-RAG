-- Add claim fingerprint for deduplication
-- Migration: 005_kb_claims_fingerprint
-- Makes repeated learns idempotent by detecting duplicate claims

-- Add fingerprint column
ALTER TABLE kb_claims
ADD COLUMN IF NOT EXISTS fingerprint TEXT;

-- Create unique index for deduplication
-- NULL fingerprints are allowed (for backwards compatibility)
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_claims_fingerprint
ON kb_claims(fingerprint) WHERE fingerprint IS NOT NULL;

-- Add index for faster fingerprint lookups
CREATE INDEX IF NOT EXISTS idx_kb_claims_fingerprint_lookup
ON kb_claims(workspace_id, fingerprint) WHERE fingerprint IS NOT NULL;

-- Comment
COMMENT ON COLUMN kb_claims.fingerprint IS 'SHA-256 hash of normalized(workspace_id|claim_type|entity_name|text) for deduplication';
