-- Migration: Change documents unique constraint to partial index
-- This allows multiple documents with the same (workspace_id, source_type, canonical_url)
-- as long as only ONE has status = 'active' (others can be 'superseded' or 'deleted')

-- Drop the existing unique constraint
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_workspace_source_canonical_unique;

-- Create a partial unique index that only applies to active documents
-- This allows the supersede/versioning pattern where old versions are marked 'superseded'
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_workspace_source_canonical_active
    ON documents (workspace_id, source_type, canonical_url)
    WHERE status = 'active';

-- Add a comment explaining the constraint
COMMENT ON INDEX idx_documents_workspace_source_canonical_active IS
    'Ensures only one active document per (workspace_id, source_type, canonical_url). Superseded/deleted documents are exempt.';
