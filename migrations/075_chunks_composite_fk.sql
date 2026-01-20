-- Migration: Add composite FK constraint on chunks to enforce workspace_id consistency
-- This ensures that a chunk's workspace_id must match its parent document's workspace_id

-- Step 1: Add unique constraint on documents(id, workspace_id) for FK reference
-- This is needed because FK target must be unique
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'documents_id_workspace_id_unique'
    ) THEN
        ALTER TABLE documents
        ADD CONSTRAINT documents_id_workspace_id_unique
        UNIQUE (id, workspace_id);
    END IF;
END $$;

-- Step 2: Drop the existing simple FK constraint if it exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_chunks_document'
    ) THEN
        ALTER TABLE chunks DROP CONSTRAINT fk_chunks_document;
    END IF;
END $$;

-- Step 3: Add composite FK constraint that validates both doc_id and workspace_id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_chunks_document_workspace'
    ) THEN
        ALTER TABLE chunks
        ADD CONSTRAINT fk_chunks_document_workspace
        FOREIGN KEY (doc_id, workspace_id)
        REFERENCES documents(id, workspace_id)
        ON DELETE CASCADE;
    END IF;
END $$;

-- Note: This constraint ensures that:
-- 1. The doc_id must exist in documents table
-- 2. The workspace_id must match the document's workspace_id
-- 3. Attempting to insert a chunk with mismatched workspace_id will fail
