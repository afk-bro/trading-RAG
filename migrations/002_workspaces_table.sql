-- Trading RAG Pipeline - Workspaces Table Migration
-- Run this migration against your Supabase Postgres database

-- ============================================
-- Workspaces Table
-- ============================================
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    owner_id UUID,
    default_vector_store TEXT NOT NULL DEFAULT 'qdrant',
    default_collection TEXT,
    default_embed_provider TEXT NOT NULL DEFAULT 'ollama',
    default_embed_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
    default_distance TEXT NOT NULL DEFAULT 'cosine' CHECK (default_distance IN ('cosine', 'dot', 'euclid')),
    is_active BOOLEAN NOT NULL DEFAULT true,
    ingestion_enabled BOOLEAN NOT NULL DEFAULT true,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workspaces indexes
CREATE INDEX IF NOT EXISTS idx_workspaces_slug ON workspaces (slug);
CREATE INDEX IF NOT EXISTS idx_workspaces_owner_id ON workspaces (owner_id) WHERE owner_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_workspaces_is_active ON workspaces (is_active) WHERE is_active = true;

-- Apply updated_at trigger to workspaces
DROP TRIGGER IF EXISTS update_workspaces_updated_at ON workspaces;
CREATE TRIGGER update_workspaces_updated_at
    BEFORE UPDATE ON workspaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Add Foreign Key Constraints to Documents Table
-- ============================================
-- First, check if constraint exists before adding
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_workspace'
    ) THEN
        ALTER TABLE documents
        ADD CONSTRAINT fk_documents_workspace
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id);
    END IF;
END $$;

-- ============================================
-- Insert Default Test Workspace
-- ============================================
INSERT INTO workspaces (id, name, slug, is_active, ingestion_enabled, config)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Test Workspace',
    'test',
    true,
    true,
    '{"chunking": {"size": 512, "overlap": 50}, "retrieval": {"top_k": 8, "min_score": 0.18}}'::jsonb
)
ON CONFLICT (slug) DO NOTHING;

-- Also insert a default workspace for general use
INSERT INTO workspaces (id, name, slug, is_active, ingestion_enabled, config)
VALUES (
    'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
    'Default Workspace',
    'default',
    true,
    true,
    '{"chunking": {"size": 512, "overlap": 50}, "retrieval": {"top_k": 8, "min_score": 0.18}}'::jsonb
)
ON CONFLICT (slug) DO NOTHING;
