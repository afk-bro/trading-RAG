-- Trading RAG Pipeline - Initial Schema
-- Run this migration against your Supabase Postgres database

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- Documents Table
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL,
    source_url TEXT,
    canonical_url TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('youtube', 'pdf', 'article', 'note', 'transcript')),
    content_hash TEXT NOT NULL,
    title TEXT,
    author TEXT,
    channel TEXT,
    published_at TIMESTAMPTZ,
    language TEXT DEFAULT 'en',
    duration_secs INTEGER,
    video_id TEXT,
    playlist_id TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'deleted')),
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ,

    -- Unique constraint for deduplication
    CONSTRAINT documents_workspace_source_canonical_unique
        UNIQUE (workspace_id, source_type, canonical_url)
);

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_workspace_published
    ON documents (workspace_id, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_source_type
    ON documents (workspace_id, source_type);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents (content_hash);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_source_url
    ON documents (workspace_id, source_url);

CREATE INDEX IF NOT EXISTS idx_documents_video_id
    ON documents (video_id) WHERE video_id IS NOT NULL;

-- ============================================
-- Chunks Table
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    section TEXT,
    time_start_secs INTEGER,
    time_end_secs INTEGER,
    page_start INTEGER,
    page_end INTEGER,
    locator_label TEXT,
    speaker TEXT,
    symbols TEXT[] DEFAULT '{}',
    entities TEXT[] DEFAULT '{}',
    topics TEXT[] DEFAULT '{}',
    quality_score REAL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Foreign key to documents
    CONSTRAINT fk_chunks_document
        FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,

    -- Unique constraint for chunk ordering
    CONSTRAINT chunks_doc_index_unique
        UNIQUE (doc_id, chunk_index)
);

-- Chunk indexes
CREATE INDEX IF NOT EXISTS idx_chunks_workspace_doc
    ON chunks (workspace_id, doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON chunks (doc_id);

-- GIN indexes for array columns (for filtering)
CREATE INDEX IF NOT EXISTS idx_chunks_symbols_gin
    ON chunks USING GIN (symbols);

CREATE INDEX IF NOT EXISTS idx_chunks_topics_gin
    ON chunks USING GIN (topics);

CREATE INDEX IF NOT EXISTS idx_chunks_entities_gin
    ON chunks USING GIN (entities);

-- ============================================
-- Chunk Vectors Table
-- ============================================
CREATE TABLE IF NOT EXISTS chunk_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL,
    workspace_id UUID NOT NULL,
    embed_provider TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    collection TEXT NOT NULL,
    vector_dim INTEGER NOT NULL,
    qdrant_point_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'indexed', 'failed')),
    indexed_at TIMESTAMPTZ,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Foreign key to chunks
    CONSTRAINT fk_chunk_vectors_chunk
        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,

    -- Unique constraint for embeddings per model/collection
    CONSTRAINT chunk_vectors_chunk_provider_model_collection_unique
        UNIQUE (chunk_id, embed_provider, embed_model, collection)
);

-- Chunk vectors indexes
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_chunk_id
    ON chunk_vectors (chunk_id);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_workspace_collection_status
    ON chunk_vectors (workspace_id, collection, status);

-- ============================================
-- Updated At Trigger Function
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to documents
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to chunks
DROP TRIGGER IF EXISTS update_chunks_updated_at ON chunks;
CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Row Level Security (optional)
-- ============================================
-- Uncomment if you want RLS enabled

-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunk_vectors ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Service role has full access to documents"
--     ON documents FOR ALL
--     USING (true);

-- CREATE POLICY "Service role has full access to chunks"
--     ON chunks FOR ALL
--     USING (true);

-- CREATE POLICY "Service role has full access to chunk_vectors"
--     ON chunk_vectors FOR ALL
--     USING (true);

-- ============================================
-- Grants (for service role)
-- ============================================
-- These should already be in place for service_role in Supabase

-- GRANT ALL ON documents TO service_role;
-- GRANT ALL ON chunks TO service_role;
-- GRANT ALL ON chunk_vectors TO service_role;
