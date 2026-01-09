-- Add KB ingestion tracking columns to backtest_tune_runs
-- These track when/how a tune_run was ingested into the Knowledge Base

ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS kb_ingested_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS kb_embedding_model_id TEXT,
ADD COLUMN IF NOT EXISTS kb_vector_dim INTEGER,
ADD COLUMN IF NOT EXISTS kb_text_hash TEXT;  -- SHA256[:16] of trial_to_text for drift detection

-- Index for finding non-ingested runs
CREATE INDEX IF NOT EXISTS idx_tune_runs_kb_missing
ON backtest_tune_runs (kb_ingested_at)
WHERE kb_ingested_at IS NULL AND status = 'completed';

-- Index for finding runs by model (useful for re-embedding)
CREATE INDEX IF NOT EXISTS idx_tune_runs_kb_model
ON backtest_tune_runs (kb_embedding_model_id)
WHERE kb_ingested_at IS NOT NULL;

COMMENT ON COLUMN backtest_tune_runs.kb_ingested_at IS 'Timestamp when this run was ingested into the KB vector store';
COMMENT ON COLUMN backtest_tune_runs.kb_embedding_model_id IS 'Embedding model ID used for KB ingestion (e.g., nomic-embed-text)';
COMMENT ON COLUMN backtest_tune_runs.kb_vector_dim IS 'Vector dimension of the embedding used for KB ingestion';
COMMENT ON COLUMN backtest_tune_runs.kb_text_hash IS 'SHA256[:16] hash of trial text for detecting template/content drift';
