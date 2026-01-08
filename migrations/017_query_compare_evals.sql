-- Query compare evaluation storage
-- Persists structured eval logs from /query/compare for analytics

CREATE TABLE IF NOT EXISTS query_compare_evals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    workspace_id UUID NOT NULL,

    -- Question identification (hash for privacy, optional preview)
    question_hash TEXT NOT NULL,
    question_preview TEXT NULL,  -- First 80 chars, only if EVAL_STORE_QUESTION_PREVIEW=true

    -- Config fingerprint for grouping by settings
    config_fingerprint TEXT NOT NULL,

    -- Rerank config
    rerank_method TEXT NULL,
    rerank_model TEXT NULL,
    candidates_k INT NOT NULL,
    top_k INT NOT NULL,
    share_candidates BOOLEAN NOT NULL DEFAULT true,
    skip_neighbors BOOLEAN NOT NULL DEFAULT true,

    -- Metrics
    jaccard DOUBLE PRECISION NOT NULL,
    spearman DOUBLE PRECISION NULL,
    rank_delta_mean DOUBLE PRECISION NULL,
    rank_delta_max INT NULL,
    overlap_count INT NOT NULL,
    union_count INT NOT NULL,

    -- Latency (ms)
    embed_ms INT NOT NULL,
    search_ms INT NOT NULL,
    vector_total_ms INT NOT NULL,
    rerank_ms INT NULL,
    rerank_total_ms INT NULL,

    -- State
    rerank_state TEXT NOT NULL,
    rerank_timeout BOOLEAN NOT NULL DEFAULT false,
    rerank_fallback BOOLEAN NOT NULL DEFAULT false,

    -- Spot-check IDs
    vector_top5_ids TEXT[] NULL,
    reranked_top5_ids TEXT[] NULL,

    -- Raw payload for future schema evolution
    payload JSONB NOT NULL
);

-- Primary query patterns
CREATE INDEX idx_qce_workspace_created
    ON query_compare_evals (workspace_id, created_at DESC);

CREATE INDEX idx_qce_workspace_config_created
    ON query_compare_evals (workspace_id, config_fingerprint, created_at DESC);

CREATE INDEX idx_qce_question_hash
    ON query_compare_evals (question_hash);

-- Partial index for "impacted" queries (jaccard < 0.8)
CREATE INDEX idx_qce_impacted
    ON query_compare_evals (workspace_id, created_at DESC)
    WHERE jaccard < 0.8;

-- Comment
COMMENT ON TABLE query_compare_evals IS
    'Stores evaluation metrics from /query/compare endpoint for rerank tuning analytics';
