-- Combined migration script: 012 through 019
-- Run this in Supabase SQL Editor to apply all missing migrations
-- Safe to run multiple times (uses IF NOT EXISTS / IF EXISTS)

BEGIN;

-- ============================================================================
-- 012: tune_run_failed_reason
-- ============================================================================
ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS failed_reason TEXT,
ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_tune_runs_failed_reason
ON backtest_tune_runs(tune_id, failed_reason)
WHERE failed_reason IS NOT NULL;

-- ============================================================================
-- 013: oos_split
-- ============================================================================
ALTER TABLE backtest_tunes
ADD COLUMN IF NOT EXISTS oos_ratio DOUBLE PRECISION;

-- Drop constraint if exists before re-adding (idempotent)
ALTER TABLE backtest_tunes
DROP CONSTRAINT IF EXISTS chk_oos_ratio_range;

ALTER TABLE backtest_tunes
ADD CONSTRAINT chk_oos_ratio_range
CHECK (oos_ratio IS NULL OR (oos_ratio > 0 AND oos_ratio <= 0.5));

ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS score_is DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS score_oos DOUBLE PRECISION;

-- ============================================================================
-- 014: tune_runs_metrics_jsonb
-- ============================================================================
ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS metrics_is JSONB,
ADD COLUMN IF NOT EXISTS metrics_oos JSONB;

-- ============================================================================
-- 015: composite_objective
-- ============================================================================
ALTER TABLE backtest_tunes
ADD COLUMN IF NOT EXISTS objective_type VARCHAR(50) DEFAULT 'sharpe',
ADD COLUMN IF NOT EXISTS objective_params JSONB;

ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS objective_score DOUBLE PRECISION;

-- ============================================================================
-- 016: tunes_gates_jsonb
-- ============================================================================
ALTER TABLE backtest_tunes
ADD COLUMN IF NOT EXISTS gates JSONB;

-- ============================================================================
-- 017: query_compare_evals (new table)
-- ============================================================================
CREATE TABLE IF NOT EXISTS query_compare_evals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    workspace_id UUID NOT NULL,
    question_hash TEXT NOT NULL,
    question_preview TEXT NULL,
    config_fingerprint TEXT NOT NULL,
    rerank_method TEXT NULL,
    rerank_model TEXT NULL,
    candidates_k INT NOT NULL,
    top_k INT NOT NULL,
    share_candidates BOOLEAN NOT NULL DEFAULT true,
    skip_neighbors BOOLEAN NOT NULL DEFAULT true,
    jaccard DOUBLE PRECISION NOT NULL,
    spearman DOUBLE PRECISION NULL,
    rank_delta_mean DOUBLE PRECISION NULL,
    rank_delta_max INT NULL,
    overlap_count INT NOT NULL,
    union_count INT NOT NULL,
    embed_ms INT NOT NULL,
    search_ms INT NOT NULL,
    vector_total_ms INT NOT NULL,
    rerank_ms INT NULL,
    rerank_total_ms INT NULL,
    rerank_state TEXT NOT NULL,
    rerank_timeout BOOLEAN NOT NULL DEFAULT false,
    rerank_fallback BOOLEAN NOT NULL DEFAULT false,
    vector_top5_ids TEXT[] NULL,
    reranked_top5_ids TEXT[] NULL,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qce_workspace_created
    ON query_compare_evals (workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qce_workspace_config_created
    ON query_compare_evals (workspace_id, config_fingerprint, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qce_question_hash
    ON query_compare_evals (question_hash);
CREATE INDEX IF NOT EXISTS idx_qce_impacted
    ON query_compare_evals (workspace_id, created_at DESC)
    WHERE jaccard < 0.8;

-- ============================================================================
-- 018: kb_ingestion_tracking
-- ============================================================================
ALTER TABLE backtest_tune_runs
ADD COLUMN IF NOT EXISTS kb_ingested_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS kb_embedding_model_id TEXT,
ADD COLUMN IF NOT EXISTS kb_vector_dim INTEGER,
ADD COLUMN IF NOT EXISTS kb_text_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_tune_runs_kb_missing
ON backtest_tune_runs (kb_ingested_at)
WHERE kb_ingested_at IS NULL AND status = 'completed';

CREATE INDEX IF NOT EXISTS idx_tune_runs_kb_model
ON backtest_tune_runs (kb_embedding_model_id)
WHERE kb_ingested_at IS NOT NULL;

-- ============================================================================
-- 019: v1.5 Live Intelligence tables
-- ============================================================================

-- Table 1: regime_cluster_stats
CREATE TABLE IF NOT EXISTS regime_cluster_stats (
    strategy_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,
    timeframe TEXT NOT NULL,
    regime_key TEXT NOT NULL,
    regime_dims JSONB NOT NULL,
    n INT NOT NULL,
    feature_schema_version INT NOT NULL DEFAULT 1,
    feature_mean JSONB NOT NULL,
    feature_var JSONB NOT NULL,
    feature_min JSONB,
    feature_max JSONB,
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (strategy_entity_id, timeframe, regime_key)
);

CREATE INDEX IF NOT EXISTS idx_cluster_stats_strategy_timeframe
    ON regime_cluster_stats(strategy_entity_id, timeframe);

-- Table 2: regime_duration_stats
CREATE TABLE IF NOT EXISTS regime_duration_stats (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    regime_key TEXT NOT NULL,
    n_segments INT NOT NULL,
    median_duration_bars INT NOT NULL,
    p25_duration_bars INT NOT NULL,
    p75_duration_bars INT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, timeframe, regime_key)
);

CREATE INDEX IF NOT EXISTS idx_duration_stats_timeframe_key
    ON regime_duration_stats(timeframe, regime_key);

-- Table 3: recommendation_records
CREATE TABLE IF NOT EXISTS recommendation_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL,
    strategy_entity_id UUID NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    params_json JSONB NOT NULL,
    params_hash TEXT NOT NULL,
    regime_key_start TEXT NOT NULL,
    regime_dims_start JSONB NOT NULL,
    regime_features_start JSONB NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    confidence_json JSONB NOT NULL,
    expected_baselines_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_records_active_unique
    ON recommendation_records(workspace_id, strategy_entity_id, symbol, timeframe)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_records_workspace_status
    ON recommendation_records(workspace_id, status);

-- Table 4: recommendation_observations
CREATE TABLE IF NOT EXISTS recommendation_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL REFERENCES recommendation_records(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    bars_seen INT NOT NULL,
    trades_seen INT NOT NULL,
    realized_metrics_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (record_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_observations_record_ts
    ON recommendation_observations(record_id, ts DESC);

-- Table 5: recommendation_evaluation_slices
CREATE TABLE IF NOT EXISTS recommendation_evaluation_slices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL REFERENCES recommendation_records(id) ON DELETE CASCADE,
    slice_start_ts TIMESTAMPTZ NOT NULL,
    slice_end_ts TIMESTAMPTZ NOT NULL,
    trigger_type TEXT NOT NULL,
    regime_key_during TEXT NOT NULL,
    realized_summary_json JSONB NOT NULL,
    expected_summary_json JSONB NOT NULL,
    performance_surprise_z FLOAT,
    drift_flags_json JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_slices_record_end
    ON recommendation_evaluation_slices(record_id, slice_end_ts DESC);

CREATE INDEX IF NOT EXISTS idx_slices_trigger_type
    ON recommendation_evaluation_slices(trigger_type);

-- ============================================================================
-- Add canceled status to backtest_tunes if not present
-- ============================================================================
ALTER TABLE backtest_tunes
DROP CONSTRAINT IF EXISTS backtest_tunes_status_check;

ALTER TABLE backtest_tunes
ADD CONSTRAINT backtest_tunes_status_check
CHECK (status IN ('queued', 'running', 'completed', 'failed', 'canceled'));

COMMIT;

SELECT 'All migrations 012-019 applied successfully!' as result;
