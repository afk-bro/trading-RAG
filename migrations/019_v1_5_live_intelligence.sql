-- migrations/019_v1_5_live_intelligence.sql
-- v1.5 "Live Intelligence" schema additions
-- Design doc: docs/plans/2026-01-09-v1.5-live-intelligence-design.md

-- =============================================================================
-- Table 1: regime_cluster_stats
-- Feature centroids + variances per regime key (for distance_z scaling)
-- =============================================================================

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

COMMENT ON TABLE regime_cluster_stats IS 'v1.5: Feature centroids and variances per regime key for distance_z scaling';

-- =============================================================================
-- Table 2: regime_duration_stats
-- Duration distributions per regime key (for persistence estimation)
-- =============================================================================

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

COMMENT ON TABLE regime_duration_stats IS 'v1.5: Historical regime duration distributions for persistence estimation';
-- NOTE: Intentionally NOT workspace-scoped. Duration stats describe market-level
-- behavior (how long regimes typically last for a given symbol/timeframe) and are
-- shared across all workspaces. Keyed by market properties, not strategy.

-- =============================================================================
-- Table 3: recommendation_records
-- Expectation contracts (long-lived, one active per symbol+strategy)
-- =============================================================================

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
    -- Status: active | superseded | inactive | closed

    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Only one active recommendation per symbol+strategy+workspace
CREATE UNIQUE INDEX IF NOT EXISTS idx_records_active_unique
    ON recommendation_records(workspace_id, strategy_entity_id, symbol, timeframe)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_records_workspace_status
    ON recommendation_records(workspace_id, status);

COMMENT ON TABLE recommendation_records IS 'v1.5: Expectation contracts for expected-vs-realized tracking';

-- =============================================================================
-- Table 4: recommendation_observations
-- Streaming realized metrics (append-only)
-- =============================================================================

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

COMMENT ON TABLE recommendation_observations IS 'v1.5: Streaming realized metrics for forward runs';

-- =============================================================================
-- Table 5: recommendation_evaluation_slices
-- Accountability checkpoints (immutable snapshots)
-- =============================================================================

CREATE TABLE IF NOT EXISTS recommendation_evaluation_slices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL REFERENCES recommendation_records(id) ON DELETE CASCADE,

    slice_start_ts TIMESTAMPTZ NOT NULL,
    slice_end_ts TIMESTAMPTZ NOT NULL,

    trigger_type TEXT NOT NULL,
    -- Trigger: regime_change | milestone | manual

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

COMMENT ON TABLE recommendation_evaluation_slices IS 'v1.5: Accountability checkpoints on regime change or milestones';
