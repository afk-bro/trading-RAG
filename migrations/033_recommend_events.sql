-- Migration: 033_recommend_events
-- Description: Create recommend_events table for analytics and attribution tracking
-- Purpose: Track tiered recommendation calls for regime coverage, tier usage, and uplift analysis

-- =============================================================================
-- recommend_events table
-- =============================================================================
-- Lightweight event log for tiered recommendations.
-- Used for:
--   1. Regime Coverage: "Do we have enough data per regime?"
--   2. Tier Usage: "How often are we falling back?"
--   3. Value Add: "Does regime selection outperform baseline?"

CREATE TABLE IF NOT EXISTS recommend_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Request context
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID REFERENCES strategy_entities(id) ON DELETE SET NULL,

    -- Query regime context
    query_regime_key TEXT,
    query_trend_tag TEXT,
    query_vol_tag TEXT,

    -- Result metadata
    tier_used TEXT NOT NULL,  -- exact | partial_trend | partial_vol | distance | global_best
    tiers_attempted JSONB NOT NULL DEFAULT '[]',  -- ["exact", "partial", "distance", "global"]

    -- Per-tier counts
    exact_count INT NOT NULL DEFAULT 0,
    partial_trend_count INT NOT NULL DEFAULT 0,
    partial_vol_count INT NOT NULL DEFAULT 0,
    distance_count INT NOT NULL DEFAULT 0,
    global_count INT NOT NULL DEFAULT 0,

    -- Request parameters
    k INT NOT NULL DEFAULT 20,
    min_samples INT NOT NULL DEFAULT 5,

    -- Result metrics
    candidate_count INT NOT NULL DEFAULT 0,
    top_candidate_score FLOAT,  -- best_oos_score of top candidate (for uplift calc)
    confidence FLOAT,  -- computed confidence (1.0 exact, 0.7 partial, 0.5 distance, 0.3 global)

    -- Performance
    duration_ms FLOAT NOT NULL DEFAULT 0,

    -- Distance tier metadata (for reproducibility)
    distance_method TEXT,  -- "euclidean" | "cosine" | "weighted"
    distance_features_version TEXT  -- "regime_v1"
);

-- =============================================================================
-- Indexes for analytics queries
-- =============================================================================

-- Time-series queries (tier usage over time)
CREATE INDEX IF NOT EXISTS idx_recommend_events_created_at
    ON recommend_events(created_at DESC);

-- Workspace + strategy scoped queries
CREATE INDEX IF NOT EXISTS idx_recommend_events_workspace_strategy
    ON recommend_events(workspace_id, strategy_entity_id, created_at DESC);

-- Tier usage analytics
CREATE INDEX IF NOT EXISTS idx_recommend_events_tier_used
    ON recommend_events(tier_used, created_at DESC);

-- Regime coverage queries
CREATE INDEX IF NOT EXISTS idx_recommend_events_regime
    ON recommend_events(workspace_id, query_regime_key)
    WHERE query_regime_key IS NOT NULL;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE recommend_events IS
    'Lightweight event log for tiered recommendation analytics. Tracks tier usage, regime coverage, and enables uplift analysis.';

COMMENT ON COLUMN recommend_events.tier_used IS
    'Final tier that satisfied min_samples: exact, partial_trend, partial_vol, distance, global_best';

COMMENT ON COLUMN recommend_events.confidence IS
    'Trust score based on tier: 1.0 (exact), 0.7 (partial), 0.5 (distance), 0.3 (global_best)';

COMMENT ON COLUMN recommend_events.top_candidate_score IS
    'best_oos_score of the top-ranked candidate. Used for uplift = score(regime) - score(baseline)';
