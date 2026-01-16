-- Strategy Registry v1: multi-engine strategy catalog with backtest summary
-- Migration: 049_strategies.sql

CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Identity
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT,

    -- Engine & Source
    engine TEXT NOT NULL DEFAULT 'pine',
    source_ref JSONB NOT NULL DEFAULT '{}',
    -- pine: {store:"local"|"github", path:"...", doc_id:"...", repo:"...", ref:"..."}
    -- python: {module:"...", entrypoint:"...", params_schema:{...}}

    -- Status
    status TEXT NOT NULL DEFAULT 'draft',
    review_status TEXT NOT NULL DEFAULT 'unreviewed',
    risk_level TEXT,

    -- Tags (mirrors MatchIntent for coverage overlap)
    tags JSONB NOT NULL DEFAULT '{}',
    -- {strategy_archetypes:[], indicators:[], timeframe_buckets:[], topics:[], risk_terms:[]}

    -- Backtest Summary (JSONB for v1 speed)
    backtest_summary JSONB,
    -- {status:"never"|"queued"|"running"|"complete"|"failed", last_backtest_at:...,
    --  best_oos_score:..., max_drawdown:..., num_trades:...,
    --  dataset_coverage:{symbols:[], start_date:..., end_date:...},
    --  rigor:{fees:..., slippage:..., walk_forward:...}, notes:...}

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT strategies_engine_check CHECK (engine IN ('pine', 'python', 'vectorbt', 'backtesting_py')),
    CONSTRAINT strategies_status_check CHECK (status IN ('draft', 'active', 'archived')),
    CONSTRAINT strategies_review_status_check CHECK (review_status IN ('unreviewed', 'reviewed', 'approved', 'rejected')),
    CONSTRAINT strategies_risk_level_check CHECK (risk_level IS NULL OR risk_level IN ('low', 'medium', 'high'))
);

-- Unique slug per workspace
CREATE UNIQUE INDEX IF NOT EXISTS idx_strategies_workspace_slug
    ON strategies(workspace_id, slug);

-- Filter by workspace + status (common query)
CREATE INDEX IF NOT EXISTS idx_strategies_workspace_status
    ON strategies(workspace_id, status);

-- Filter by workspace + active (cockpit list)
CREATE INDEX IF NOT EXISTS idx_strategies_workspace_active
    ON strategies(workspace_id) WHERE status = 'active';

-- GIN index on tags for overlap queries with MatchIntent
CREATE INDEX IF NOT EXISTS idx_strategies_tags_gin
    ON strategies USING GIN (tags);

-- Backtest status filtering (optional, can add later if needed)
-- CREATE INDEX IF NOT EXISTS idx_strategies_backtest_status
--     ON strategies((backtest_summary->>'status')) WHERE backtest_summary IS NOT NULL;

-- Comments
COMMENT ON TABLE strategies IS 'Multi-engine strategy registry with metadata, tags, and backtest summary';
COMMENT ON COLUMN strategies.engine IS 'Execution engine: pine, python, vectorbt, backtesting_py';
COMMENT ON COLUMN strategies.source_ref IS 'Engine-specific source pointer (path, module, doc_id, etc.)';
COMMENT ON COLUMN strategies.tags IS 'MatchIntent-compatible tags for coverage overlap computation';
COMMENT ON COLUMN strategies.backtest_summary IS 'Latest backtest results (JSONB cache for v1)';
COMMENT ON COLUMN strategies.review_status IS 'Human review workflow: unreviewed -> reviewed -> approved/rejected';
