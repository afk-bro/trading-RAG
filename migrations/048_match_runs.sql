-- Coverage gap detection: track every /match-pine call for analytics
-- Migration: 048_match_runs.sql

CREATE TABLE IF NOT EXISTS match_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Source info
    source_type TEXT NOT NULL DEFAULT 'youtube',
    source_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    video_id TEXT,

    -- Intent extraction
    intent_signature TEXT NOT NULL,  -- SHA256 hash of canonical intent
    intent_json JSONB,               -- Full MatchIntent for analysis

    -- Query info
    query_used TEXT NOT NULL,
    filters_applied JSONB NOT NULL DEFAULT '{}',
    top_k INT NOT NULL DEFAULT 10,

    -- Results
    total_searched INT NOT NULL DEFAULT 0,
    best_score FLOAT,
    avg_top_k_score FLOAT,
    num_above_threshold INT NOT NULL DEFAULT 0,

    -- Coverage assessment
    weak_coverage BOOLEAN NOT NULL DEFAULT FALSE,
    reason_codes TEXT[] NOT NULL DEFAULT '{}',

    -- Indexing
    CONSTRAINT match_runs_workspace_id_fk FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
);

-- Index for coverage gap queries
CREATE INDEX IF NOT EXISTS idx_match_runs_workspace_weak
    ON match_runs(workspace_id, weak_coverage, created_at DESC);

-- Index for intent deduplication/aggregation
CREATE INDEX IF NOT EXISTS idx_match_runs_intent_signature
    ON match_runs(workspace_id, intent_signature);

-- Index for video lookup
CREATE INDEX IF NOT EXISTS idx_match_runs_video_id
    ON match_runs(video_id) WHERE video_id IS NOT NULL;

COMMENT ON TABLE match_runs IS 'Tracks every /match-pine call for coverage gap detection and analytics';
COMMENT ON COLUMN match_runs.intent_signature IS 'SHA256 hash of canonical intent for dedup/aggregation';
COMMENT ON COLUMN match_runs.weak_coverage IS 'True if coverage gap detected (best_score < 0.45 OR num_above_threshold == 0)';
COMMENT ON COLUMN match_runs.reason_codes IS 'Coverage gap reason codes: NO_RESULTS_ABOVE_THRESHOLD, LOW_BEST_SCORE, LOW_SIGNAL_INPUT';
