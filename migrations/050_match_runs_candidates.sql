-- Add candidate strategy columns to match_runs for cockpit display
-- Migration: 050_match_runs_candidates.sql

-- Store candidate strategy IDs computed at match time (point-in-time snapshot)
ALTER TABLE match_runs
    ADD COLUMN IF NOT EXISTS candidate_strategy_ids UUID[] DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS candidate_scores JSONB;

-- Index for filtering runs that have candidates
CREATE INDEX IF NOT EXISTS idx_match_runs_has_candidates
    ON match_runs (workspace_id, created_at DESC)
    WHERE candidate_strategy_ids IS NOT NULL AND array_length(candidate_strategy_ids, 1) > 0;

-- Index for weak coverage queries (cockpit left panel)
CREATE INDEX IF NOT EXISTS idx_match_runs_weak_coverage
    ON match_runs (workspace_id, created_at DESC)
    WHERE weak_coverage = true;

-- Comments
COMMENT ON COLUMN match_runs.candidate_strategy_ids IS 'Strategy IDs with tag overlap at match time (point-in-time snapshot)';
COMMENT ON COLUMN match_runs.candidate_scores IS 'Candidate scores: {strategy_id: {score, matched_tags}}';
