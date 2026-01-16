-- Add explanations cache to match_runs for LLM-generated strategy explanations
-- Migration: 052_match_runs_explanations_cache.sql

-- Cache structure: {strategy_id: {explanation, model, provider, latency_ms, generated_at, strategy_updated_at}}
ALTER TABLE match_runs
ADD COLUMN IF NOT EXISTS explanations_cache JSONB DEFAULT '{}';

COMMENT ON COLUMN match_runs.explanations_cache IS 'Cached LLM explanations per strategy_id, invalidated if strategy changes';
