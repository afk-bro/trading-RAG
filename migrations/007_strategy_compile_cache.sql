-- Migration: 007_strategy_compile_cache
-- Add columns to cache compile artifacts (deterministic for spec version)

ALTER TABLE kb_strategy_specs
ADD COLUMN IF NOT EXISTS compiled_param_schema JSONB,
ADD COLUMN IF NOT EXISTS compiled_backtest_config JSONB,
ADD COLUMN IF NOT EXISTS compiled_pseudocode TEXT,
ADD COLUMN IF NOT EXISTS compiled_at TIMESTAMPTZ;

COMMENT ON COLUMN kb_strategy_specs.compiled_param_schema IS 'Cached JSON Schema for UI parameter forms';
COMMENT ON COLUMN kb_strategy_specs.compiled_backtest_config IS 'Cached engine-agnostic backtest configuration';
COMMENT ON COLUMN kb_strategy_specs.compiled_pseudocode IS 'Cached human-readable strategy pseudocode';
COMMENT ON COLUMN kb_strategy_specs.compiled_at IS 'Timestamp of last compilation';
