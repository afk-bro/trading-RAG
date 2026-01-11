-- Migration: 032_tune_regime_attribution
-- Add regime attribution columns to backtest_tunes for regime-based queries

-- Regime versioning (for fingerprint stability)
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS regime_schema_version TEXT,
  ADD COLUMN IF NOT EXISTS tag_ruleset_id TEXT;

-- Human-readable regime key: "regime_v1_1|default_v1|uptrend|high_vol|noisy"
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS regime_key TEXT;

-- SHA256 hash for indexing (derived from regime_key)
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS regime_fingerprint TEXT;

-- Denormalized tags for fast SQL filtering
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS trend_tag TEXT;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS vol_tag TEXT;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS efficiency_tag TEXT;

-- Explicit OOS best (existing best_score may be IS or ambiguous)
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS best_oos_score DOUBLE PRECISION;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS best_oos_params JSONB;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS best_oos_run_id UUID REFERENCES backtest_runs(id) ON DELETE SET NULL;

-- Optional: baseline for uplift calculation
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS baseline_oos_score DOUBLE PRECISION;

-- Fast regime-based lookups
CREATE INDEX IF NOT EXISTS idx_tunes_regime_key
  ON backtest_tunes(regime_key)
  WHERE regime_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tunes_trend_vol
  ON backtest_tunes(trend_tag, vol_tag)
  WHERE trend_tag IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tunes_best_oos_score
  ON backtest_tunes(best_oos_score DESC NULLS LAST)
  WHERE best_oos_score IS NOT NULL;

-- Multi-tenant regime lookup
CREATE INDEX IF NOT EXISTS idx_tunes_workspace_regime
  ON backtest_tunes(workspace_id, regime_key)
  WHERE regime_key IS NOT NULL;
