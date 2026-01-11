-- Migration 028: KB Eligible Trials View
-- Implements Phase 4 of the trial ingestion design:
--   - Unified view of tune_runs and test_variants eligible for KB ingestion
--   - Filters by kb_status and trial_status
--   - Provides consistent columns for ingestion pipeline

-- ============================================================================
-- Drop existing view if it exists (for re-running migration)
-- ============================================================================

DROP VIEW IF EXISTS kb_eligible_trials;

-- ============================================================================
-- Create unified eligible trials view
-- ============================================================================

CREATE VIEW kb_eligible_trials AS
  -- Tune runs (from parameter tuning)
  SELECT
    'tune_run'::TEXT AS source_type,
    'tune'::TEXT AS experiment_type,
    tr.run_id AS source_id,
    tr.tune_id AS group_id,
    t.workspace_id,
    t.strategy_entity_id,
    -- Strategy name from kb_entities
    e.name AS strategy_name,
    tr.params,
    tr.status AS trial_status,
    -- Regime stored in metrics JSONB for tune_runs
    tr.metrics_is->'regime' AS regime_is,
    tr.metrics_oos->'regime' AS regime_oos,
    COALESCE(tr.metrics_is->'regime'->>'schema_version', 'regime_v1') AS regime_schema_version,
    -- OOS metrics (fractions, not percentages)
    (tr.metrics_oos->>'sharpe')::FLOAT AS sharpe_oos,
    (tr.metrics_oos->>'return_pct')::FLOAT AS return_frac_oos,
    (tr.metrics_oos->>'max_drawdown_pct')::FLOAT AS max_dd_frac_oos,
    (tr.metrics_oos->>'trades')::INT AS n_trades_oos,
    -- IS metrics for overfit calculation
    (tr.metrics_is->>'sharpe')::FLOAT AS sharpe_is,
    -- KB status
    tr.kb_status,
    tr.kb_promoted_at,
    tr.kb_status_changed_at,
    -- Objective
    t.objective_type,
    tr.objective_score,
    -- Timestamps
    tr.created_at
  FROM backtest_tune_runs tr
  JOIN backtest_tunes t ON tr.tune_id = t.id
  LEFT JOIN kb_entities e ON t.strategy_entity_id = e.id
  WHERE tr.kb_status IN ('candidate', 'promoted')
    AND tr.status = 'completed'

  UNION ALL

  -- Test variants (from run plans)
  SELECT
    'test_variant'::TEXT AS source_type,
    COALESCE(r.summary->>'experiment_type', 'sweep')::TEXT AS experiment_type,
    r.id AS source_id,
    r.run_plan_id AS group_id,
    r.workspace_id,
    r.strategy_entity_id,
    -- Strategy name from summary or kb_entities
    COALESCE(r.summary->>'strategy_name', e.name) AS strategy_name,
    r.params,
    r.status AS trial_status,
    -- Regime stored as first-class columns for test_variants
    r.regime_is,
    r.regime_oos,
    r.regime_schema_version,
    -- OOS metrics from summary (test variants use full run as OOS)
    (r.summary->>'sharpe')::FLOAT AS sharpe_oos,
    (r.summary->>'return_pct')::FLOAT AS return_frac_oos,
    (r.summary->>'max_drawdown_pct')::FLOAT AS max_dd_frac_oos,
    (r.summary->>'trade_count')::INT AS n_trades_oos,
    -- No IS metrics for test variants (no IS/OOS split)
    NULL::FLOAT AS sharpe_is,
    -- KB status
    r.kb_status,
    r.kb_promoted_at,
    r.kb_status_changed_at,
    -- Objective (from run_plan via join, or default)
    COALESCE(rp.objective_name, 'sharpe') AS objective_type,
    r.objective_score,
    -- Timestamps
    r.created_at
  FROM backtest_runs r
  LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
  LEFT JOIN run_plans rp ON r.run_plan_id = rp.id
  WHERE r.run_kind = 'test_variant'
    AND r.kb_status IN ('candidate', 'promoted')
    AND r.status IN ('completed', 'success')
    -- Policy: candidates need regime_oos, promoted can skip
    AND (
      r.kb_status = 'promoted'
      OR (r.kb_status = 'candidate' AND r.regime_oos IS NOT NULL)
    );

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON VIEW kb_eligible_trials IS 'Unified view of trials eligible for KB ingestion (candidate or promoted status)';

-- ============================================================================
-- Indexes to support view performance
-- ============================================================================

-- Index for tune_runs by kb_status (already exists from 027)
CREATE INDEX IF NOT EXISTS idx_tune_runs_kb_status
    ON backtest_tune_runs(kb_status)
    WHERE kb_status IN ('candidate', 'promoted');

-- Index for filtering completed tune_runs with eligible status
CREATE INDEX IF NOT EXISTS idx_tune_runs_eligible
    ON backtest_tune_runs(tune_id, status, kb_status)
    WHERE status = 'completed' AND kb_status IN ('candidate', 'promoted');
