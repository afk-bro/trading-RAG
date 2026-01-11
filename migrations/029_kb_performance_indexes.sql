-- Migration 029: Performance Indexes for KB Eligible Trials View
-- Adds composite indexes to support efficient querying of kb_eligible_trials view.
--
-- Note: The view extracts from JSONB for test_variant metrics (summary->sharpe, etc).
-- For high-volume workloads, consider promoting these to first-class columns:
--   - experiment_type (currently COALESCE from summary)
--   - strategy_name (currently from summary or kb_entities join)
--   - sharpe, return_pct, max_drawdown_pct, trade_count (currently from summary JSONB)
-- This would eliminate JSONB extraction at query time.

-- ============================================================================
-- 1. Composite index for backtest_runs (test_variant side of view)
-- ============================================================================

-- Covers: WHERE run_kind = 'test_variant' AND kb_status IN (...) AND status IN (...)
CREATE INDEX IF NOT EXISTS idx_backtest_runs_kb_eligible
    ON backtest_runs(workspace_id, run_kind, kb_status, status)
    WHERE run_kind = 'test_variant'
      AND kb_status IN ('candidate', 'promoted')
      AND status IN ('completed', 'success');

-- Index for created_at ordering (for incremental ingestion with since filter)
CREATE INDEX IF NOT EXISTS idx_backtest_runs_kb_created
    ON backtest_runs(workspace_id, created_at DESC)
    WHERE run_kind = 'test_variant'
      AND kb_status IN ('candidate', 'promoted');

-- ============================================================================
-- 2. Composite index for backtest_tune_runs (tune_run side of view)
-- ============================================================================

-- Covers: WHERE status = 'completed' AND kb_status IN (...)
-- Note: tune_runs don't have run_kind, they're always tune trials
CREATE INDEX IF NOT EXISTS idx_tune_runs_kb_composite
    ON backtest_tune_runs(workspace_id, kb_status, status, created_at DESC)
    WHERE status = 'completed'
      AND kb_status IN ('candidate', 'promoted');

-- ============================================================================
-- 3. Index for kb_trial_index archive queries
-- ============================================================================

-- For finding archived entries to clean up or restore
CREATE INDEX IF NOT EXISTS idx_kb_trial_index_archived
    ON kb_trial_index(workspace_id, archived_at)
    WHERE archived_at IS NOT NULL;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON INDEX idx_backtest_runs_kb_eligible IS
    'Composite index for kb_eligible_trials view - test_variant side';
COMMENT ON INDEX idx_backtest_runs_kb_created IS
    'Index for incremental ingestion with created_at filter';
COMMENT ON INDEX idx_tune_runs_kb_composite IS
    'Composite index for kb_eligible_trials view - tune_run side';
