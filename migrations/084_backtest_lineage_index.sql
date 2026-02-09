-- Index for lineage queries: find previous completed run for same strategy.
-- NOTE: Do not use CONCURRENTLY — migration runner wraps in transaction.
CREATE INDEX IF NOT EXISTS idx_backtest_runs_lineage
    ON backtest_runs(workspace_id, strategy_entity_id, completed_at DESC)
    WHERE status = 'completed';
