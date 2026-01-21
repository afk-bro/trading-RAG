-- Strategy Lifecycle v0.5: link backtests to strategy versions
-- Migration: 078_backtest_version_linkage.sql

-- Optional FK: link backtest runs to specific strategy version
ALTER TABLE backtest_runs
ADD COLUMN IF NOT EXISTS strategy_version_id UUID REFERENCES strategy_versions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_backtest_runs_version
    ON backtest_runs(strategy_version_id) WHERE strategy_version_id IS NOT NULL;

COMMENT ON COLUMN backtest_runs.strategy_version_id IS 'Optional link to strategy version that was tested';

-- Optional FK: link tune sessions to specific strategy version
ALTER TABLE backtest_tunes
ADD COLUMN IF NOT EXISTS strategy_version_id UUID REFERENCES strategy_versions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_backtest_tunes_version
    ON backtest_tunes(strategy_version_id) WHERE strategy_version_id IS NOT NULL;

COMMENT ON COLUMN backtest_tunes.strategy_version_id IS 'Optional link to strategy version that was tuned';

-- NOTE: wfo_runs linkage deferred until table is created (migration 070 not yet applied)
