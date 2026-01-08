-- Migration 016: Add gates policy snapshot to tunes
-- Stores the gate configuration used during tuning for audit/reproducibility

BEGIN;

ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS gates JSONB;

COMMENT ON COLUMN backtest_tunes.gates IS
  'Gate policy snapshot applied during tuning (e.g., max_dd_pct, min_trades).';

COMMIT;
