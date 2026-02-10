-- Add updated_at to backtest_run_events for idempotency tracking.
--
-- Lets callers see when events were last (re-)written. Safe to re-run
-- (IF NOT EXISTS + COALESCE handles existing rows).

ALTER TABLE backtest_run_events
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

UPDATE backtest_run_events
    SET updated_at = created_at
    WHERE updated_at IS NULL;
