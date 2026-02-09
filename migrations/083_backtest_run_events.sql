-- Backtest run events for replay.
--
-- Stores structured events emitted during a backtest run (e.g. ORB forming,
-- breakout confirmed, entry signal).  Kept as a single JSONB array per run
-- for simplicity; row-per-event can come later if streaming is needed.
--
-- Each event has at minimum:
--   { "type": "orb_range_locked", "bar_index": 30, "ts": "...", ... }

CREATE TABLE IF NOT EXISTS backtest_run_events (
    run_id        UUID PRIMARY KEY REFERENCES backtest_runs(id) ON DELETE CASCADE,
    workspace_id  UUID NOT NULL REFERENCES workspaces(id),
    events        JSONB NOT NULL DEFAULT '[]'::jsonb,
    event_count   INT GENERATED ALWAYS AS (jsonb_array_length(events)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_run_events_ws
    ON backtest_run_events(workspace_id);

COMMENT ON TABLE backtest_run_events IS
    'Structured events emitted during a backtest run, used for replay and coaching.';
COMMENT ON COLUMN backtest_run_events.events IS
    'JSONB array of event objects. Each has at least type, bar_index, ts fields.';
