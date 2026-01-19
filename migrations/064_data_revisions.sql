-- migrations/064_data_revisions.sql
-- Data revision tracking for drift detection

CREATE TABLE IF NOT EXISTS data_revisions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange_id  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    start_ts     TIMESTAMPTZ NOT NULL,
    end_ts       TIMESTAMPTZ NOT NULL,
    row_count    INT NOT NULL,
    checksum     TEXT NOT NULL,            -- deterministic sample hash
    computed_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (exchange_id, symbol, timeframe, start_ts, end_ts)
);

CREATE INDEX IF NOT EXISTS idx_data_revisions_lookup
    ON data_revisions (exchange_id, symbol, timeframe, computed_at DESC);

COMMENT ON TABLE data_revisions IS 'Checksums for detecting data drift in OHLCV ranges';
COMMENT ON COLUMN data_revisions.checksum IS 'SHA256 truncated to 16 chars of sampled candles';
