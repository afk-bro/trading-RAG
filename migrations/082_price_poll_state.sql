-- Migration: 082_price_poll_state
-- Purpose: Scheduling state for live price polling per (exchange, symbol, timeframe)
-- Enables scheduled fetches with exponential backoff on failures

-- ===========================================
-- Create price_poll_state table
-- ===========================================

CREATE TABLE IF NOT EXISTS price_poll_state (
    exchange_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    next_poll_at TIMESTAMPTZ,
    failure_count INT NOT NULL DEFAULT 0,
    last_success_at TIMESTAMPTZ,
    last_candle_ts TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (exchange_id, symbol, timeframe)
);

-- Index for efficient due pair selection (NULLS FIRST = new pairs are due immediately)
CREATE INDEX IF NOT EXISTS idx_price_poll_state_due
    ON price_poll_state (next_poll_at NULLS FIRST)
    WHERE next_poll_at IS NULL OR next_poll_at <= NOW();

-- Index for staleness queries (exclude never-polled pairs)
CREATE INDEX IF NOT EXISTS idx_price_poll_state_staleness
    ON price_poll_state (last_candle_ts)
    WHERE last_candle_ts IS NOT NULL;

-- Comments
COMMENT ON TABLE price_poll_state IS 'Scheduling state for live price polling per (exchange, symbol, timeframe)';
COMMENT ON COLUMN price_poll_state.exchange_id IS 'CCXT exchange identifier (e.g., binance, kraken)';
COMMENT ON COLUMN price_poll_state.symbol IS 'Canonical symbol (e.g., BTC/USDT)';
COMMENT ON COLUMN price_poll_state.timeframe IS 'Candle timeframe (e.g., 1m, 5m, 1h)';
COMMENT ON COLUMN price_poll_state.next_poll_at IS 'Scheduled time for next poll (NULL = poll immediately)';
COMMENT ON COLUMN price_poll_state.failure_count IS 'Consecutive failures for exponential backoff';
COMMENT ON COLUMN price_poll_state.last_success_at IS 'Timestamp of last successful fetch';
COMMENT ON COLUMN price_poll_state.last_candle_ts IS 'Timestamp of most recent candle fetched';
COMMENT ON COLUMN price_poll_state.last_error IS 'Error message from last failed fetch';
