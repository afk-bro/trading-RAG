-- migrations/062_ohlcv_candles.sql
-- OHLCV candle storage for market data

CREATE TABLE IF NOT EXISTS ohlcv_candles (
    exchange_id    TEXT NOT NULL,
    symbol         TEXT NOT NULL,           -- canonical: 'BTC-USDT'
    timeframe      TEXT NOT NULL CHECK (timeframe IN ('1m','5m','15m','1h','1d')),
    ts             TIMESTAMPTZ NOT NULL,    -- candle close, aligned to TF boundary, UTC
    open           DOUBLE PRECISION NOT NULL,
    high           DOUBLE PRECISION NOT NULL CHECK (high >= GREATEST(open, close, low)),
    low            DOUBLE PRECISION NOT NULL CHECK (low <= LEAST(open, close, high)),
    close          DOUBLE PRECISION NOT NULL,
    volume         DOUBLE PRECISION NOT NULL CHECK (volume >= 0),
    PRIMARY KEY (exchange_id, symbol, timeframe, ts)
);

-- Efficient range queries
CREATE INDEX IF NOT EXISTS idx_ohlcv_range
    ON ohlcv_candles (exchange_id, symbol, timeframe, ts DESC);

-- Comment for documentation
COMMENT ON TABLE ohlcv_candles IS 'OHLCV market data storage for backtesting automation';
COMMENT ON COLUMN ohlcv_candles.ts IS 'Candle close timestamp, UTC, aligned to timeframe boundary';
