-- migrations/063_core_symbols.sql
-- Core symbols and symbol request tracking

CREATE TABLE IF NOT EXISTS core_symbols (
    exchange_id      TEXT NOT NULL,
    canonical_symbol TEXT NOT NULL,
    raw_symbol       TEXT NOT NULL,
    timeframes       TEXT[] DEFAULT ARRAY['1m','5m','15m','1h','1d'],
    is_enabled       BOOLEAN DEFAULT true,
    added_at         TIMESTAMPTZ DEFAULT now(),
    added_by         TEXT,
    UNIQUE (exchange_id, canonical_symbol)
);

CREATE INDEX IF NOT EXISTS idx_core_symbols_enabled
    ON core_symbols (exchange_id) WHERE is_enabled = true;

-- Write-only log for future auto-promote feature
CREATE TABLE IF NOT EXISTS symbol_requests (
    id               BIGSERIAL PRIMARY KEY,
    exchange_id      TEXT NOT NULL,
    canonical_symbol TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    requested_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_symbol_requests_symbol
    ON symbol_requests (exchange_id, canonical_symbol, requested_at DESC);

COMMENT ON TABLE core_symbols IS 'Universe of symbols to keep warm via scheduled sync';
COMMENT ON TABLE symbol_requests IS 'Log of ad-hoc symbol requests for future auto-promote';
