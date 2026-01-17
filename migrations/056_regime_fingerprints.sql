-- Migration: 056_regime_fingerprints
-- Materialized regime fingerprints for instant similarity queries
--
-- Problem: Regime similarity queries recompute vectors on every request
-- Solution: Precompute and store regime hashes + vectors for fast lookup

-- =============================================================================
-- Table: Materialized regime fingerprints
-- =============================================================================

CREATE TABLE IF NOT EXISTS regime_fingerprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to tune run (the trial that produced this regime)
    tune_id UUID NOT NULL REFERENCES backtest_tunes(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,

    -- 32-byte hash for exact regime matching
    -- SHA256 of canonical regime vector representation
    fingerprint BYTEA NOT NULL,

    -- Raw regime vector for similarity computation
    -- [atr_norm, rsi, bb_width, efficiency, trend_strength, zscore]
    regime_vector FLOAT8[] NOT NULL,

    -- Denormalized regime tags for SQL filtering
    trend_tag TEXT,
    vol_tag TEXT,
    efficiency_tag TEXT,

    -- Schema version for compatibility
    regime_schema_version TEXT NOT NULL DEFAULT 'regime_v1_1',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint: one fingerprint per run
    CONSTRAINT uq_regime_fingerprint_run UNIQUE (run_id)
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Hash index for O(1) exact fingerprint lookup
-- "Find all trials with exactly this regime"
CREATE INDEX IF NOT EXISTS idx_regime_fingerprints_hash
    ON regime_fingerprints USING hash(fingerprint);

-- B-tree index for tune-based lookups
CREATE INDEX IF NOT EXISTS idx_regime_fingerprints_tune
    ON regime_fingerprints(tune_id);

-- Tag-based filtering (same as backtest_tunes)
CREATE INDEX IF NOT EXISTS idx_regime_fingerprints_tags
    ON regime_fingerprints(trend_tag, vol_tag, efficiency_tag)
    WHERE trend_tag IS NOT NULL;

-- GIN index on regime_vector for array operators (optional, for advanced queries)
-- Example: WHERE regime_vector && ARRAY[0.5, 0.6] (overlap)
CREATE INDEX IF NOT EXISTS idx_regime_fingerprints_vector_gin
    ON regime_fingerprints USING gin(regime_vector);

-- =============================================================================
-- Helper function: Compute fingerprint from vector
-- =============================================================================

CREATE OR REPLACE FUNCTION compute_regime_fingerprint(regime_vec FLOAT8[])
RETURNS BYTEA AS $$
DECLARE
    canonical_text TEXT;
BEGIN
    -- Canonical representation: round to 4 decimal places, join with pipes
    -- Example: "0.0145|45.2300|0.0234|0.6543|0.7821|-0.5234"
    SELECT string_agg(ROUND(v::numeric, 4)::text, '|')
    INTO canonical_text
    FROM unnest(regime_vec) AS v;

    -- Return SHA256 hash as BYTEA
    RETURN digest(canonical_text, 'sha256');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- Helper function: Euclidean distance between regime vectors
-- =============================================================================

CREATE OR REPLACE FUNCTION regime_distance(vec1 FLOAT8[], vec2 FLOAT8[])
RETURNS FLOAT8 AS $$
DECLARE
    dist FLOAT8 := 0;
    i INT;
BEGIN
    IF array_length(vec1, 1) != array_length(vec2, 1) THEN
        RETURN NULL;
    END IF;

    FOR i IN 1..array_length(vec1, 1) LOOP
        dist := dist + power(vec1[i] - vec2[i], 2);
    END LOOP;

    RETURN sqrt(dist);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE regime_fingerprints IS 'Materialized regime fingerprints for instant similarity queries. Links tune runs to their precomputed regime vectors.';
COMMENT ON COLUMN regime_fingerprints.fingerprint IS 'SHA256 hash of canonical regime vector (32 bytes). Use hash index for O(1) exact matching.';
COMMENT ON COLUMN regime_fingerprints.regime_vector IS 'Raw numeric regime features: [atr_norm, rsi, bb_width, efficiency, trend_strength, zscore]. Use for similarity computation.';
COMMENT ON FUNCTION compute_regime_fingerprint(FLOAT8[]) IS 'Compute 32-byte SHA256 fingerprint from regime vector. Rounds to 4 decimals for stability.';
COMMENT ON FUNCTION regime_distance(FLOAT8[], FLOAT8[]) IS 'Euclidean distance between two regime vectors. Returns NULL if dimensions differ.';
