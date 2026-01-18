-- migrations/069_artifact_index.sql
-- Artifact tracking for tune/WFO outputs

CREATE TABLE IF NOT EXISTS artifact_index (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID NOT NULL,
    job_id          UUID REFERENCES jobs(id) ON DELETE SET NULL,
    run_id          UUID NOT NULL,  -- tune_id or wfo_id
    job_type        TEXT NOT NULL CHECK (job_type IN ('tune', 'wfo')),
    artifact_kind   TEXT NOT NULL CHECK (artifact_kind IN ('tune_json', 'trials_csv', 'equity_csv', 'wfo_json', 'candidates_csv')),
    artifact_path   TEXT NOT NULL,
    file_size_bytes BIGINT,
    data_revision   JSONB,
    is_pinned       BOOLEAN DEFAULT false,
    pinned_at       TIMESTAMPTZ,
    pinned_by       TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Index for workspace-scoped queries
CREATE INDEX IF NOT EXISTS idx_artifact_workspace
    ON artifact_index (workspace_id, job_type, created_at DESC);

-- Index for run lookups
CREATE INDEX IF NOT EXISTS idx_artifact_run
    ON artifact_index (run_id);

-- Unique constraint to prevent duplicate artifacts
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_unique
    ON artifact_index (run_id, artifact_kind);

COMMENT ON TABLE artifact_index IS 'Index of generated artifacts from tune/WFO jobs';
COMMENT ON COLUMN artifact_index.run_id IS 'Reference to tune_id or wfo_id';
COMMENT ON COLUMN artifact_index.artifact_kind IS 'Type: tune_json, trials_csv, equity_csv, wfo_json, candidates_csv';
COMMENT ON COLUMN artifact_index.artifact_path IS 'Relative path under data/artifacts/';
COMMENT ON COLUMN artifact_index.is_pinned IS 'Pinned artifacts are retained indefinitely';
