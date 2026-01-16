-- Add pine_metadata JSONB column to documents table
-- Stores structured metadata for Pine scripts (script_type, inputs, features, lint)
-- Populated during Pine script ingestion for efficient querying

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS pine_metadata JSONB;

COMMENT ON COLUMN documents.pine_metadata IS
    'Structured metadata for Pine scripts: schema_version, script_type, pine_version, inputs, imports, features, lint_summary, lint_findings';
