-- migrations/072_ops_alerts_occurrence_count.sql
-- Add occurrence_count to track how many times an alert has been triggered

ALTER TABLE ops_alerts ADD COLUMN occurrence_count INTEGER NOT NULL DEFAULT 1;

COMMENT ON COLUMN ops_alerts.occurrence_count IS 'Number of times this alert has been triggered (incremented on each upsert)';
