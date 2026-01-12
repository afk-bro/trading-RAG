-- migrations/040_trade_events_retention.sql
-- Add retention support columns to trade_events

-- Add severity column with default
ALTER TABLE trade_events
    ADD COLUMN IF NOT EXISTS severity TEXT NOT NULL DEFAULT 'info';

-- Add check constraint for severity values
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'trade_events_severity_check'
    ) THEN
        ALTER TABLE trade_events
            ADD CONSTRAINT trade_events_severity_check
            CHECK (severity IN ('debug', 'info', 'warn', 'error'));
    END IF;
END $$;

-- Add pinned column
ALTER TABLE trade_events
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;

-- Index for retention queries
CREATE INDEX IF NOT EXISTS idx_trade_events_retention
    ON trade_events(created_at, severity, pinned);
