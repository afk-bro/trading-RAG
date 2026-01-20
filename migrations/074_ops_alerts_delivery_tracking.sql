-- Migration: 074_ops_alerts_delivery_tracking
-- Purpose: Add delivery tracking columns for Telegram notifications
-- Enables idempotent notification delivery with per-type timestamps

-- ===========================================
-- Delivery tracking columns
-- ===========================================

ALTER TABLE ops_alerts
    ADD COLUMN IF NOT EXISTS notified_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recovery_notified_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS escalation_notified_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS telegram_message_id TEXT,
    ADD COLUMN IF NOT EXISTS delivery_attempts INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_delivery_error TEXT;

COMMENT ON COLUMN ops_alerts.notified_at IS 'When activation notification was sent';
COMMENT ON COLUMN ops_alerts.recovery_notified_at IS 'When recovery notification was sent';
COMMENT ON COLUMN ops_alerts.escalated_at IS 'When severity was increased (set on escalation, cleared on de-escalation)';
COMMENT ON COLUMN ops_alerts.escalation_notified_at IS 'When escalation notification was sent (cleared on new escalation)';
COMMENT ON COLUMN ops_alerts.telegram_message_id IS 'Last Telegram message ID (for reference/threading)';
COMMENT ON COLUMN ops_alerts.delivery_attempts IS 'Total failed delivery attempts across all notification types';
COMMENT ON COLUMN ops_alerts.last_delivery_error IS 'Last delivery error message for debugging';

-- ===========================================
-- Index for pending notifications query
-- ===========================================

-- Partial index for unnotified active alerts (activations)
CREATE INDEX IF NOT EXISTS idx_ops_alerts_pending_activation
    ON ops_alerts (workspace_id, created_at DESC)
    WHERE status = 'active' AND notified_at IS NULL;

-- Partial index for unnotified resolved alerts (recoveries)
CREATE INDEX IF NOT EXISTS idx_ops_alerts_pending_recovery
    ON ops_alerts (workspace_id, resolved_at DESC)
    WHERE status = 'resolved' AND recovery_notified_at IS NULL;

-- Partial index for unnotified escalations
CREATE INDEX IF NOT EXISTS idx_ops_alerts_pending_escalation
    ON ops_alerts (workspace_id, escalated_at DESC)
    WHERE escalated_at IS NOT NULL
      AND escalation_notified_at IS NULL
      AND notified_at IS NOT NULL;

-- ===========================================
-- Verification queries (run manually)
-- ===========================================
-- Check columns exist:
-- \d ops_alerts
--
-- Check indexes:
-- \di idx_ops_alerts_pending_*
