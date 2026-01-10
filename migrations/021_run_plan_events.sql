-- Migration 021: Run Plan Events
-- Adds RUN_STARTED and RUN_COMPLETED event types for the Test Generator / Run Orchestrator.

-- Drop and recreate the event_type constraint to include new types
ALTER TABLE trade_events
DROP CONSTRAINT IF EXISTS trade_events_event_type_check;

ALTER TABLE trade_events
ADD CONSTRAINT trade_events_event_type_check
CHECK (event_type IN (
    -- Intent lifecycle
    'intent_emitted',
    'intent_validated',
    'intent_invalid',
    -- Policy evaluation
    'policy_evaluated',
    'intent_approved',
    'intent_rejected',
    -- Execution
    'order_submitted',
    'order_filled',
    'order_partial_fill',
    'order_cancelled',
    'order_rejected',
    -- Position changes
    'position_opened',
    'position_closed',
    'position_scaled',
    -- System events
    'kill_switch_activated',
    'kill_switch_deactivated',
    'regime_drift_detected',
    -- Run plan events (new)
    'run_started',
    'run_completed'
));

-- Add comments for new event types
COMMENT ON COLUMN trade_events.event_type IS 'Type of event (intent_*, policy_*, order_*, position_*, kill_switch_*, regime_*, run_*)';
