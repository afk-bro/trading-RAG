-- Migration: 024_run_event_types
-- Add run_failed and run_cancelled to trade_events event_type check

ALTER TABLE trade_events DROP CONSTRAINT IF EXISTS trade_events_event_type_check;

ALTER TABLE trade_events ADD CONSTRAINT trade_events_event_type_check
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
    -- Run plan events
    'run_started',
    'run_completed',
    'run_failed',
    'run_cancelled'
));
