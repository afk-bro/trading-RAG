-- Migration 020: Trade Events Journal
-- Append-only audit trail for trade intents, policy decisions, and execution events.

-- trade_events: Immutable event log for all trading decisions
CREATE TABLE IF NOT EXISTS trade_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id TEXT NOT NULL,
    workspace_id UUID NOT NULL,

    -- Event type and timing
    event_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Context
    strategy_entity_id UUID,
    symbol TEXT,
    timeframe TEXT,

    -- References
    intent_id UUID,
    order_id TEXT,
    position_id TEXT,

    -- Payload (event-specific JSON)
    payload JSONB NOT NULL DEFAULT '{}',

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Index for listing events by workspace (most common query)
CREATE INDEX IF NOT EXISTS idx_trade_events_workspace_created
    ON trade_events (workspace_id, created_at DESC);

-- Index for finding events by correlation_id (trace linked events)
CREATE INDEX IF NOT EXISTS idx_trade_events_correlation
    ON trade_events (correlation_id, created_at);

-- Index for filtering by event type
CREATE INDEX IF NOT EXISTS idx_trade_events_type_created
    ON trade_events (event_type, created_at DESC);

-- Index for strategy-specific events
CREATE INDEX IF NOT EXISTS idx_trade_events_strategy
    ON trade_events (strategy_entity_id, created_at DESC)
    WHERE strategy_entity_id IS NOT NULL;

-- Index for symbol-specific events
CREATE INDEX IF NOT EXISTS idx_trade_events_symbol
    ON trade_events (symbol, created_at DESC)
    WHERE symbol IS NOT NULL;

-- Index for finding events by intent_id
CREATE INDEX IF NOT EXISTS idx_trade_events_intent
    ON trade_events (intent_id)
    WHERE intent_id IS NOT NULL;

-- Constraint: event_type must be valid
ALTER TABLE trade_events
ADD CONSTRAINT trade_events_event_type_check
CHECK (event_type IN (
    'intent_emitted',
    'intent_validated',
    'intent_invalid',
    'policy_evaluated',
    'intent_approved',
    'intent_rejected',
    'order_submitted',
    'order_filled',
    'order_partial_fill',
    'order_cancelled',
    'order_rejected',
    'position_opened',
    'position_closed',
    'position_scaled',
    'kill_switch_activated',
    'kill_switch_deactivated',
    'regime_drift_detected'
));

-- Comment on table
COMMENT ON TABLE trade_events IS 'Append-only audit trail for trading decisions and executions';
COMMENT ON COLUMN trade_events.correlation_id IS 'Links related events together for tracing';
COMMENT ON COLUMN trade_events.event_type IS 'Type of event (intent_*, policy_*, order_*, position_*, kill_switch_*, regime_*)';
COMMENT ON COLUMN trade_events.payload IS 'Event-specific data (intent details, decision reasons, order fills, etc.)';
