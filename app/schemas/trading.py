"""Trading schemas: intents, policies, events, execution, strategies."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ===========================================
# Strategy Spec Schemas
# ===========================================


class StrategySpecStatus(str, Enum):
    """Status of a strategy specification."""

    DRAFT = "draft"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


class StrategySpecResponse(BaseModel):
    """Response for GET /kb/strategies/{entity_id}/spec."""

    id: UUID = Field(..., description="Spec ID")
    strategy_entity_id: UUID = Field(..., description="Strategy entity ID")
    strategy_name: str = Field(..., description="Strategy name")
    spec_json: dict = Field(..., description="The compiled specification")
    status: StrategySpecStatus = Field(..., description="Approval status")
    version: int = Field(..., description="Spec version number")
    derived_from_claim_ids: list[str] = Field(
        default_factory=list, description="Source claim IDs"
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    approved_by: Optional[str] = Field(None, description="Approver identifier")


class StrategySpecRefreshRequest(BaseModel):
    """Request for POST /kb/strategies/{entity_id}/spec/refresh."""

    pass  # No body needed, entity_id is in path


class StrategyCompileResponse(BaseModel):
    """Response for POST /kb/strategies/{entity_id}/compile."""

    spec_id: str = Field(..., description="Source spec ID")
    spec_version: int = Field(..., description="Spec version used")
    spec_status: StrategySpecStatus = Field(..., description="Spec approval status")
    param_schema: dict = Field(..., description="JSON Schema for parameter UI form")
    backtest_config: dict = Field(
        ..., description="Engine-agnostic backtest configuration"
    )
    pseudocode: str = Field(..., description="Human-readable strategy description")
    citations: list[str] = Field(..., description="Claim IDs used to derive the spec")


class StrategySpecStatusUpdate(BaseModel):
    """Request for PATCH /kb/strategies/{entity_id}/spec."""

    status: StrategySpecStatus = Field(..., description="New status")
    approved_by: Optional[str] = Field(
        None, description="Approver identifier (for approved status)"
    )


# ===========================================
# Trade Intent & Policy Engine Schemas
# ===========================================


class IntentAction(str, Enum):
    """What the brain wants to do."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    CANCEL_ORDER = "cancel_order"


class TradeIntent(BaseModel):
    """
    A declaration of what the trading brain wants to do.

    Provider-agnostic: this is what the strategy wants, not how
    to execute it. The Policy Engine decides if it's allowed.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description="Intent UUID")
    correlation_id: str = Field(..., description="Correlation ID for tracing")
    workspace_id: UUID = Field(..., description="Workspace this intent belongs to")

    # What
    action: IntentAction = Field(..., description="Requested action")
    strategy_entity_id: UUID = Field(..., description="Strategy making the request")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 4h)")

    # Parameters (optional, depends on action)
    quantity: Optional[float] = Field(None, ge=0, description="Position size")
    price: Optional[float] = Field(None, description="Limit price (None for market)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")

    # Context
    signal_strength: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Signal confidence [0,1]"
    )
    regime_snapshot: Optional[dict] = Field(None, description="Current regime state")
    reason: Optional[str] = Field(None, description="Human-readable reason for intent")

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When intent was created"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class PolicyReason(str, Enum):
    """Why a policy decision was made."""

    # Rejections
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    REGIME_DRIFT = "regime_drift"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    COOLDOWN_ACTIVE = "cooldown_active"
    INVALID_SYMBOL = "invalid_symbol"
    INVALID_TIMEFRAME = "invalid_timeframe"
    STRATEGY_DISABLED = "strategy_disabled"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"

    # Approvals
    ALL_RULES_PASSED = "all_rules_passed"
    MANUAL_OVERRIDE = "manual_override"


class PolicyDecision(BaseModel):
    """
    The Policy Engine's verdict on a TradeIntent.

    This is the gatekeeper's output: approved, rejected, or held.
    """

    # Decision
    approved: bool = Field(..., description="Whether intent is approved for execution")
    reason: PolicyReason = Field(..., description="Primary reason for decision")
    reason_details: Optional[str] = Field(None, description="Additional details")

    # Audit trail
    rules_evaluated: list[str] = Field(
        default_factory=list, description="Rules that were evaluated"
    )
    rules_passed: list[str] = Field(
        default_factory=list, description="Rules that passed"
    )
    rules_failed: list[str] = Field(
        default_factory=list, description="Rules that failed"
    )

    # Modifications (for partial approvals)
    modified_quantity: Optional[float] = Field(
        None, description="Adjusted quantity if capped"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-blocking warnings"
    )

    # Context
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    evaluation_ms: Optional[int] = Field(None, description="Evaluation time in ms")


class PositionState(BaseModel):
    """Current position for a symbol."""

    symbol: str = Field(..., description="Trading symbol")
    side: Optional[str] = Field(None, description="'long', 'short', or None if flat")
    quantity: float = Field(default=0.0, description="Position size")
    entry_price: Optional[float] = Field(None, description="Average entry price")
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized P&L")
    realized_pnl_today: Optional[float] = Field(None, description="Realized P&L today")


class CurrentState(BaseModel):
    """
    Minimal current state snapshot for policy evaluation.

    This is what the Policy Engine needs to make decisions.
    Kept minimal to avoid stale state issues.
    """

    # System state
    kill_switch_active: bool = Field(default=False, description="Global kill switch")
    trading_enabled: bool = Field(default=True, description="Trading allowed")

    # Positions (optional - may not be available)
    positions: list[PositionState] = Field(
        default_factory=list, description="Current positions"
    )

    # Account metrics (optional)
    account_equity: Optional[float] = Field(None, description="Current account equity")
    daily_pnl: Optional[float] = Field(None, description="Today's realized P&L")
    max_drawdown_today: Optional[float] = Field(
        None, description="Today's max drawdown %"
    )

    # Regime (from v1.5)
    current_regime: Optional[dict] = Field(None, description="Current regime snapshot")
    regime_distance_z: Optional[float] = Field(
        None, description="Z-score from training regime"
    )

    # Timestamps
    snapshot_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# Trade Event Journal Schemas
# ===========================================


class TradeEventType(str, Enum):
    """Types of events recorded in the trade journal."""

    # Intent lifecycle
    INTENT_EMITTED = "intent_emitted"
    INTENT_VALIDATED = "intent_validated"
    INTENT_INVALID = "intent_invalid"

    # Policy evaluation
    POLICY_EVALUATED = "policy_evaluated"
    INTENT_APPROVED = "intent_approved"
    INTENT_REJECTED = "intent_rejected"

    # Execution
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Position changes
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_SCALED = "position_scaled"

    # System events
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"
    REGIME_DRIFT_DETECTED = "regime_drift_detected"

    # Run plan events (Test Generator / Orchestrator)
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # Job runner events
    JOB_FAILED = "job_failed"


class TradeEvent(BaseModel):
    """
    Immutable event record for the trade journal.

    Append-only audit trail of all trading decisions.
    """

    id: UUID = Field(default_factory=uuid4)
    correlation_id: str = Field(..., description="Links related events together")
    workspace_id: UUID = Field(..., description="Workspace this event belongs to")

    # Event type and timing
    event_type: TradeEventType = Field(..., description="Type of event")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Context
    strategy_entity_id: Optional[UUID] = Field(
        None, description="Strategy that triggered event"
    )
    symbol: Optional[str] = Field(None, description="Trading symbol if applicable")
    timeframe: Optional[str] = Field(None, description="Timeframe if applicable")

    # References
    intent_id: Optional[UUID] = Field(None, description="Related intent ID")
    order_id: Optional[str] = Field(None, description="External order ID")
    position_id: Optional[str] = Field(None, description="External position ID")

    # Payload (event-specific data)
    payload: dict = Field(default_factory=dict, description="Event-specific data")

    # Metadata
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class TradeEventListResponse(BaseModel):
    """Response for GET /admin/trade/events."""

    items: list[TradeEvent] = Field(..., description="Event list")
    total: int = Field(..., description="Total matching events")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class IntentEvaluateRequest(BaseModel):
    """Request for POST /intents/evaluate."""

    intent: TradeIntent = Field(..., description="Intent to evaluate")
    state: Optional[CurrentState] = Field(
        None, description="Current state (uses defaults if not provided)"
    )
    dry_run: bool = Field(default=False, description="If true, don't journal the event")


class IntentEvaluateResponse(BaseModel):
    """Response for POST /intents/evaluate."""

    intent_id: UUID = Field(..., description="Intent that was evaluated")
    decision: PolicyDecision = Field(..., description="Policy engine decision")
    events_recorded: int = Field(default=0, description="Number of events journaled")
    correlation_id: str = Field(..., description="Correlation ID for tracing")


# ===========================================
# Paper Execution Schemas
# ===========================================


class OrderSide(str, Enum):
    """Order side for execution."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExecutionMode(str, Enum):
    """Execution mode."""

    PAPER = "paper"
    LIVE = "live"  # Future


class PaperOrder(BaseModel):
    """Simulated order for paper trading."""

    id: UUID = Field(default_factory=uuid4)
    intent_id: UUID = Field(..., description="Intent that triggered this order")
    correlation_id: str = Field(..., description="Correlation ID for tracing")
    workspace_id: UUID = Field(..., description="Workspace this order belongs to")

    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., gt=0, description="Order quantity")
    fill_price: float = Field(..., gt=0, description="Execution price")

    status: OrderStatus = Field(default=OrderStatus.FILLED, description="Order status")
    fees: float = Field(default=0.0, ge=0, description="Execution fees")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = Field(
        default=None, description="When order was filled"
    )

    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class PaperPosition(BaseModel):
    """Paper trading position state."""

    workspace_id: UUID = Field(..., description="Workspace this position belongs to")
    symbol: str = Field(..., description="Trading symbol")
    side: Optional[str] = Field(
        None, description="Position side ('long' or None if flat)"
    )
    quantity: float = Field(default=0.0, ge=0, description="Position size")
    avg_price: float = Field(default=0.0, ge=0, description="Average entry price")

    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    realized_pnl: float = Field(
        default=0.0, description="Realized P&L from this position"
    )

    opened_at: Optional[datetime] = Field(None, description="When position was opened")
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Tracking
    order_ids: list[str] = Field(
        default_factory=list, description="Orders that built this position"
    )
    intent_ids: list[str] = Field(
        default_factory=list, description="Intents that triggered orders"
    )


class PaperState(BaseModel):
    """Complete paper trading state for a workspace."""

    workspace_id: UUID = Field(..., description="Workspace this state belongs to")

    # Cash ledger
    starting_equity: float = Field(default=10000.0, description="Starting equity")
    cash: float = Field(default=10000.0, description="Current cash balance")
    realized_pnl: float = Field(default=0.0, description="Total realized P&L")

    # Positions by symbol
    positions: dict[str, PaperPosition] = Field(
        default_factory=dict, description="Positions keyed by symbol"
    )

    # Tracking
    orders_count: int = Field(default=0, description="Total orders executed")
    trades_count: int = Field(default=0, description="Total trades (round trips)")

    # Reconciliation
    last_event_id: Optional[UUID] = Field(None, description="Last processed event ID")
    last_event_at: Optional[datetime] = Field(None, description="Last event timestamp")
    reconciled_at: Optional[datetime] = Field(
        None, description="When state was reconciled"
    )


class ExecutionRequest(BaseModel):
    """Request for POST /execute/intents."""

    intent: TradeIntent = Field(..., description="Intent to execute")
    fill_price: float = Field(..., gt=0, description="Fill price (required)")
    mode: ExecutionMode = Field(
        default=ExecutionMode.PAPER, description="Execution mode"
    )


class ExecutionResult(BaseModel):
    """Result of intent execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    intent_id: UUID = Field(..., description="Intent that was executed")

    order_id: Optional[UUID] = Field(None, description="Order ID if created")
    fill_price: Optional[float] = Field(None, description="Actual fill price")
    quantity_filled: float = Field(default=0.0, description="Quantity filled")
    fees: float = Field(default=0.0, description="Fees charged")

    position_action: Optional[str] = Field(
        None, description="Position action: opened, closed, scaled"
    )
    position: Optional[PaperPosition] = Field(None, description="Updated position")

    events_recorded: int = Field(default=0, description="Events journaled")
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for tracing"
    )

    # Error info
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")


class ReconciliationResult(BaseModel):
    """Result of journal reconciliation."""

    success: bool = Field(..., description="Whether reconciliation succeeded")
    workspace_id: UUID = Field(..., description="Workspace reconciled")

    events_replayed: int = Field(default=0, description="Events processed")
    orders_rebuilt: int = Field(default=0, description="Orders reconstructed")
    positions_rebuilt: int = Field(default=0, description="Positions with qty > 0")

    cash_after: float = Field(default=0.0, description="Cash after reconciliation")
    realized_pnl_after: float = Field(default=0.0, description="Realized P&L after")

    last_event_at: Optional[datetime] = Field(None, description="Last event timestamp")
    errors: list[str] = Field(default_factory=list, description="Errors encountered")


# =============================================================================
# Strategy Registry
# =============================================================================


class StrategyEngine(str, Enum):
    """Supported execution engines for strategies."""

    PINE = "pine"
    PYTHON = "python"
    VECTORBT = "vectorbt"
    BACKTESTING_PY = "backtesting_py"


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class StrategyReviewStatus(str, Enum):
    """Strategy human review status."""

    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class StrategyRiskLevel(str, Enum):
    """Strategy risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BacktestSummaryStatus(str, Enum):
    """Backtest summary status."""

    NEVER = "never"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class StrategyTags(BaseModel):
    """Tags mirroring MatchIntent for coverage overlap computation."""

    strategy_archetypes: list[str] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    timeframe_buckets: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    risk_terms: list[str] = Field(default_factory=list)


class BacktestSummary(BaseModel):
    """Backtest summary stored as JSONB on strategies."""

    status: BacktestSummaryStatus = Field(
        default=BacktestSummaryStatus.NEVER, description="Backtest status"
    )
    last_backtest_at: Optional[datetime] = Field(None, description="Last run timestamp")
    best_oos_score: Optional[float] = Field(None, description="Best OOS score")
    max_drawdown: Optional[float] = Field(None, description="Max drawdown percentage")
    num_trades: Optional[int] = Field(None, description="Number of trades")
    dataset_coverage: Optional[dict] = Field(
        None, description="Symbols and time ranges tested"
    )
    rigor: Optional[dict] = Field(
        None, description="Fees, slippage, walk-forward settings"
    )
    notes: Optional[str] = Field(None, description="Human notes")


class StrategySourceRef(BaseModel):
    """Engine-specific source reference."""

    # Common
    store: Optional[str] = Field(None, description="local, github, tradingview")
    path: Optional[str] = Field(None, description="File path or script path")
    doc_id: Optional[UUID] = Field(None, description="Reference to documents table")

    # Pine-specific
    repo: Optional[str] = Field(None, description="GitHub repo (org/repo)")
    ref: Optional[str] = Field(None, description="Git ref (branch/tag/sha)")

    # Python-specific
    module: Optional[str] = Field(None, description="Python module path")
    entrypoint: Optional[str] = Field(None, description="Function entrypoint")
    params_schema: Optional[dict] = Field(None, description="JSON Schema for params")


class StrategyCreateRequest(BaseModel):
    """Request for POST /strategies."""

    workspace_id: UUID = Field(..., description="Workspace ID")
    name: str = Field(..., min_length=1, max_length=200, description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    engine: StrategyEngine = Field(
        default=StrategyEngine.PINE, description="Execution engine"
    )
    source_ref: Optional[StrategySourceRef] = Field(
        None, description="Engine-specific source"
    )
    status: StrategyStatus = Field(
        default=StrategyStatus.DRAFT, description="Initial status"
    )
    risk_level: Optional[StrategyRiskLevel] = Field(None, description="Risk level")
    tags: Optional[StrategyTags] = Field(None, description="Intent-compatible tags")


class StrategyUpdateRequest(BaseModel):
    """Request for PATCH /strategies/{id}."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None)
    status: Optional[StrategyStatus] = Field(None)
    review_status: Optional[StrategyReviewStatus] = Field(None)
    risk_level: Optional[StrategyRiskLevel] = Field(None)
    source_ref: Optional[StrategySourceRef] = Field(None)
    tags: Optional[StrategyTags] = Field(None)
    backtest_summary: Optional[BacktestSummary] = Field(None)


class StrategyListItem(BaseModel):
    """Single item in strategy list response."""

    id: UUID = Field(..., description="Strategy ID")
    name: str = Field(..., description="Strategy name")
    slug: str = Field(..., description="URL-safe slug")
    engine: StrategyEngine = Field(..., description="Execution engine")
    status: StrategyStatus = Field(..., description="Lifecycle status")
    review_status: StrategyReviewStatus = Field(..., description="Review status")
    risk_level: Optional[StrategyRiskLevel] = Field(None, description="Risk level")
    tags: StrategyTags = Field(default_factory=StrategyTags, description="Tags")
    backtest_summary: Optional[BacktestSummary] = Field(
        None, description="Latest backtest summary"
    )
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: datetime = Field(..., description="Updated timestamp")


class StrategyListResponse(BaseModel):
    """Response for GET /strategies."""

    items: list[StrategyListItem] = Field(..., description="Strategies")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")
    has_more: bool = Field(..., description="More results available")


class StrategyDetailResponse(BaseModel):
    """Response for GET /strategies/{id}."""

    id: UUID = Field(..., description="Strategy ID")
    workspace_id: UUID = Field(..., description="Workspace ID")
    name: str = Field(..., description="Strategy name")
    slug: str = Field(..., description="URL-safe slug")
    description: Optional[str] = Field(None, description="Description")
    engine: StrategyEngine = Field(..., description="Execution engine")
    source_ref: Optional[StrategySourceRef] = Field(
        None, description="Engine-specific source"
    )
    status: StrategyStatus = Field(..., description="Lifecycle status")
    review_status: StrategyReviewStatus = Field(..., description="Review status")
    risk_level: Optional[StrategyRiskLevel] = Field(None, description="Risk level")
    tags: StrategyTags = Field(default_factory=StrategyTags, description="Tags")
    backtest_summary: Optional[BacktestSummary] = Field(
        None, description="Latest backtest summary"
    )
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: datetime = Field(..., description="Updated timestamp")


class CandidateStrategy(BaseModel):
    """Candidate strategy in coverage response."""

    strategy_id: UUID = Field(..., description="Strategy ID")
    name: str = Field(..., description="Strategy name")
    score: float = Field(..., description="Tag overlap score")
    matched_tags: list[str] = Field(..., description="Tags that matched")


class StrategyCard(BaseModel):
    """Lightweight strategy card for bulk fetches (cockpit UI)."""

    id: UUID = Field(..., description="Strategy ID")
    name: str = Field(..., description="Strategy name")
    slug: str = Field(..., description="URL-safe slug")
    engine: StrategyEngine = Field(..., description="Execution engine")
    status: StrategyStatus = Field(..., description="Lifecycle status")
    tags: StrategyTags = Field(default_factory=StrategyTags, description="Tags")
    backtest_status: Optional[BacktestSummaryStatus] = Field(
        None, description="Backtest status"
    )
    last_backtest_at: Optional[datetime] = Field(
        None, description="Last backtest timestamp"
    )
    best_oos_score: Optional[float] = Field(None, description="Best OOS score")
    max_drawdown: Optional[float] = Field(None, description="Max drawdown percentage")


# =============================================================================
# Strategy Versions (Lifecycle v0.5)
# =============================================================================


class StrategyVersionState(str, Enum):
    """Strategy version lifecycle state."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRED = "retired"


class StrategyVersionCreateRequest(BaseModel):
    """Request for POST /strategies/{id}/versions."""

    config_snapshot: dict = Field(..., description="Immutable strategy configuration")
    version_tag: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        description="Optional version tag (e.g., v1.0-beta)",
    )
    regime_awareness: Optional[dict] = Field(
        default_factory=dict, description="Regime-specific behavior config"
    )
    created_by: Optional[str] = Field(
        None, description="Actor creating version (admin:<token> or system)"
    )


class StrategyVersionResponse(BaseModel):
    """Response for strategy version endpoints."""

    id: UUID = Field(..., description="Version UUID")
    strategy_id: UUID = Field(..., description="Parent strategy UUID")
    strategy_entity_id: UUID = Field(
        ..., description="Legacy entity ID for FK compatibility"
    )
    version_number: int = Field(..., description="Auto-incremented version number")
    version_tag: Optional[str] = Field(None, description="Optional version tag")
    config_snapshot: dict = Field(..., description="Immutable strategy configuration")
    config_hash: str = Field(..., description="SHA256 hash of config for deduplication")
    state: StrategyVersionState = Field(..., description="Current lifecycle state")
    regime_awareness: dict = Field(default_factory=dict, description="Regime config")
    created_at: datetime = Field(..., description="Creation timestamp")
    created_by: Optional[str] = Field(None, description="Actor who created version")
    activated_at: Optional[datetime] = Field(None, description="When activated")
    paused_at: Optional[datetime] = Field(None, description="When paused")
    retired_at: Optional[datetime] = Field(None, description="When retired")
    kb_strategy_spec_id: Optional[UUID] = Field(None, description="Source spec ID")


class StrategyVersionListItem(BaseModel):
    """Single item in version list response."""

    id: UUID = Field(..., description="Version UUID")
    version_number: int = Field(..., description="Version number")
    version_tag: Optional[str] = Field(None, description="Version tag")
    state: StrategyVersionState = Field(..., description="Lifecycle state")
    config_hash: str = Field(..., description="Config hash (first 16 chars)")
    created_at: datetime = Field(..., description="Creation timestamp")
    created_by: Optional[str] = Field(None, description="Creator")
    activated_at: Optional[datetime] = Field(None, description="When activated")


class StrategyVersionListResponse(BaseModel):
    """Response for GET /strategies/{id}/versions."""

    items: list[StrategyVersionListItem] = Field(..., description="Version list")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")
    has_more: bool = Field(..., description="More results available")


class VersionTransitionRequest(BaseModel):
    """Request for state transition endpoints (activate, pause, retire)."""

    reason: Optional[str] = Field(
        None, max_length=500, description="Reason for transition"
    )
    triggered_by: str = Field(
        ..., min_length=1, max_length=100, description="Actor triggering transition"
    )


class VersionTransitionResponse(BaseModel):
    """Response for version transition audit."""

    id: UUID = Field(..., description="Transition record UUID")
    version_id: UUID = Field(..., description="Version UUID")
    from_state: Optional[StrategyVersionState] = Field(
        None, description="Previous state (None for creation)"
    )
    to_state: StrategyVersionState = Field(..., description="New state")
    triggered_by: str = Field(..., description="Actor who triggered")
    triggered_at: datetime = Field(..., description="When triggered")
    reason: Optional[str] = Field(None, description="Reason for transition")


# =============================================================================
# Strategy Intelligence Snapshots (v1.5)
# =============================================================================


class IntelSnapshotCreateRequest(BaseModel):
    """Request for creating an intelligence snapshot."""

    strategy_version_id: UUID = Field(
        ..., description="Strategy version this intel is for"
    )
    as_of_ts: datetime = Field(..., description="Market time the intel refers to")
    regime: str = Field(
        ..., min_length=1, max_length=100, description="Regime classification"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregated confidence [0, 1]"
    )
    confidence_components: Optional[dict] = Field(
        default_factory=dict,
        description="Breakdown of confidence factors (e.g., regime_fit, backtest_oos)",
    )
    features: Optional[dict] = Field(
        default_factory=dict, description="Raw feature values used for computation"
    )
    explain: Optional[dict] = Field(
        default_factory=dict, description="Human-readable explanation"
    )
    engine_version: Optional[str] = Field(
        None, max_length=50, description="Version of computation engine"
    )
    inputs_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA256 of inputs for deduplication",
    )
    run_id: Optional[UUID] = Field(None, description="Link to job/workflow run")


class IntelSnapshotResponse(BaseModel):
    """Response for intel snapshot endpoints."""

    id: UUID = Field(..., description="Snapshot UUID")
    workspace_id: UUID = Field(..., description="Workspace scope")
    strategy_version_id: UUID = Field(..., description="Strategy version UUID")
    as_of_ts: datetime = Field(..., description="Market time the intel refers to")
    computed_at: datetime = Field(..., description="When intel was computed")
    regime: str = Field(..., description="Regime classification")
    confidence_score: float = Field(..., description="Aggregated confidence [0, 1]")
    confidence_components: dict = Field(
        default_factory=dict, description="Confidence breakdown"
    )
    features: dict = Field(default_factory=dict, description="Feature values")
    explain: dict = Field(
        default_factory=dict, description="Human-readable explanation"
    )
    engine_version: Optional[str] = Field(
        None, description="Computation engine version"
    )
    inputs_hash: Optional[str] = Field(None, description="Input hash for deduplication")
    run_id: Optional[UUID] = Field(None, description="Job/run link")


class IntelSnapshotListItem(BaseModel):
    """Single item in intel snapshot list."""

    id: UUID = Field(..., description="Snapshot UUID")
    strategy_version_id: UUID = Field(..., description="Strategy version UUID")
    as_of_ts: datetime = Field(..., description="Market time")
    computed_at: datetime = Field(..., description="Computation time")
    regime: str = Field(..., description="Regime")
    confidence_score: float = Field(..., description="Confidence [0, 1]")


class IntelSnapshotListResponse(BaseModel):
    """Response for GET /strategies/{id}/versions/{vid}/intel."""

    items: list[IntelSnapshotListItem] = Field(..., description="Snapshot list")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page size")
    next_cursor: Optional[datetime] = Field(
        None, description="Cursor for next page (as_of_ts)"
    )
