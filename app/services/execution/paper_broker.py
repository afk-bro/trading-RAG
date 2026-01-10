"""Paper trading broker adapter."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import structlog

from app.config import get_settings
from app.schemas import (
    TradeIntent,
    IntentAction,
    ExecutionResult,
    PaperPosition,
    PaperState,
    PaperOrder,
    OrderSide,
    OrderStatus,
    ReconciliationResult,
    TradeEvent,
    TradeEventType,
    CurrentState,
)
from app.services.execution.base import BrokerAdapter
from app.services.policy_engine import PolicyEngine
from app.repositories.trade_events import TradeEventsRepository, EventFilters


logger = structlog.get_logger(__name__)

# Supported actions in PR1
SUPPORTED_ACTIONS = {IntentAction.OPEN_LONG, IntentAction.CLOSE_LONG}


class PaperBroker(BrokerAdapter):
    """
    Paper trading broker for simulation.

    Features:
    - Deterministic fill simulation (caller provides price)
    - Journal-based state persistence (event sourcing)
    - Reconciliation from ORDER_FILLED events
    - Idempotency on (workspace_id, intent_id, mode)
    - Policy re-evaluation before execution

    Design decisions:
    - fill_price required (no slippage calculation)
    - MARKET orders only (immediate fill)
    - Long-only in PR1 (OPEN_LONG, CLOSE_LONG)
    - Full close only (SELL qty must == position.qty)
    - fees = 0.0 constant (configurable later)
    """

    def __init__(self, events_repo: TradeEventsRepository):
        """
        Initialize paper broker.

        Args:
            events_repo: Trade events repository for journaling
        """
        self._events_repo = events_repo
        self._states: dict[UUID, PaperState] = {}  # workspace_id -> state
        self._settings = get_settings()
        self._policy_engine = PolicyEngine()

    @property
    def mode(self) -> str:
        return "paper"

    async def execute_intent(
        self,
        intent: TradeIntent,
        fill_price: float,
    ) -> ExecutionResult:
        """Execute an intent via paper simulation."""
        log = logger.bind(
            intent_id=str(intent.id),
            correlation_id=intent.correlation_id,
            workspace_id=str(intent.workspace_id),
            action=intent.action.value,
            symbol=intent.symbol,
        )

        # 1. Validate action is supported
        if intent.action not in SUPPORTED_ACTIONS:
            log.warning("unsupported_action", action=intent.action.value)
            return ExecutionResult(
                success=False,
                intent_id=intent.id,
                error=f"Unsupported action: {intent.action.value}. Supported: OPEN_LONG, CLOSE_LONG",
                error_code="UNSUPPORTED_ACTION",
            )

        # 2. Idempotency check: query for existing ORDER_FILLED with this intent_id
        existing = await self._check_idempotency(intent.workspace_id, intent.id)
        if existing:
            log.info("idempotency_hit", prior_correlation_id=existing.correlation_id)
            return ExecutionResult(
                success=False,
                intent_id=intent.id,
                error="Intent already executed",
                error_code="ALREADY_EXECUTED",
                correlation_id=existing.correlation_id,
            )

        # 3. Re-evaluate policy internally
        state = self._get_or_create_state(intent.workspace_id)
        current_state = self._build_current_state(state, intent.workspace_id)
        decision = self._policy_engine.evaluate(intent, current_state)

        if not decision.approved:
            # Journal rejection and return
            rejection_event = TradeEvent(
                correlation_id=intent.correlation_id,
                workspace_id=intent.workspace_id,
                event_type=TradeEventType.INTENT_REJECTED,
                strategy_entity_id=intent.strategy_entity_id,
                symbol=intent.symbol,
                timeframe=intent.timeframe,
                intent_id=intent.id,
                payload={
                    "reason": decision.reason.value,
                    "reason_details": decision.reason_details,
                    "rules_failed": decision.rules_failed,
                    "mode": self.mode,
                },
            )
            await self._events_repo.insert(rejection_event)

            log.info(
                "policy_rejected",
                reason=decision.reason.value,
                rules_failed=decision.rules_failed,
            )
            return ExecutionResult(
                success=False,
                intent_id=intent.id,
                error=f"Policy rejected: {decision.reason.value}",
                error_code="POLICY_REJECTED",
                correlation_id=intent.correlation_id,
                events_recorded=1,
            )

        # 4. Validate fill_price
        if fill_price <= 0:
            return ExecutionResult(
                success=False,
                intent_id=intent.id,
                error="fill_price must be positive",
                error_code="INVALID_FILL_PRICE",
            )

        # 5. Get quantity
        quantity = decision.modified_quantity or intent.quantity or 0.0
        if quantity <= 0:
            return ExecutionResult(
                success=False,
                intent_id=intent.id,
                error="Invalid quantity",
                error_code="INVALID_QUANTITY",
            )

        # 6. Convert action to side
        side = self._action_to_side(intent.action)

        # 7/8. Handle position update with validation
        position_result = self._validate_and_update_position(
            state, intent, side, quantity, fill_price
        )
        if position_result.get("error"):
            return ExecutionResult(
                success=False,
                intent_id=intent.id,
                error=position_result["error"],
                error_code=position_result.get("error_code", "VALIDATION_ERROR"),
            )

        position_action = position_result["action"]
        position = position_result["position"]

        # 9. Generate order_id
        order_id = uuid4()

        # 10. Update cash ledger
        fees = 0.0  # PR1: constant, configurable later
        self._update_cash_ledger(
            state, side, quantity, fill_price, fees, position_result
        )

        # 11. Create order record
        order = PaperOrder(
            id=order_id,
            intent_id=intent.id,
            correlation_id=intent.correlation_id,
            workspace_id=intent.workspace_id,
            symbol=intent.symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            status=OrderStatus.FILLED,
            fees=fees,
            filled_at=datetime.utcnow(),
        )

        # 12. Journal events
        events = self._create_execution_events(intent, order, position_action, position)
        events_recorded = await self._events_repo.insert_many(events)

        # Update state tracking
        state.orders_count += 1
        if position_action == "closed":
            state.trades_count += 1  # Round trip complete
        state.last_event_id = events[-1].id if events else None
        state.last_event_at = datetime.utcnow()

        log.info(
            "paper_execution_complete",
            order_id=str(order_id),
            fill_price=fill_price,
            quantity=quantity,
            position_action=position_action,
            cash=state.cash,
            realized_pnl=state.realized_pnl,
        )

        return ExecutionResult(
            success=True,
            intent_id=intent.id,
            order_id=order_id,
            fill_price=fill_price,
            quantity_filled=quantity,
            fees=fees,
            position_action=position_action,
            position=position,
            events_recorded=events_recorded,
            correlation_id=intent.correlation_id,
        )

    async def _check_idempotency(
        self, workspace_id: UUID, intent_id: UUID
    ) -> Optional[TradeEvent]:
        """Check if intent was already executed."""
        filters = EventFilters(
            workspace_id=workspace_id,
            intent_id=intent_id,
            event_types=[TradeEventType.ORDER_FILLED],
        )
        events, _ = await self._events_repo.list_events(filters, limit=1, offset=0)

        # Also check payload for mode=paper
        for event in events:
            if event.payload.get("mode") == self.mode:
                return event
        return None

    def _build_current_state(
        self, state: PaperState, workspace_id: UUID
    ) -> CurrentState:
        """Build CurrentState for policy evaluation."""
        # Convert positions to PositionState format
        from app.schemas import PositionState

        positions = []
        for symbol, pos in state.positions.items():
            if pos.quantity > 0:
                positions.append(
                    PositionState(
                        symbol=symbol,
                        side=pos.side or "long",
                        quantity=pos.quantity,
                        entry_price=pos.avg_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl_today=pos.realized_pnl,
                    )
                )

        return CurrentState(
            kill_switch_active=False,
            trading_enabled=True,
            positions=positions,
            account_equity=state.cash
            + sum(p.quantity * p.avg_price for p in state.positions.values()),
            daily_pnl=state.realized_pnl,
        )

    def _action_to_side(self, action: IntentAction) -> OrderSide:
        """Convert intent action to order side."""
        if action == IntentAction.OPEN_LONG:
            return OrderSide.BUY
        elif action == IntentAction.CLOSE_LONG:
            return OrderSide.SELL
        else:
            # Should not reach here due to validation
            return OrderSide.BUY

    def _validate_and_update_position(
        self,
        state: PaperState,
        intent: TradeIntent,
        side: OrderSide,
        quantity: float,
        fill_price: float,
    ) -> dict:
        """Validate and update position, returning action and position."""
        symbol = intent.symbol
        existing = state.positions.get(symbol)
        now = datetime.utcnow()

        if intent.action == IntentAction.OPEN_LONG:
            if existing is None or existing.quantity == 0:
                # New position
                position = PaperPosition(
                    workspace_id=intent.workspace_id,
                    symbol=symbol,
                    side="long",
                    quantity=quantity,
                    avg_price=fill_price,
                    opened_at=now,
                    last_updated_at=now,
                    order_ids=[],
                    intent_ids=[str(intent.id)],
                )
                state.positions[symbol] = position
                return {"action": "opened", "position": position}
            else:
                # Scale position
                old_qty = existing.quantity
                new_qty = old_qty + quantity
                new_avg = (
                    existing.avg_price * old_qty + fill_price * quantity
                ) / new_qty

                existing.quantity = new_qty
                existing.avg_price = new_avg
                existing.last_updated_at = now
                existing.intent_ids.append(str(intent.id))
                return {"action": "scaled", "position": existing}

        elif intent.action == IntentAction.CLOSE_LONG:
            if existing is None or existing.quantity == 0:
                return {
                    "error": "No position to close",
                    "error_code": "NO_POSITION",
                }

            # Validate full close only (PR1)
            if abs(quantity - existing.quantity) > 0.0001:  # Float tolerance
                return {
                    "error": f"Partial close not supported. Position qty: {existing.quantity}, requested: {quantity}",
                    "error_code": "PARTIAL_CLOSE_NOT_SUPPORTED",
                }

            # Calculate realized P&L
            realized_pnl = (fill_price - existing.avg_price) * quantity

            existing.realized_pnl += realized_pnl
            existing.quantity = 0
            existing.side = None
            existing.last_updated_at = now
            existing.intent_ids.append(str(intent.id))

            return {
                "action": "closed",
                "position": existing,
                "realized_pnl": realized_pnl,
            }

        return {"error": "Unknown action", "error_code": "UNKNOWN_ACTION"}

    def _update_cash_ledger(
        self,
        state: PaperState,
        side: OrderSide,
        quantity: float,
        fill_price: float,
        fees: float,
        position_result: dict,
    ) -> None:
        """Update cash and P&L based on fill."""
        if side == OrderSide.BUY:
            # Cash decreases when buying
            state.cash -= (quantity * fill_price) + fees
        elif side == OrderSide.SELL:
            # Cash increases when selling
            state.cash += (quantity * fill_price) - fees

            # Update total realized P&L
            if "realized_pnl" in position_result:
                state.realized_pnl += position_result["realized_pnl"]

    def _create_execution_events(
        self,
        intent: TradeIntent,
        order: PaperOrder,
        position_action: str,
        position: PaperPosition,
    ) -> list[TradeEvent]:
        """Create journal events for execution."""
        events = []
        now = datetime.utcnow()

        # ORDER_FILLED (source of truth for reconciliation)
        events.append(
            TradeEvent(
                correlation_id=intent.correlation_id,
                workspace_id=intent.workspace_id,
                event_type=TradeEventType.ORDER_FILLED,
                strategy_entity_id=intent.strategy_entity_id,
                symbol=intent.symbol,
                timeframe=intent.timeframe,
                intent_id=intent.id,
                order_id=str(order.id),
                payload={
                    "order_id": str(order.id),
                    "intent_id": str(intent.id),
                    "symbol": intent.symbol,
                    "side": order.side.value,
                    "qty": order.quantity,
                    "fill_price": order.fill_price,
                    "fees": order.fees,
                    "ts": now.isoformat(),
                    "mode": self.mode,
                },
            )
        )

        # POSITION_* event (observability breadcrumb)
        position_event_type = {
            "opened": TradeEventType.POSITION_OPENED,
            "closed": TradeEventType.POSITION_CLOSED,
            "scaled": TradeEventType.POSITION_SCALED,
        }.get(position_action)

        if position_event_type:
            events.append(
                TradeEvent(
                    correlation_id=intent.correlation_id,
                    workspace_id=intent.workspace_id,
                    event_type=position_event_type,
                    strategy_entity_id=intent.strategy_entity_id,
                    symbol=intent.symbol,
                    timeframe=intent.timeframe,
                    intent_id=intent.id,
                    order_id=str(order.id),
                    position_id=f"{intent.workspace_id}:{intent.symbol}",
                    payload={
                        "side": position.side,
                        "quantity": position.quantity,
                        "avg_price": position.avg_price,
                        "realized_pnl": position.realized_pnl,
                    },
                )
            )

        return events

    def _get_or_create_state(self, workspace_id: UUID) -> PaperState:
        """Get or create workspace state."""
        if workspace_id not in self._states:
            self._states[workspace_id] = PaperState(
                workspace_id=workspace_id,
                starting_equity=self._settings.paper_starting_equity,
                cash=self._settings.paper_starting_equity,
            )
        return self._states[workspace_id]

    async def get_positions(self, workspace_id: UUID) -> list[PaperPosition]:
        """Get all open positions for workspace."""
        state = self._states.get(workspace_id)
        if state is None:
            return []
        return [p for p in state.positions.values() if p.quantity > 0]

    async def get_position(
        self, workspace_id: UUID, symbol: str
    ) -> Optional[PaperPosition]:
        """Get position for symbol."""
        state = self._states.get(workspace_id)
        if state is None:
            return None
        return state.positions.get(symbol)

    async def get_state(self, workspace_id: UUID) -> PaperState:
        """Get complete paper state."""
        return self._get_or_create_state(workspace_id)

    async def reconcile_from_journal(self, workspace_id: UUID) -> ReconciliationResult:
        """Rebuild state from journal events."""
        log = logger.bind(workspace_id=str(workspace_id))
        log.info("reconcile_start")

        # Clear existing state
        state = PaperState(
            workspace_id=workspace_id,
            starting_equity=self._settings.paper_starting_equity,
            cash=self._settings.paper_starting_equity,
        )

        # Fetch ORDER_FILLED events only (source of truth)
        # Paginate through all matching events to avoid missing fills in
        # workspaces with more than the per-page limit.
        filters = EventFilters(
            workspace_id=workspace_id,
            event_types=[TradeEventType.ORDER_FILLED],
        )
        events: list[TradeEvent] = []
        offset = 0
        page_size = 10000
        total = 0

        while True:
            page, page_total = await self._events_repo.list_events(
                filters=filters,
                limit=page_size,
                offset=offset,
            )
            if not page:
                break

            events.extend(page)
            total = page_total
            offset += len(page)

            if offset >= total:
                break

        errors: list[str] = []
        processed_order_ids: set[str] = set()

        # Sort by created_at
        events_sorted = sorted(events, key=lambda e: e.created_at)

        for event in events_sorted:
            try:
                self._replay_fill_event(state, event, processed_order_ids, errors)
            except Exception as e:
                errors.append(f"Event {event.id}: {str(e)}")

        # Count non-zero positions
        positions_rebuilt = sum(1 for p in state.positions.values() if p.quantity > 0)

        # Store rebuilt state
        self._states[workspace_id] = state
        state.reconciled_at = datetime.utcnow()

        log.info(
            "reconcile_complete",
            events_replayed=len(events_sorted),
            positions_rebuilt=positions_rebuilt,
            orders_rebuilt=state.orders_count,
            errors_count=len(errors),
        )

        return ReconciliationResult(
            success=len(errors) == 0,
            workspace_id=workspace_id,
            events_replayed=len(events_sorted),
            orders_rebuilt=state.orders_count,
            positions_rebuilt=positions_rebuilt,
            cash_after=state.cash,
            realized_pnl_after=state.realized_pnl,
            last_event_at=events_sorted[-1].created_at if events_sorted else None,
            errors=errors,
        )

    def _replay_fill_event(
        self,
        state: PaperState,
        event: TradeEvent,
        processed_order_ids: set[str],
        errors: list[str],
    ) -> None:
        """Replay a single ORDER_FILLED event."""
        payload = event.payload
        order_id = payload.get("order_id")

        # Dedupe by order_id
        if order_id and order_id in processed_order_ids:
            return
        if order_id:
            processed_order_ids.add(order_id)

        symbol = payload.get("symbol") or event.symbol
        side = payload.get("side")
        qty = payload.get("qty", 0)
        fill_price = payload.get("fill_price", 0)
        fees = payload.get("fees", 0)
        mode = payload.get("mode")

        # Only replay paper mode events
        if mode and mode != self.mode:
            return

        if not symbol or qty <= 0 or fill_price <= 0:
            errors.append(f"Event {event.id}: Invalid payload")
            return

        # Get or create position
        if symbol not in state.positions:
            state.positions[symbol] = PaperPosition(
                workspace_id=event.workspace_id,
                symbol=symbol,
                side=None,
                quantity=0,
                avg_price=0,
                last_updated_at=event.created_at,
            )

        position = state.positions[symbol]

        if side == "buy":
            # BUY: open or scale position
            state.cash -= (qty * fill_price) + fees

            if position.quantity == 0:
                # New position
                position.side = "long"
                position.quantity = qty
                position.avg_price = fill_price
                position.opened_at = event.created_at
            else:
                # Scale position
                old_qty = position.quantity
                new_qty = old_qty + qty
                position.avg_price = (
                    position.avg_price * old_qty + fill_price * qty
                ) / new_qty
                position.quantity = new_qty

            position.last_updated_at = event.created_at

        elif side == "sell":
            # SELL: close position
            if position.quantity == 0:
                errors.append(f"Event {event.id}: SELL with no position for {symbol}")
                return

            # Validate full close
            if abs(qty - position.quantity) > 0.0001:
                errors.append(
                    f"Event {event.id}: SELL qty ({qty}) != position qty ({position.quantity}) for {symbol}"
                )
                # Still process to maintain cash consistency
                qty = min(qty, position.quantity)

            # Cash accounting:
            # - We add gross sale proceeds and subtract fees to reflect the actual cash movement.
            # P&L accounting:
            # - realized_pnl is tracked *net of fees* so performance metrics include trading costs.
            # - This means fees affect both the cash ledger (as an outflow) and the realized_pnl metric.
            # - Equity is only reduced once in aggregate (via cash); realized_pnl is a reporting figure.
            state.cash += (qty * fill_price) - fees
            realized_pnl = (fill_price - position.avg_price) * qty - fees
            state.realized_pnl += realized_pnl
            position.realized_pnl += realized_pnl

            position.quantity -= qty
            if position.quantity <= 0:
                position.quantity = 0
                position.side = None
            position.last_updated_at = event.created_at

        state.orders_count += 1
        state.last_event_id = event.id
        state.last_event_at = event.created_at

    async def reset(self, workspace_id: UUID) -> None:
        """Reset paper state (dev only)."""
        if workspace_id in self._states:
            del self._states[workspace_id]
        logger.warning("paper_state_reset", workspace_id=str(workspace_id))
