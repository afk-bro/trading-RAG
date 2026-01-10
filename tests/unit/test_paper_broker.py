"""Unit tests for paper broker."""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.execution.paper_broker import PaperBroker, SUPPORTED_ACTIONS
from app.schemas import (
    TradeIntent,
    IntentAction,
    PolicyDecision,
    PolicyReason,
    TradeEventType,
)


@pytest.fixture
def mock_events_repo():
    """Create mock events repository."""
    repo = MagicMock()
    repo.insert = AsyncMock(return_value=uuid4())
    repo.insert_many = AsyncMock(return_value=2)
    repo.list_events = AsyncMock(return_value=([], 0))
    return repo


@pytest.fixture
def paper_broker(mock_events_repo):
    """Create paper broker with mocked repo."""
    return PaperBroker(mock_events_repo)


@pytest.fixture
def sample_intent():
    """Create sample OPEN_LONG intent."""
    return TradeIntent(
        correlation_id="test-123",
        workspace_id=uuid4(),
        action=IntentAction.OPEN_LONG,
        strategy_entity_id=uuid4(),
        symbol="BTC/USDT",
        timeframe="5m",
        quantity=1.0,
    )


class TestPaperBrokerValidation:
    """Tests for intent validation."""

    @pytest.mark.asyncio
    async def test_rejects_unsupported_action(self, paper_broker, sample_intent):
        """Rejects unsupported action with 400-equivalent error."""
        # Modify to unsupported action
        sample_intent.action = IntentAction.OPEN_SHORT

        result = await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        assert not result.success
        assert result.error_code == "UNSUPPORTED_ACTION"
        assert "open_short" in result.error.lower()

    @pytest.mark.asyncio
    async def test_supported_actions_are_open_long_close_long(self):
        """Only OPEN_LONG and CLOSE_LONG are supported in PR1."""
        assert SUPPORTED_ACTIONS == {IntentAction.OPEN_LONG, IntentAction.CLOSE_LONG}

    @pytest.mark.asyncio
    async def test_rejects_invalid_fill_price(self, paper_broker, sample_intent):
        """Rejects negative or zero fill price."""
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(sample_intent, fill_price=0)

        assert not result.success
        assert result.error_code == "INVALID_FILL_PRICE"

    @pytest.mark.asyncio
    async def test_rejects_zero_quantity(self, paper_broker):
        """Rejects intent with zero quantity."""
        intent = TradeIntent(
            correlation_id="test-123",
            workspace_id=uuid4(),
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=0,  # Zero quantity
        )

        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(intent, fill_price=50000.0)

        assert not result.success
        assert result.error_code == "INVALID_QUANTITY"


class TestPaperBrokerPolicyRecheck:
    """Tests for internal policy re-evaluation."""

    @pytest.mark.asyncio
    async def test_rejects_when_policy_fails(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """Rejects intent when policy engine rejects."""
        # Mock policy engine to reject
        mock_decision = PolicyDecision(
            approved=False,
            reason=PolicyReason.KILL_SWITCH_ACTIVE,
            reason_details="Kill switch is active",
            rules_evaluated=["kill_switch"],
            rules_passed=[],
            rules_failed=["kill_switch"],
        )

        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            with patch.object(
                paper_broker._policy_engine, "evaluate", return_value=mock_decision
            ):
                result = await paper_broker.execute_intent(
                    sample_intent, fill_price=50000.0
                )

        assert not result.success
        assert result.error_code == "POLICY_REJECTED"
        assert "kill_switch" in result.error.lower()

        # Should have journaled rejection event
        mock_events_repo.insert.assert_called_once()


class TestPaperBrokerOpenLong:
    """Tests for OPEN_LONG execution."""

    @pytest.mark.asyncio
    async def test_opens_long_position(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """Opens new long position, cash decreases."""
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(
                sample_intent, fill_price=50000.0
            )

        assert result.success
        assert result.position_action == "opened"
        assert result.fill_price == 50000.0
        assert result.quantity_filled == 1.0
        assert result.position.side == "long"
        assert result.position.quantity == 1.0
        assert result.position.avg_price == 50000.0

        # Verify cash decreased
        state = await paper_broker.get_state(sample_intent.workspace_id)
        assert state.cash == 10000.0 - 50000.0  # starting - (qty * price)

    @pytest.mark.asyncio
    async def test_scales_position_on_duplicate_open_long(
        self, paper_broker, mock_events_repo
    ):
        """Scales position when OPEN_LONG on existing position."""
        workspace_id = uuid4()

        # First OPEN_LONG
        intent1 = TradeIntent(
            correlation_id="test-1",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result1 = await paper_broker.execute_intent(intent1, fill_price=50000.0)

        assert result1.success
        assert result1.position_action == "opened"

        # Second OPEN_LONG (scale)
        intent2 = TradeIntent(
            correlation_id="test-2",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result2 = await paper_broker.execute_intent(intent2, fill_price=52000.0)

        assert result2.success
        assert result2.position_action == "scaled"
        assert result2.position.quantity == 2.0
        # Avg price: (50000 * 1 + 52000 * 1) / 2 = 51000
        assert result2.position.avg_price == 51000.0


class TestPaperBrokerCloseLong:
    """Tests for CLOSE_LONG execution."""

    @pytest.mark.asyncio
    async def test_closes_long_position_with_pnl(self, paper_broker, mock_events_repo):
        """Closes position, cash increases, realized P&L calculated."""
        workspace_id = uuid4()

        # Open position first
        open_intent = TradeIntent(
            correlation_id="test-open",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(open_intent, fill_price=50000.0)

        # Close position
        close_intent = TradeIntent(
            correlation_id="test-close",
            workspace_id=workspace_id,
            action=IntentAction.CLOSE_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(close_intent, fill_price=51000.0)

        assert result.success
        assert result.position_action == "closed"
        assert result.position.quantity == 0
        assert result.position.side is None

        # Verify P&L
        state = await paper_broker.get_state(workspace_id)
        # Profit: (51000 - 50000) * 1 = 1000
        assert state.realized_pnl == 1000.0
        # Cash: 10000 - 50000 + 51000 = 11000
        assert state.cash == 11000.0

    @pytest.mark.asyncio
    async def test_rejects_partial_close(self, paper_broker, mock_events_repo):
        """Rejects partial close (qty != position.qty)."""
        workspace_id = uuid4()

        # Open position
        open_intent = TradeIntent(
            correlation_id="test-open",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=2.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(open_intent, fill_price=50000.0)

        # Try partial close
        close_intent = TradeIntent(
            correlation_id="test-close",
            workspace_id=workspace_id,
            action=IntentAction.CLOSE_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,  # Less than position qty of 2.0
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(close_intent, fill_price=51000.0)

        assert not result.success
        assert result.error_code == "PARTIAL_CLOSE_NOT_SUPPORTED"

    @pytest.mark.asyncio
    async def test_rejects_close_with_no_position(self, paper_broker, mock_events_repo):
        """Rejects CLOSE_LONG when no position exists."""
        close_intent = TradeIntent(
            correlation_id="test-close",
            workspace_id=uuid4(),
            action=IntentAction.CLOSE_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(close_intent, fill_price=51000.0)

        assert not result.success
        assert result.error_code == "NO_POSITION"


class TestPaperBrokerJournaling:
    """Tests for event journaling."""

    @pytest.mark.asyncio
    async def test_journals_order_filled_with_required_fields(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """Journals ORDER_FILLED with replay-friendly payload."""
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        # Get the events that were journaled
        mock_events_repo.insert_many.assert_called_once()
        events = mock_events_repo.insert_many.call_args[0][0]

        # Find ORDER_FILLED event
        fill_event = next(
            e for e in events if e.event_type == TradeEventType.ORDER_FILLED
        )
        payload = fill_event.payload

        # Verify required fields for replay
        assert "order_id" in payload
        assert "intent_id" in payload
        assert "symbol" in payload
        assert payload["side"] == "buy"
        assert payload["qty"] == 1.0
        assert payload["fill_price"] == 50000.0
        assert "fees" in payload
        assert "ts" in payload
        assert payload["mode"] == "paper"


class TestPaperBrokerIdempotency:
    """Tests for idempotency."""

    @pytest.mark.asyncio
    async def test_duplicate_execute_returns_409(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """Duplicate execute with same intent_id returns error with prior correlation_id."""
        # First execution
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result1 = await paper_broker.execute_intent(
                sample_intent, fill_price=50000.0
            )

        assert result1.success

        # Mock idempotency check to return existing event
        from app.schemas import TradeEvent

        existing_event = TradeEvent(
            correlation_id="test-123",
            workspace_id=sample_intent.workspace_id,
            event_type=TradeEventType.ORDER_FILLED,
            symbol=sample_intent.symbol,
            payload={"mode": "paper"},
        )

        with patch.object(
            paper_broker, "_check_idempotency", return_value=existing_event
        ):
            result2 = await paper_broker.execute_intent(
                sample_intent, fill_price=50000.0
            )

        assert not result2.success
        assert result2.error_code == "ALREADY_EXECUTED"
        assert result2.correlation_id == "test-123"

    @pytest.mark.asyncio
    async def test_idempotency_no_state_change_on_duplicate(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """State doesn't change on duplicate execution."""
        # First execution
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        state_after_first = await paper_broker.get_state(sample_intent.workspace_id)
        cash_after_first = state_after_first.cash
        orders_after_first = state_after_first.orders_count

        # Mock idempotency check to return existing event
        from app.schemas import TradeEvent

        existing_event = TradeEvent(
            correlation_id="test-123",
            workspace_id=sample_intent.workspace_id,
            event_type=TradeEventType.ORDER_FILLED,
            symbol=sample_intent.symbol,
            payload={"mode": "paper"},
        )

        with patch.object(
            paper_broker, "_check_idempotency", return_value=existing_event
        ):
            await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        state_after_second = await paper_broker.get_state(sample_intent.workspace_id)

        # State should be unchanged
        assert state_after_second.cash == cash_after_first
        assert state_after_second.orders_count == orders_after_first


class TestPaperBrokerReconciliation:
    """Tests for journal reconciliation."""

    @pytest.mark.asyncio
    async def test_reconciles_empty_journal(self, paper_broker, mock_events_repo):
        """Reconciles with empty journal."""
        workspace_id = uuid4()
        mock_events_repo.list_events = AsyncMock(return_value=([], 0))

        result = await paper_broker.reconcile_from_journal(workspace_id)

        assert result.success
        assert result.events_replayed == 0
        assert result.positions_rebuilt == 0
        assert result.cash_after == 10000.0  # Starting equity
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_reconcile_determinism(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """State matches after reset + reconcile."""

        workspace_id = sample_intent.workspace_id

        # Execute intent
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        state_before = await paper_broker.get_state(workspace_id)

        # Capture the events that were journaled
        events = mock_events_repo.insert_many.call_args[0][0]
        fill_event = next(
            e for e in events if e.event_type == TradeEventType.ORDER_FILLED
        )

        # Reset in-memory state
        await paper_broker.reset(workspace_id)

        # Verify reset
        state_after_reset = await paper_broker.get_state(workspace_id)
        assert state_after_reset.cash == 10000.0
        assert len(state_after_reset.positions) == 0

        # Mock list_events to return the fill event
        mock_events_repo.list_events = AsyncMock(return_value=([fill_event], 1))

        # Reconcile
        reconcile_result = await paper_broker.reconcile_from_journal(workspace_id)

        assert reconcile_result.success
        assert reconcile_result.events_replayed == 1

        # Verify state matches pre-reset
        state_after_reconcile = await paper_broker.get_state(workspace_id)
        assert state_after_reconcile.cash == state_before.cash
        assert len(state_after_reconcile.positions) == len(state_before.positions)

    @pytest.mark.asyncio
    async def test_reconcile_dedupe_by_order_id(self, paper_broker, mock_events_repo):
        """Duplicate order_id in journal doesn't double-apply."""
        from app.schemas import TradeEvent

        workspace_id = uuid4()
        order_id = str(uuid4())

        # Create duplicate events with same order_id
        fill_event = TradeEvent(
            correlation_id="test-123",
            workspace_id=workspace_id,
            event_type=TradeEventType.ORDER_FILLED,
            symbol="BTC/USDT",
            payload={
                "order_id": order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "qty": 1.0,
                "fill_price": 50000.0,
                "fees": 0,
                "mode": "paper",
            },
        )

        # Return same event twice (simulating duplicate)
        mock_events_repo.list_events = AsyncMock(
            return_value=([fill_event, fill_event], 2)
        )

        result = await paper_broker.reconcile_from_journal(workspace_id)

        assert result.success
        # Should only process once due to dedupe
        assert result.orders_rebuilt == 1

    @pytest.mark.asyncio
    async def test_reconcile_error_on_invalid_sell(
        self, paper_broker, mock_events_repo
    ):
        """SELL with qty != position.qty surfaces in errors."""
        from app.schemas import TradeEvent

        workspace_id = uuid4()

        # Create a SELL event without corresponding BUY
        sell_event = TradeEvent(
            correlation_id="test-123",
            workspace_id=workspace_id,
            event_type=TradeEventType.ORDER_FILLED,
            symbol="BTC/USDT",
            payload={
                "order_id": str(uuid4()),
                "symbol": "BTC/USDT",
                "side": "sell",
                "qty": 1.0,
                "fill_price": 51000.0,
                "fees": 0,
                "mode": "paper",
            },
        )

        mock_events_repo.list_events = AsyncMock(return_value=([sell_event], 1))

        result = await paper_broker.reconcile_from_journal(workspace_id)

        # Should have error for SELL with no position
        assert len(result.errors) > 0
        assert "no position" in result.errors[0].lower()


class TestPaperBrokerState:
    """Tests for state management."""

    @pytest.mark.asyncio
    async def test_get_positions_returns_open_only(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """get_positions only returns positions with qty > 0."""
        workspace_id = sample_intent.workspace_id

        # Open position
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        positions = await paper_broker.get_positions(workspace_id)
        assert len(positions) == 1
        assert positions[0].quantity == 1.0

        # Close position
        close_intent = TradeIntent(
            correlation_id="test-close",
            workspace_id=workspace_id,
            action=IntentAction.CLOSE_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(close_intent, fill_price=51000.0)

        positions = await paper_broker.get_positions(workspace_id)
        assert len(positions) == 0  # No open positions

    @pytest.mark.asyncio
    async def test_reset_clears_state(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """Reset clears in-memory state."""
        workspace_id = sample_intent.workspace_id

        # Execute to create state
        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            await paper_broker.execute_intent(sample_intent, fill_price=50000.0)

        state = await paper_broker.get_state(workspace_id)
        assert state.cash != 10000.0  # Modified

        # Reset
        await paper_broker.reset(workspace_id)

        # State should be fresh
        state = await paper_broker.get_state(workspace_id)
        assert state.cash == 10000.0
        assert len(state.positions) == 0
