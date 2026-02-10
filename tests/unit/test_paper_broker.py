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


class TestPaperBrokerStrategyGating:
    """Tests for strategy state gating (paused means zero orders)."""

    @pytest.fixture
    def mock_version_repo(self):
        """Create mock version repository."""
        repo = MagicMock()
        repo.is_entity_active = AsyncMock(return_value=True)
        return repo

    @pytest.fixture
    def gated_broker(self, mock_events_repo, mock_version_repo):
        """Create paper broker with strategy gating enabled."""
        return PaperBroker(mock_events_repo, version_repo=mock_version_repo)

    @pytest.mark.asyncio
    async def test_blocks_execution_when_strategy_paused(
        self, gated_broker, sample_intent, mock_events_repo
    ):
        """Blocks execution when strategy has no active version (paused)."""
        # Mock version repo to return is_entity_active=False (paused)
        gated_broker._version_repo.is_entity_active = AsyncMock(return_value=False)

        with patch.object(gated_broker, "_check_idempotency", return_value=None):
            result = await gated_broker.execute_intent(
                sample_intent, fill_price=50000.0
            )

        assert not result.success
        assert result.error_code == "STRATEGY_PAUSED"
        assert "paused" in result.error.lower()
        assert result.events_recorded == 1  # Rejection journaled

        # Verify rejection event was journaled
        mock_events_repo.insert.assert_called_once()
        rejection_event = mock_events_repo.insert.call_args[0][0]
        assert rejection_event.event_type == TradeEventType.INTENT_REJECTED
        assert rejection_event.payload["reason"] == "STRATEGY_PAUSED"

    @pytest.mark.asyncio
    async def test_allows_execution_when_strategy_active(
        self, gated_broker, sample_intent, mock_events_repo
    ):
        """Allows execution when strategy has an active version."""
        # Mock version repo to return is_entity_active=True (active)
        gated_broker._version_repo.is_entity_active = AsyncMock(return_value=True)

        with patch.object(gated_broker, "_check_idempotency", return_value=None):
            result = await gated_broker.execute_intent(
                sample_intent, fill_price=50000.0
            )

        assert result.success
        assert result.position_action == "opened"

    @pytest.mark.asyncio
    async def test_backward_compatible_without_version_repo(
        self, paper_broker, sample_intent, mock_events_repo
    ):
        """Executes without gating when version_repo is None (backward compatible)."""
        # paper_broker fixture doesn't have version_repo
        assert paper_broker._version_repo is None

        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(
                sample_intent, fill_price=50000.0
            )

        assert result.success
        assert result.position_action == "opened"

    @pytest.mark.asyncio
    async def test_strategy_check_uses_correct_entity_id(
        self, gated_broker, sample_intent, mock_events_repo
    ):
        """Strategy check queries with the intent's strategy_entity_id."""
        gated_broker._version_repo.is_entity_active = AsyncMock(return_value=True)

        with patch.object(gated_broker, "_check_idempotency", return_value=None):
            await gated_broker.execute_intent(sample_intent, fill_price=50000.0)

        # Verify is_entity_active was called with the correct entity ID
        gated_broker._version_repo.is_entity_active.assert_called_once_with(
            sample_intent.strategy_entity_id
        )


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


class TestPaperBrokerStructuralSizing:
    """Tests for structural sizing integration in paper broker."""

    @pytest.mark.asyncio
    async def test_uses_computed_quantity(self, paper_broker, mock_events_repo):
        """Broker fills with sizing rule's quantity, not intent's."""
        intent = TradeIntent(
            correlation_id="test-sizing",
            workspace_id=uuid4(),
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="MNQ",
            timeframe="5m",
            quantity=999.0,  # Intent wants 999, sizing should override
            price=18000.0,
            stop_loss=17990.0,
            metadata={"risk_budget_dollars": 500.0},
        )

        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(intent, fill_price=18000.0)

        assert result.success
        # MNQ: 10pt stop * $2 = $20/contract, 500/20 = 25
        assert result.quantity_filled == 25.0

    @pytest.mark.asyncio
    async def test_rejects_when_sizing_fails(self, paper_broker, mock_events_repo):
        """Wide stop + small budget → POLICY_REJECTED."""
        intent = TradeIntent(
            correlation_id="test-sizing-reject",
            workspace_id=uuid4(),
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="NQ",
            timeframe="5m",
            quantity=1.0,
            price=18000.0,
            stop_loss=17900.0,  # 100pt stop on NQ = $2000 risk
            metadata={"risk_budget_dollars": 500.0},
        )

        with patch.object(paper_broker, "_check_idempotency", return_value=None):
            result = await paper_broker.execute_intent(intent, fill_price=18000.0)

        assert not result.success
        assert result.error_code == "POLICY_REJECTED"
        assert "risk_too_high" in result.error.lower()

    @pytest.mark.asyncio
    async def test_intent_metadata_budget_takes_precedence(self, mock_events_repo):
        """Intent metadata risk_budget_dollars overrides config fallback."""
        # Create broker with config fallback
        with patch("app.services.execution.paper_broker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.paper_starting_equity = 10000.0
            settings.paper_default_risk_budget_dollars = 100.0  # Config: $100
            mock_settings.return_value = settings

            broker = PaperBroker(mock_events_repo)

        # Intent metadata says $500 — should override config $100
        intent = TradeIntent(
            correlation_id="test-precedence",
            workspace_id=uuid4(),
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="MNQ",
            timeframe="5m",
            quantity=1.0,
            price=18000.0,
            stop_loss=17990.0,
            metadata={"risk_budget_dollars": 500.0},
        )

        with patch.object(broker, "_check_idempotency", return_value=None):
            result = await broker.execute_intent(intent, fill_price=18000.0)

        assert result.success
        # With $500 budget: 10pt * $2 = $20/contract, 500/20 = 25
        assert result.quantity_filled == 25.0


class TestPaperBrokerEvalProfile:
    """Tests for eval account profile integration."""

    @pytest.mark.asyncio
    async def test_risk_budget_from_eval_profile(self, mock_events_repo):
        """R_day computed from profile + state equity."""
        from app.services.backtest.engines.eval_profile import EvalAccountProfile

        workspace_id = uuid4()
        profile = EvalAccountProfile(
            account_size=50_000,
            max_drawdown_dollars=2_000,
            max_daily_loss_dollars=1_000,
            risk_fraction=0.15,
            r_min_dollars=100.0,
            r_max_dollars=300.0,
        )

        with patch("app.services.execution.paper_broker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.paper_starting_equity = 50_000.0
            settings.paper_default_risk_budget_dollars = None
            settings.paper_max_position_size_pct = 0.20
            mock_settings.return_value = settings

            broker = PaperBroker(
                mock_events_repo,
                eval_profiles={workspace_id: profile},
            )

        # Build state and check risk budget resolution
        state = broker._get_or_create_state(workspace_id)
        assert state.peak_equity == 50_000.0

        intent = TradeIntent(
            correlation_id="test-eval",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="MNQ",
            timeframe="5m",
            quantity=1.0,
        )

        current_state = broker._build_current_state(state, workspace_id, intent)
        # At start: room=2000, raw=300, capped at 300
        assert current_state.risk_budget_dollars == 300.0

    @pytest.mark.asyncio
    async def test_peak_equity_updated_after_execution(self, mock_events_repo):
        """state.peak_equity tracks high-water mark after execution."""
        from app.services.backtest.engines.eval_profile import EvalAccountProfile

        workspace_id = uuid4()
        profile = EvalAccountProfile(
            account_size=10_000,
            max_drawdown_dollars=500,
            max_daily_loss_dollars=250,
        )

        with patch("app.services.execution.paper_broker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.paper_starting_equity = 10_000.0
            settings.paper_default_risk_budget_dollars = None
            settings.paper_max_position_size_pct = 0.20
            mock_settings.return_value = settings

            broker = PaperBroker(
                mock_events_repo,
                eval_profiles={workspace_id: profile},
            )

        # Open position — mock policy to approve (we're testing peak_equity, not policy)
        approved_decision = PolicyDecision(
            approved=True,
            reason=PolicyReason.ALL_RULES_PASSED,
        )

        open_intent = TradeIntent(
            correlation_id="test-open",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(
            broker, "_check_idempotency", return_value=None
        ), patch.object(
            broker._policy_engine, "evaluate", return_value=approved_decision
        ):
            result = await broker.execute_intent(open_intent, fill_price=100.0)
        assert result.success, f"Execution failed: {result.error}"

        # Close at profit
        close_intent = TradeIntent(
            correlation_id="test-close",
            workspace_id=workspace_id,
            action=IntentAction.CLOSE_LONG,
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            quantity=1.0,
        )
        with patch.object(
            broker, "_check_idempotency", return_value=None
        ), patch.object(
            broker._policy_engine, "evaluate", return_value=approved_decision
        ):
            result = await broker.execute_intent(close_intent, fill_price=150.0)
        assert result.success, f"Close failed: {result.error}"

        state = await broker.get_state(workspace_id)
        # Cash: 10000 - 100 + 150 = 10050
        assert state.cash == 10050.0
        # Peak should be updated to current equity
        assert state.peak_equity == 10050.0

    @pytest.mark.asyncio
    async def test_eval_blown_returns_zero_budget(self, mock_events_repo):
        """Blown eval -> 0 budget from profile."""
        from app.services.backtest.engines.eval_profile import EvalAccountProfile

        workspace_id = uuid4()
        profile = EvalAccountProfile(
            account_size=10_000,
            max_drawdown_dollars=100,  # Very tight
            max_daily_loss_dollars=50,
        )

        with patch("app.services.execution.paper_broker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.paper_starting_equity = 10_000.0
            settings.paper_default_risk_budget_dollars = None
            settings.paper_max_position_size_pct = 0.20
            mock_settings.return_value = settings

            broker = PaperBroker(
                mock_events_repo,
                eval_profiles={workspace_id: profile},
            )

        # Simulate drawdown by manipulating state
        state = broker._get_or_create_state(workspace_id)
        state.cash = 9_850.0  # Lost $150, peak is $10,000
        # peak_equity stays at 10000, equity is 9850
        # trailing DD = 150 > max_drawdown 100 -> blown

        intent = TradeIntent(
            correlation_id="test-blown",
            workspace_id=workspace_id,
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid4(),
            symbol="MNQ",
            timeframe="5m",
            quantity=1.0,
        )

        current_state = broker._build_current_state(state, workspace_id, intent)
        assert current_state.risk_budget_dollars == 0.0
