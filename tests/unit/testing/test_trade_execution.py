"""Trade execution tests for RunOrchestrator._simulate_bars().

Tests invariants around trade execution, position state, and PnL calculation.
Uses a deterministic test strategy stub to isolate execution logic.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.schemas import PaperState, IntentAction, TradeIntent
from app.services.strategy.models import (
    OHLCVBar,
    ExecutionSpec,
    MarketSnapshot,
    EntryConfig,
    ExitConfig,
    RiskConfig,
    StrategyEvaluation,
)
from app.services.testing.run_orchestrator import RunOrchestrator


# =============================================================================
# Test Helpers
# =============================================================================


def make_bars(
    price_data: list[tuple[float, float, float, float, float]]
) -> list[OHLCVBar]:
    """Create OHLCV bars from (open, high, low, close, volume) tuples.

    Timestamps start at 2024-01-01 and increment daily.
    """
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i, (o, h, l, c, v) in enumerate(price_data):
        bars.append(
            OHLCVBar(
                ts=base_ts + timedelta(days=i),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
        )
    return bars


def make_flat_bars(close_price: float, count: int) -> list[OHLCVBar]:
    """Create N bars at a constant price."""
    return make_bars(
        [(close_price, close_price, close_price, close_price, 1000.0)] * count
    )


def make_spec(
    workspace_id=None,
    dollars_per_trade: float = 1000.0,
    lookback_days: int = 2,
) -> ExecutionSpec:
    """Create a minimal ExecutionSpec for testing."""
    return ExecutionSpec(
        strategy_id="test_strategy",
        instance_id=uuid4(),
        name="Test Strategy",
        workspace_id=workspace_id or uuid4(),
        symbols=["TEST"],
        timeframe="daily",
        entry=EntryConfig(type="test_strategy", lookback_days=lookback_days),
        exit=ExitConfig(type="eod"),
        risk=RiskConfig(dollars_per_trade=dollars_per_trade, max_positions=5),
    )


def make_paper_state(
    workspace_id=None,
    starting_equity: float = 10000.0,
) -> PaperState:
    """Create a fresh PaperState for testing."""
    return PaperState(
        workspace_id=workspace_id or uuid4(),
        starting_equity=starting_equity,
        cash=starting_equity,
        realized_pnl=0.0,
    )


def make_intent(
    action: IntentAction,
    symbol: str = "TEST",
    qty: float = 1.0,
) -> TradeIntent:
    """Create a TradeIntent for testing."""
    return TradeIntent(
        id=uuid4(),
        correlation_id=str(uuid4()),
        workspace_id=uuid4(),
        strategy_entity_id=uuid4(),
        symbol=symbol,
        timeframe="daily",
        action=action,
        qty=qty,
        reason="test",
    )


class DeterministicRunner:
    """Test strategy runner that emits intents on specific bar indices.

    Configure with a dict mapping bar index -> list of intents to emit.
    Example: {1: [OPEN_LONG], 3: [CLOSE_LONG]} opens on bar 1, closes on bar 3.
    """

    def __init__(self, intent_schedule: dict[int, list[IntentAction]]):
        """
        Args:
            intent_schedule: Dict mapping bar index to list of IntentActions
        """
        self._schedule = intent_schedule
        self._call_count = 0

    def evaluate(
        self,
        spec: ExecutionSpec,
        snapshot: MarketSnapshot,
        paper_state: PaperState,
    ) -> StrategyEvaluation:
        """Return scheduled intents based on call count (bar index)."""
        bar_index = self._call_count
        self._call_count += 1

        intents = []
        if bar_index in self._schedule:
            for action in self._schedule[bar_index]:
                intents.append(make_intent(action, symbol=spec.symbols[0]))

        return StrategyEvaluation(
            spec_id=str(spec.instance_id),
            symbol=spec.symbols[0],
            ts=snapshot.ts,
            intents=intents,
            signals=[f"bar_{bar_index}"],
            metadata={"bar_index": bar_index},
        )


# =============================================================================
# Trade Creation Tests
# =============================================================================


class TestTradeCreation:
    """Tests for trade creation invariants."""

    def test_buy_intent_creates_trade(self):
        """A BUY intent on a bar should create exactly one trade entry."""
        # 5 bars at $100, BUY on bar 0 (first bar after lookback)
        bars = make_flat_bars(100.0, 5)
        spec = make_spec(lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner({0: [IntentAction.OPEN_LONG]})

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, events = orchestrator._simulate_bars(
            spec, paper_state, bars, MagicMock()
        )

        # Position opened but not closed (force closed at end)
        assert len(trades) == 1
        assert trades[0]["entry_price"] == 100.0
        assert trades[0]["symbol"] == "TEST"

    def test_no_intent_no_trade(self):
        """No intent should result in no trades."""
        bars = make_flat_bars(100.0, 5)
        spec = make_spec(lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner({})  # No intents

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, events = orchestrator._simulate_bars(
            spec, paper_state, bars, MagicMock()
        )

        assert len(trades) == 0

    def test_buy_then_sell_creates_single_round_trip(self):
        """BUY then SELL creates one complete trade."""
        bars = make_flat_bars(100.0, 10)
        spec = make_spec(lookback_days=2)
        paper_state = make_paper_state()
        # BUY on bar 0, SELL on bar 2
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],
                2: [IntentAction.CLOSE_LONG],
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        assert len(trades) == 1
        assert trades[0]["entry_price"] == 100.0
        assert trades[0]["exit_price"] == 100.0
        assert trades[0]["pnl"] == 0.0  # Flat price, no gain


# =============================================================================
# Position State Tests
# =============================================================================


class TestPositionState:
    """Tests for position state invariants."""

    def test_position_qty_after_buy(self):
        """BUY should create position with correct quantity."""
        bars = make_flat_bars(100.0, 5)
        spec = make_spec(dollars_per_trade=1000.0, lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner({0: [IntentAction.OPEN_LONG]})

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        # Position was opened with quantity = dollars_per_trade / price = 1000/100 = 10
        # Verified via the closed trade (force-closed at end)
        assert len(trades) == 1
        assert trades[0]["qty"] == 10.0
        assert trades[0]["entry_price"] == 100.0

    def test_position_cleared_after_sell(self):
        """After SELL closing, position should be cleared."""
        bars = make_flat_bars(100.0, 10)
        spec = make_spec(lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],
                2: [IntentAction.CLOSE_LONG],
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        # Position should be cleared
        assert "TEST" not in paper_state.positions

    def test_cash_restored_after_round_trip(self):
        """Cash should be restored after round-trip trade at same price."""
        bars = make_flat_bars(100.0, 10)
        spec = make_spec(dollars_per_trade=1000.0, lookback_days=2)
        paper_state = make_paper_state(starting_equity=10000.0)
        # BUY then SELL at same price - cash should be unchanged
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],
                2: [IntentAction.CLOSE_LONG],
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        # Cash should be back to starting (flat trade, no PnL)
        assert paper_state.cash == 10000.0
        assert paper_state.realized_pnl == 0.0


# =============================================================================
# Insufficient Balance Tests
# =============================================================================


class TestInsufficientBalance:
    """Tests for rejection when balance is insufficient."""

    def test_no_fill_when_insufficient_cash(self):
        """BUY intent should not fill when cash is insufficient."""
        bars = make_flat_bars(100.0, 5)
        spec = make_spec(dollars_per_trade=5000.0, lookback_days=2)  # Needs $5000
        paper_state = make_paper_state(starting_equity=1000.0)  # Only $1000
        runner = DeterministicRunner({0: [IntentAction.OPEN_LONG]})

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        # No trades should be created
        assert len(trades) == 0
        # Cash unchanged
        assert paper_state.cash == 1000.0

    def test_second_buy_rejected_when_no_cash_left(self):
        """Second BUY should be rejected if first position used all cash."""
        bars = make_flat_bars(100.0, 10)
        spec = make_spec(dollars_per_trade=9000.0, lookback_days=2)  # Needs $9000
        paper_state = make_paper_state(starting_equity=10000.0)  # Has $10000
        # Try two BUYs - second should fail
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],
                1: [IntentAction.OPEN_LONG],  # Should be rejected
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        # Only one trade (force-closed at end)
        assert len(trades) == 1
        # Cash should be 10000 - 9000 = 1000 after first buy
        # (before force close returns proceeds)


# =============================================================================
# PnL Calculation Tests
# =============================================================================


class TestPnLCalculation:
    """Tests for PnL calculation correctness."""

    def test_profitable_trade_pnl(self):
        """PnL should be positive when exit_price > entry_price."""
        # Prices: bar0=100, bar1=100, bar2=100, bar3=110 (exit)
        bars = make_bars(
            [
                (100, 100, 100, 100, 1000),  # bar 0
                (100, 100, 100, 100, 1000),  # bar 1
                (100, 100, 100, 100, 1000),  # bar 2
                (100, 110, 100, 110, 1000),  # bar 3 - price rises
                (110, 110, 110, 110, 1000),  # bar 4
            ]
        )
        spec = make_spec(dollars_per_trade=1000.0, lookback_days=2)
        paper_state = make_paper_state()
        # BUY at bar 0 (price=100), SELL at bar 2 (price=110 - bar 3 in actual index)
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],  # Entry at 100
                2: [IntentAction.CLOSE_LONG],  # Exit at 110
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        assert len(trades) == 1
        trade = trades[0]

        # Entry at bar index 2 (lookback=2, so first eval is bar[2])
        # But our runner counts from 0 for simplicity
        # Let's verify the actual prices
        assert trade["entry_price"] == 100.0
        assert trade["exit_price"] == 110.0

        # qty = 1000 / 100 = 10
        # pnl = (110 - 100) * 10 = 100
        assert trade["qty"] == 10.0
        assert trade["pnl"] == 100.0

    def test_losing_trade_pnl(self):
        """PnL should be negative when exit_price < entry_price."""
        bars = make_bars(
            [
                (100, 100, 100, 100, 1000),
                (100, 100, 100, 100, 1000),
                (100, 100, 100, 100, 1000),
                (100, 100, 90, 90, 1000),  # Price drops
                (90, 90, 90, 90, 1000),
            ]
        )
        spec = make_spec(dollars_per_trade=1000.0, lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],  # Entry at 100
                2: [IntentAction.CLOSE_LONG],  # Exit at 90
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        assert len(trades) == 1
        trade = trades[0]

        assert trade["entry_price"] == 100.0
        assert trade["exit_price"] == 90.0
        # qty = 10, pnl = (90 - 100) * 10 = -100
        assert trade["pnl"] == -100.0

    def test_realized_pnl_accumulates(self):
        """realized_pnl in paper_state should accumulate across trades."""
        bars = make_flat_bars(100.0, 15)
        spec = make_spec(dollars_per_trade=1000.0, lookback_days=2)
        paper_state = make_paper_state()
        # Two round trips at same price (0 PnL each)
        runner = DeterministicRunner(
            {
                0: [IntentAction.OPEN_LONG],
                2: [IntentAction.CLOSE_LONG],
                4: [IntentAction.OPEN_LONG],
                6: [IntentAction.CLOSE_LONG],
            }
        )

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        assert len(trades) == 2
        assert paper_state.realized_pnl == 0.0  # Two flat trades


# =============================================================================
# Force Close Tests
# =============================================================================


class TestForceClose:
    """Tests for force-close at end of data."""

    def test_open_position_force_closed_at_end(self):
        """Open position should be force-closed at end of bar data."""
        bars = make_flat_bars(100.0, 5)
        spec = make_spec(lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner({0: [IntentAction.OPEN_LONG]})  # No explicit close

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        # Should have one trade from force close
        assert len(trades) == 1
        # Position should be cleared
        assert "TEST" not in paper_state.positions

    def test_force_close_uses_last_bar_price(self):
        """Force close should use the last bar's close price."""
        bars = make_bars(
            [
                (100, 100, 100, 100, 1000),
                (100, 100, 100, 100, 1000),
                (100, 100, 100, 100, 1000),
                (100, 100, 100, 150, 1000),  # Last bar closes at 150
            ]
        )
        spec = make_spec(dollars_per_trade=1000.0, lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner({0: [IntentAction.OPEN_LONG]})

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        trades, _ = orchestrator._simulate_bars(spec, paper_state, bars, MagicMock())

        assert len(trades) == 1
        assert trades[0]["exit_price"] == 150.0
        # Entry at 100, exit at 150, qty=10 -> pnl = 500
        assert trades[0]["pnl"] == 500.0


# =============================================================================
# Event Count Tests
# =============================================================================


class TestEventCount:
    """Tests for event counting."""

    def test_events_count_matches_evaluations(self):
        """events_count should match number of runner.evaluate() calls."""
        bars = make_flat_bars(100.0, 10)
        spec = make_spec(lookback_days=2)
        paper_state = make_paper_state()
        runner = DeterministicRunner({})

        orchestrator = RunOrchestrator(
            events_repo=MagicMock(),
            runner=runner,
        )

        _, events_count = orchestrator._simulate_bars(
            spec, paper_state, bars, MagicMock()
        )

        # With 10 bars and lookback=2, we should evaluate bars 2-9 = 8 bars
        assert events_count == 8
