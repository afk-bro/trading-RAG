"""Unit tests for StrategyRunner and strategy implementations.

Tests behavior of the strategy execution engine:
- StrategyRunner.evaluate() method
- evaluate_breakout_52w_high() function
- round_quantity() helper
"""

import uuid
from datetime import datetime, timedelta

import pytest

from app.schemas import IntentAction, PaperPosition, PaperState
from app.services.strategy.models import (
    EntryConfig,
    ExitConfig,
    ExecutionSpec,
    MarketSnapshot,
    OHLCVBar,
    RiskConfig,
)
from app.services.strategy.runner import StrategyRunner
from app.services.strategy.strategies import (
    evaluate_breakout_52w_high,
    round_quantity,
    QUANTITY_DECIMALS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workspace_id() -> uuid.UUID:
    """Provide a consistent workspace ID for tests."""
    return uuid.uuid4()


@pytest.fixture
def make_bars():
    """Factory to create OHLCV bars with specified highs."""

    def _make_bars(
        highs: list[float], base_ts: datetime = None, close: float = None
    ) -> list[OHLCVBar]:
        """
        Create bars with specified high values.

        Args:
            highs: List of high prices for each bar
            base_ts: Starting timestamp (defaults to now - len(highs) days)
            close: Close price for last bar (defaults to same as high)

        Returns:
            List of OHLCVBar with the specified highs
        """
        if base_ts is None:
            base_ts = datetime.utcnow()
        bars = []
        for i, high in enumerate(highs):
            ts = base_ts - timedelta(days=len(highs) - 1 - i)
            bar_close = close if (close is not None and i == len(highs) - 1) else high
            bars.append(
                OHLCVBar(
                    ts=ts,
                    open=high - 1.0,
                    high=high,
                    low=high - 2.0,
                    close=bar_close,
                    volume=10000.0,
                )
            )
        return bars

    return _make_bars


@pytest.fixture
def make_snapshot(make_bars):
    """Factory to create MarketSnapshot."""

    def _make_snapshot(
        symbol: str = "AAPL",
        highs: list[float] = None,
        last_price: float = None,
        high_52w: float = None,
        is_eod: bool = False,
    ) -> MarketSnapshot:
        """
        Create a market snapshot.

        Args:
            symbol: Symbol for snapshot
            highs: List of high prices for bars (default: [100, 102, 104])
            last_price: Override last price (else uses bars[-1].close)
            high_52w: Pre-computed 52w high (else computed from bars)
            is_eod: End-of-day flag

        Returns:
            MarketSnapshot with specified configuration
        """
        if highs is None:
            highs = [100.0, 102.0, 104.0]

        now = datetime.utcnow()
        bars = make_bars(highs, base_ts=now)

        return MarketSnapshot(
            symbol=symbol,
            ts=now,
            timeframe="daily",
            bars=bars,
            last_price=last_price,
            high_52w=high_52w,
            is_eod=is_eod,
        )

    return _make_snapshot


@pytest.fixture
def make_spec(workspace_id):
    """Factory to create ExecutionSpec."""

    def _make_spec(
        symbols: list[str] = None,
        dollars_per_trade: float = 1000.0,
        max_positions: int = 5,
        lookback_days: int = 252,
    ) -> ExecutionSpec:
        """
        Create an ExecutionSpec for testing.

        Args:
            symbols: List of symbols to trade (default: ["AAPL"])
            dollars_per_trade: Position sizing in dollars
            max_positions: Maximum concurrent positions
            lookback_days: Lookback for 52w high calculation

        Returns:
            ExecutionSpec with breakout_52w_high entry type
        """
        if symbols is None:
            symbols = ["AAPL"]

        return ExecutionSpec(
            strategy_id="breakout_52w_high",
            name="Test Breakout Strategy",
            workspace_id=workspace_id,
            symbols=symbols,
            timeframe="daily",
            entry=EntryConfig(type="breakout_52w_high", lookback_days=lookback_days),
            exit=ExitConfig(type="eod"),
            risk=RiskConfig(
                dollars_per_trade=dollars_per_trade, max_positions=max_positions
            ),
        )

    return _make_spec


@pytest.fixture
def make_paper_state(workspace_id):
    """Factory to create PaperState."""

    def _make_paper_state(
        positions: dict[str, tuple[float, float]] = None,
    ) -> PaperState:
        """
        Create a PaperState for testing.

        Args:
            positions: Dict of symbol -> (quantity, avg_price)

        Returns:
            PaperState with specified positions
        """
        state = PaperState(
            workspace_id=workspace_id,
            starting_equity=10000.0,
            cash=10000.0,
            realized_pnl=0.0,
            positions={},
        )

        if positions:
            for symbol, (qty, avg_price) in positions.items():
                state.positions[symbol] = PaperPosition(
                    workspace_id=workspace_id,
                    symbol=symbol,
                    side="long" if qty > 0 else None,
                    quantity=qty,
                    avg_price=avg_price,
                )

        return state

    return _make_paper_state


# =============================================================================
# round_quantity() Tests
# =============================================================================


class TestRoundQuantity:
    """Tests for round_quantity helper function."""

    def test_floors_correctly_8_decimals(self):
        """round_quantity floors to 8 decimal places by default."""
        # 0.123456789 should floor to 0.12345678
        result = round_quantity(0.123456789)
        assert result == 0.12345678

    def test_floors_correctly_custom_decimals(self):
        """round_quantity respects custom decimal places."""
        # 0.12999 with 2 decimals should floor to 0.12
        result = round_quantity(0.12999, decimals=2)
        assert result == 0.12

    def test_exact_value_unchanged(self):
        """Exact values within precision are unchanged."""
        result = round_quantity(0.12345678)
        assert result == 0.12345678

    def test_zero_stays_zero(self):
        """Zero quantity stays zero."""
        result = round_quantity(0.0)
        assert result == 0.0

    def test_very_small_qty_floors_to_zero(self):
        """Very small quantities floor to zero."""
        # 0.000000001 (9 decimal places) floors to 0 with 8 decimals
        result = round_quantity(0.000000001)
        assert result == 0.0

    def test_default_decimals_is_8(self):
        """Verify QUANTITY_DECIMALS constant is 8."""
        assert QUANTITY_DECIMALS == 8


# =============================================================================
# evaluate_breakout_52w_high() Direct Tests
# =============================================================================


class TestBreakoutStrategyDirect:
    """Direct tests for evaluate_breakout_52w_high function."""

    def test_division_by_zero_guard_zero_price(
        self, make_spec, make_snapshot, make_paper_state
    ):
        """Returns early with signal when last_price <= 0."""
        spec = make_spec()
        # Create snapshot with zero close price
        snapshot = make_snapshot(highs=[100.0, 102.0, 0.0], last_price=0.0)
        paper_state = make_paper_state()
        eval_id = uuid.uuid4()

        result = evaluate_breakout_52w_high(
            spec, snapshot, paper_state, eval_id, at_max_positions=False
        )

        assert len(result.intents) == 0
        assert "entry_skipped_invalid_price" in result.signals
        assert result.metadata["last_price"] == 0.0

    def test_division_by_zero_guard_negative_price(
        self, make_spec, make_snapshot, make_paper_state
    ):
        """Returns early with signal when last_price is negative."""
        spec = make_spec()
        snapshot = make_snapshot(highs=[100.0, 102.0, -1.0], last_price=-1.0)
        paper_state = make_paper_state()
        eval_id = uuid.uuid4()

        result = evaluate_breakout_52w_high(
            spec, snapshot, paper_state, eval_id, at_max_positions=False
        )

        assert len(result.intents) == 0
        assert "entry_skipped_invalid_price" in result.signals

    def test_no_history_guard(self, make_spec, make_paper_state, workspace_id):
        """Returns early when no prior bars and no pre-computed high_52w."""
        # With lookback_days=0, bars[-(0+1):-1] = bars[-1:-1] = empty slice
        spec = make_spec(lookback_days=0)
        paper_state = make_paper_state()
        eval_id = uuid.uuid4()

        # Create snapshot with 2 bars (minimum required)
        now = datetime.utcnow()
        bars = [
            OHLCVBar(
                ts=now - timedelta(days=1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=10000.0,
            ),
            OHLCVBar(
                ts=now,
                open=102.0,
                high=110.0,
                low=101.0,
                close=108.0,
                volume=10000.0,
            ),
        ]
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=now,
            timeframe="daily",
            bars=bars,
            # No high_52w provided, and with lookback=0, prior_bars slice is empty
        )

        result = evaluate_breakout_52w_high(
            spec, snapshot, paper_state, eval_id, at_max_positions=False
        )

        assert len(result.intents) == 0
        assert "entry_skipped_no_history" in result.signals

    def test_partial_lookback_signal(self, make_spec, make_paper_state, workspace_id):
        """Signals when available bars < lookback_days."""
        spec = make_spec(lookback_days=252)  # Standard 52 weeks
        paper_state = make_paper_state()
        eval_id = uuid.uuid4()

        # Create snapshot with only 5 bars (4 prior, 1 current)
        now = datetime.utcnow()
        bars = [
            OHLCVBar(
                ts=now - timedelta(days=i),
                open=100.0 + i,
                high=105.0 + i,
                low=99.0 + i,
                close=102.0 + i,
                volume=10000.0,
            )
            for i in range(4, -1, -1)  # Days 4, 3, 2, 1, 0
        ]
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=now,
            timeframe="daily",
            bars=bars,
            last_price=95.0,  # Below high, so no entry
        )

        result = evaluate_breakout_52w_high(
            spec, snapshot, paper_state, eval_id, at_max_positions=False
        )

        # Should have a partial lookback signal
        partial_signals = [
            s for s in result.signals if s.startswith("lookback_partial")
        ]
        assert len(partial_signals) == 1
        assert "4_of_252" in partial_signals[0]


# =============================================================================
# StrategyRunner.evaluate() Tests
# =============================================================================


class TestStrategyRunner:
    """Tests for StrategyRunner.evaluate() method."""

    @pytest.fixture
    def runner(self):
        """Create a fresh StrategyRunner."""
        return StrategyRunner()

    # -------------------------------------------------------------------------
    # Entry Tests
    # -------------------------------------------------------------------------

    def test_no_entry_price_below_52w_high(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """Price below 52w high produces no intents."""
        spec = make_spec()
        # Prior bars have highs [100, 102], current bar high is 104
        # 52w high from prior bars = 102
        # Set last_price = 101 (below 102)
        snapshot = make_snapshot(highs=[100.0, 102.0, 101.0], last_price=101.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 0
        # No breakout signal since price is below 52w high
        assert "breakout_entry_triggered" not in result.signals

    def test_entry_triggered_price_above_52w_high(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """Price above 52w high with no position triggers OPEN_LONG intent."""
        spec = make_spec(dollars_per_trade=1000.0)
        # Prior bars have highs [100, 102], current bar closes at 105
        # 52w high from prior = 102, price = 105 > 102
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 1
        intent = result.intents[0]
        assert intent.action == IntentAction.OPEN_LONG
        assert intent.symbol == "AAPL"
        assert "breakout_entry_triggered" in result.signals

    def test_no_duplicate_entry_with_existing_position(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """No entry when already holding a position."""
        spec = make_spec()
        # Breakout condition met
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        # Already have a position
        paper_state = make_paper_state(positions={"AAPL": (10.0, 100.0)})

        result = runner.evaluate(spec, snapshot, paper_state)

        # No entry intent because we already have position
        entry_intents = [
            i for i in result.intents if i.action == IntentAction.OPEN_LONG
        ]
        assert len(entry_intents) == 0

    # -------------------------------------------------------------------------
    # Exit Tests
    # -------------------------------------------------------------------------

    def test_eod_exit_with_position(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """EOD with position triggers CLOSE_LONG intent."""
        spec = make_spec()
        snapshot = make_snapshot(highs=[100.0, 102.0, 101.0], is_eod=True)
        paper_state = make_paper_state(positions={"AAPL": (10.0, 100.0)})

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 1
        intent = result.intents[0]
        assert intent.action == IntentAction.CLOSE_LONG
        assert intent.quantity == 10.0  # Full position close
        assert "eod_exit_triggered" in result.signals

    def test_no_exit_without_position(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """EOD with no position produces no exit intent."""
        spec = make_spec()
        snapshot = make_snapshot(highs=[100.0, 102.0, 101.0], is_eod=True)
        paper_state = make_paper_state()  # No positions

        result = runner.evaluate(spec, snapshot, paper_state)

        # No exit intent (no position to close)
        exit_intents = [
            i for i in result.intents if i.action == IntentAction.CLOSE_LONG
        ]
        assert len(exit_intents) == 0

    # -------------------------------------------------------------------------
    # Quantity Calculation Tests
    # -------------------------------------------------------------------------

    def test_quantity_calculation(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """Quantity = dollars_per_trade / price, rounded to 8 decimals."""
        spec = make_spec(dollars_per_trade=1000.0)
        # Price = 105, qty = 1000 / 105 = 9.52380952...
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 1
        intent = result.intents[0]
        expected_qty = round_quantity(1000.0 / 105.0)
        assert intent.quantity == expected_qty
        assert intent.quantity == 9.52380952  # Verify the actual value

    def test_zero_qty_skipped(self, runner, make_spec, make_snapshot, make_paper_state):
        """No entry intent when rounded quantity <= 0."""
        # Very high price results in very small qty that rounds to 0
        spec = make_spec(dollars_per_trade=0.000000001)  # Tiny allocation
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        # No intent emitted due to zero qty
        assert len(result.intents) == 0
        assert "entry_skipped_zero_qty" in result.signals

    # -------------------------------------------------------------------------
    # Max Positions Tests
    # -------------------------------------------------------------------------

    def test_max_positions_blocks_entry(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """At max_positions, entry is blocked but signal is added."""
        spec = make_spec(max_positions=2)
        # Breakout condition met
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        # Already at max positions (2 positions, max=2)
        paper_state = make_paper_state(
            positions={
                "MSFT": (10.0, 200.0),
                "GOOGL": (5.0, 150.0),
            }
        )

        result = runner.evaluate(spec, snapshot, paper_state)

        # Entry should be blocked
        entry_intents = [
            i for i in result.intents if i.action == IntentAction.OPEN_LONG
        ]
        assert len(entry_intents) == 0
        # Should have signal explaining why blocked
        assert "entry_blocked_max_positions" in result.signals

    def test_max_positions_allows_exit(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """At max_positions + has position + is_eod, exit is allowed."""
        # EOD flag set
        snapshot = make_snapshot(
            symbol="MSFT", highs=[100.0, 102.0, 101.0], is_eod=True
        )
        # At max positions, including MSFT
        paper_state = make_paper_state(
            positions={
                "MSFT": (10.0, 200.0),
                "GOOGL": (5.0, 150.0),
            }
        )

        result = runner.evaluate(
            make_spec(symbols=["MSFT"], max_positions=2),
            snapshot,
            paper_state,
        )

        # Exit should be allowed even at max positions
        exit_intents = [
            i for i in result.intents if i.action == IntentAction.CLOSE_LONG
        ]
        assert len(exit_intents) == 1
        assert exit_intents[0].symbol == "MSFT"
        assert "eod_exit_triggered" in result.signals

    # -------------------------------------------------------------------------
    # Symbol Filtering Tests
    # -------------------------------------------------------------------------

    def test_symbol_filtering(self, runner, make_spec, make_snapshot, make_paper_state):
        """Snapshot symbol not in spec.symbols produces no intent."""
        spec = make_spec(symbols=["AAPL", "GOOGL"])  # MSFT not in list
        snapshot = make_snapshot(symbol="MSFT", highs=[100.0, 102.0, 105.0])
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 0
        assert "symbol_not_in_spec" in result.signals

    # -------------------------------------------------------------------------
    # Intent Fields Tests
    # -------------------------------------------------------------------------

    def test_intent_fields_complete(
        self, runner, make_spec, make_snapshot, make_paper_state, workspace_id
    ):
        """All required TradeIntent fields are populated correctly."""
        spec = make_spec(dollars_per_trade=1000.0)
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 1
        intent = result.intents[0]

        # Check all required fields
        assert intent.id is not None
        assert intent.correlation_id is not None
        assert intent.workspace_id == workspace_id
        assert intent.action == IntentAction.OPEN_LONG
        assert intent.strategy_entity_id == spec.instance_id
        assert intent.symbol == "AAPL"
        assert intent.timeframe == "daily"
        assert intent.quantity > 0
        assert "Breakout:" in intent.reason

    def test_shared_correlation_id(
        self, runner, make_spec, make_paper_state, workspace_id
    ):
        """All intents from one evaluation share same correlation_id (evaluation_id)."""
        # This test would require a scenario with multiple intents
        # For breakout_52w_high, we can only get one intent at a time
        # But we can verify the evaluation_id is used as correlation_id
        spec = make_spec(dollars_per_trade=1000.0)
        now = datetime.utcnow()
        bars = [
            OHLCVBar(
                ts=now - timedelta(days=1),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=10000.0,
            ),
            OHLCVBar(
                ts=now,
                open=101.0,
                high=105.0,
                low=100.0,
                close=105.0,
                volume=10000.0,
            ),
        ]
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=now,
            timeframe="daily",
            bars=bars,
            last_price=105.0,
        )
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert len(result.intents) == 1
        # Verify correlation_id matches evaluation_id
        assert result.intents[0].correlation_id == str(result.evaluation_id)

    # -------------------------------------------------------------------------
    # 52w High Calculation Tests
    # -------------------------------------------------------------------------

    def test_prior_bars_only_excludes_current_bar(
        self, runner, make_spec, make_paper_state, workspace_id
    ):
        """52w high computed from prior bars only, excluding current bar."""
        spec = make_spec(lookback_days=5)
        now = datetime.utcnow()

        # Create bars where current bar has highest high
        # Prior bars: highs = [100, 102, 103, 104]
        # Current bar: high = 110 (should be excluded from 52w high calc)
        bars = [
            OHLCVBar(
                ts=now - timedelta(days=4),
                open=99,
                high=100.0,
                low=98,
                close=99.5,
                volume=1000,
            ),
            OHLCVBar(
                ts=now - timedelta(days=3),
                open=100,
                high=102.0,
                low=99,
                close=101,
                volume=1000,
            ),
            OHLCVBar(
                ts=now - timedelta(days=2),
                open=101,
                high=103.0,
                low=100,
                close=102,
                volume=1000,
            ),
            OHLCVBar(
                ts=now - timedelta(days=1),
                open=102,
                high=104.0,
                low=101,
                close=103,
                volume=1000,
            ),
            OHLCVBar(
                ts=now, open=103, high=110.0, low=102, close=105, volume=1000
            ),  # Current
        ]
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=now,
            timeframe="daily",
            bars=bars,
            last_price=105.0,  # Above prior 52w high of 104
        )
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        # 52w high should be 104 (from prior bars), not 110 (current bar)
        assert result.metadata["high_52w"] == 104.0
        # Entry should trigger since 105 > 104
        assert len(result.intents) == 1
        assert "breakout_entry_triggered" in result.signals

    def test_uses_precomputed_high_52w_if_provided(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """Uses snapshot.high_52w if provided instead of computing from bars."""
        spec = make_spec()
        # Provide pre-computed high_52w that differs from bar highs
        snapshot = make_snapshot(
            highs=[100.0, 102.0, 105.0],
            last_price=105.0,
            high_52w=110.0,  # Pre-computed, higher than bar highs
        )
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        # Should use pre-computed value
        assert result.metadata["high_52w"] == 110.0
        # No breakout since 105 < 110
        assert len(result.intents) == 0

    # -------------------------------------------------------------------------
    # Unknown Strategy Type Test
    # -------------------------------------------------------------------------

    def test_unknown_strategy_type_raises_valueerror(
        self, runner, make_paper_state, make_snapshot, workspace_id
    ):
        """Unknown strategy type raises ValueError."""
        # Create spec with unregistered entry type
        spec = ExecutionSpec(
            strategy_id="unknown_strategy",
            name="Unknown Strategy",
            workspace_id=workspace_id,
            symbols=["AAPL"],
            timeframe="daily",
            entry=EntryConfig(type="nonexistent_entry_type"),
            exit=ExitConfig(type="eod"),
            risk=RiskConfig(dollars_per_trade=1000.0),
        )
        snapshot = make_snapshot()
        paper_state = make_paper_state()

        with pytest.raises(ValueError) as exc_info:
            runner.evaluate(spec, snapshot, paper_state)

        assert "Unknown strategy type" in str(exc_info.value)
        assert "nonexistent_entry_type" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # Signal Explanation Tests
    # -------------------------------------------------------------------------

    def test_signals_provide_explanations(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """Signals list explains each decision made."""
        spec = make_spec()
        # Breakout triggered
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        # Should have breakout signal explaining the entry
        assert len(result.signals) >= 1
        assert "breakout_entry_triggered" in result.signals

    def test_metadata_contains_debug_info(
        self, runner, make_spec, make_snapshot, make_paper_state
    ):
        """Metadata contains debug information (52w_high, price, etc.)."""
        spec = make_spec()
        snapshot = make_snapshot(highs=[100.0, 102.0, 105.0], last_price=105.0)
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        # Metadata should contain useful debug info
        assert "high_52w" in result.metadata
        assert "last_price" in result.metadata
        assert "at_max_positions" in result.metadata
        assert result.metadata["last_price"] == 105.0

    # -------------------------------------------------------------------------
    # Snapshot Validation Tests
    # -------------------------------------------------------------------------

    def test_snapshot_validation_insufficient_bars(self):
        """MarketSnapshot with <2 bars raises ValueError."""
        now = datetime.utcnow()

        # Create snapshot with only 1 bar (less than minimum 2)
        bars = [
            OHLCVBar(
                ts=now,
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=10000.0,
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            MarketSnapshot(
                symbol="AAPL",
                ts=now,
                timeframe="daily",
                bars=bars,
            )

        assert "at least 2 bars" in str(exc_info.value)


# =============================================================================
# Runner Registration Tests
# =============================================================================


class TestStrategyRunnerRegistration:
    """Tests for strategy registration functionality."""

    def test_default_strategies_registered(self):
        """Default strategies are registered on init."""
        runner = StrategyRunner()
        # breakout_52w_high should be registered by default
        assert "breakout_52w_high" in runner._strategies

    def test_register_custom_strategy(self):
        """Can register custom strategy functions."""
        runner = StrategyRunner()

        def custom_strategy(spec, snapshot, paper_state, eval_id, at_max):
            from app.services.strategy.models import StrategyEvaluation

            return StrategyEvaluation(
                spec_id=str(spec.instance_id),
                symbol=snapshot.symbol,
                ts=snapshot.ts,
                intents=[],
                signals=["custom_signal"],
                evaluation_id=eval_id,
            )

        runner.register_strategy("custom_type", custom_strategy)
        assert "custom_type" in runner._strategies

    def test_custom_strategy_invoked(
        self, make_snapshot, make_paper_state, workspace_id
    ):
        """Registered custom strategy is invoked correctly."""
        runner = StrategyRunner()
        invoked = {"count": 0}

        def custom_strategy(spec, snapshot, paper_state, eval_id, at_max):
            from app.services.strategy.models import StrategyEvaluation

            invoked["count"] += 1
            return StrategyEvaluation(
                spec_id=str(spec.instance_id),
                symbol=snapshot.symbol,
                ts=snapshot.ts,
                intents=[],
                signals=["custom_invoked"],
                evaluation_id=eval_id,
            )

        runner.register_strategy("custom_type", custom_strategy)

        spec = ExecutionSpec(
            strategy_id="custom",
            name="Custom Strategy",
            workspace_id=workspace_id,
            symbols=["AAPL"],
            timeframe="daily",
            entry=EntryConfig(type="custom_type"),
            exit=ExitConfig(type="eod"),
            risk=RiskConfig(dollars_per_trade=1000.0),
        )
        snapshot = make_snapshot()
        paper_state = make_paper_state()

        result = runner.evaluate(spec, snapshot, paper_state)

        assert invoked["count"] == 1
        assert "custom_invoked" in result.signals
