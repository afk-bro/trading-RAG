"""Unit tests for strategy runner models."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas import TradeIntent, IntentAction
from app.services.strategy.models import (
    OHLCVBar,
    MarketSnapshot,
    EntryConfig,
    ExitConfig,
    RiskConfig,
    ExecutionSpec,
    StrategyEvaluation,
)


# =============================================================================
# OHLCVBar Tests
# =============================================================================


class TestOHLCVBar:
    """Tests for OHLCVBar model."""

    def test_valid_bar(self):
        """Create a valid OHLCV bar."""
        bar = OHLCVBar(
            ts=datetime.now(timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=10000.0,
        )
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 103.0
        assert bar.volume == 10000.0

    def test_naive_ts_rejected(self):
        """Naive datetime is rejected by the tz-aware validator."""
        with pytest.raises(ValidationError, match="timezone-aware"):
            OHLCVBar(
                ts=datetime(2024, 1, 1),  # naive
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=10000.0,
            )

    def test_non_utc_tz_accepted(self):
        """Non-UTC tz-aware datetime is accepted (converted at usage site)."""
        from zoneinfo import ZoneInfo

        bar = OHLCVBar(
            ts=datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York")),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=10000.0,
        )
        assert bar.ts.tzinfo is not None

    def test_missing_required_fields(self):
        """All OHLCV fields are required."""
        with pytest.raises(ValidationError) as exc_info:
            OHLCVBar(ts=datetime.now(timezone.utc), open=100.0)

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert "high" in missing_fields
        assert "low" in missing_fields
        assert "close" in missing_fields
        assert "volume" in missing_fields


# =============================================================================
# MarketSnapshot Tests
# =============================================================================


class TestMarketSnapshot:
    """Tests for MarketSnapshot model validation."""

    @pytest.fixture
    def sample_bars(self) -> list[OHLCVBar]:
        """Create sample bars for testing."""
        now = datetime.now(timezone.utc)
        return [
            OHLCVBar(
                ts=now - timedelta(days=2),
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=10000.0,
            ),
            OHLCVBar(
                ts=now - timedelta(days=1),
                open=102.0,
                high=108.0,
                low=101.0,
                close=107.0,
                volume=12000.0,
            ),
            OHLCVBar(
                ts=now,
                open=107.0,
                high=110.0,
                low=106.0,
                close=109.0,
                volume=11000.0,
            ),
        ]

    def test_valid_snapshot(self, sample_bars):
        """Create a valid market snapshot."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=now,
            timeframe="daily",
            bars=sample_bars,
        )
        assert snapshot.symbol == "AAPL"
        assert snapshot.timeframe == "daily"
        assert len(snapshot.bars) == 3
        assert snapshot.is_eod is False
        assert snapshot.last_price is None
        assert snapshot.high_52w is None
        assert snapshot.low_52w is None

    def test_snapshot_with_optional_fields(self, sample_bars):
        """Create snapshot with all optional fields populated."""
        now = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=now,
            timeframe="daily",
            bars=sample_bars,
            last_price=109.0,
            high_52w=150.0,
            low_52w=80.0,
            is_eod=True,
        )
        assert snapshot.last_price == 109.0
        assert snapshot.high_52w == 150.0
        assert snapshot.low_52w == 80.0
        assert snapshot.is_eod is True

    def test_requires_at_least_two_bars(self):
        """Validation fails with fewer than 2 bars."""
        now = datetime.now(timezone.utc)
        single_bar = [
            OHLCVBar(
                ts=now,
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=10000.0,
            )
        ]
        with pytest.raises(ValidationError) as exc_info:
            MarketSnapshot(
                symbol="AAPL",
                ts=now,
                timeframe="daily",
                bars=single_bar,
            )
        assert "at least 2 bars" in str(exc_info.value)

    def test_requires_at_least_two_bars_empty(self):
        """Validation fails with empty bars list."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError) as exc_info:
            MarketSnapshot(
                symbol="AAPL",
                ts=now,
                timeframe="daily",
                bars=[],
            )
        assert "at least 2 bars" in str(exc_info.value)

    def test_bar_timestamp_cannot_exceed_snapshot_ts(self):
        """Latest bar timestamp cannot be after snapshot timestamp."""
        now = datetime.now(timezone.utc)
        future_bar = now + timedelta(hours=1)

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
                ts=future_bar,  # This is in the future
                open=102.0,
                high=108.0,
                low=101.0,
                close=107.0,
                volume=12000.0,
            ),
        ]

        with pytest.raises(ValidationError) as exc_info:
            MarketSnapshot(
                symbol="AAPL",
                ts=now,
                timeframe="daily",
                bars=bars,
            )
        assert "cannot exceed snapshot ts" in str(exc_info.value)

    def test_bar_timestamp_can_equal_snapshot_ts(self, sample_bars):
        """Bar timestamp can equal snapshot timestamp (edge case)."""
        # Use the last bar's timestamp as snapshot ts
        snapshot_ts = sample_bars[-1].ts
        snapshot = MarketSnapshot(
            symbol="AAPL",
            ts=snapshot_ts,
            timeframe="daily",
            bars=sample_bars,
        )
        assert snapshot.bars[-1].ts == snapshot.ts


# =============================================================================
# Config Models Tests
# =============================================================================


class TestEntryConfig:
    """Tests for EntryConfig model."""

    def test_valid_entry_config(self):
        """Create a valid entry config."""
        config = EntryConfig(type="breakout_52w_high")
        assert config.type == "breakout_52w_high"
        assert config.lookback_days == 252  # default

    def test_custom_lookback_days(self):
        """Override default lookback days."""
        config = EntryConfig(type="breakout_52w_high", lookback_days=200)
        assert config.lookback_days == 200

    def test_missing_type(self):
        """Type field is required."""
        with pytest.raises(ValidationError):
            EntryConfig()


class TestExitConfig:
    """Tests for ExitConfig model."""

    def test_valid_exit_config(self):
        """Create a valid exit config."""
        config = ExitConfig(type="eod")
        assert config.type == "eod"

    def test_missing_type(self):
        """Type field is required."""
        with pytest.raises(ValidationError):
            ExitConfig()


class TestRiskConfig:
    """Tests for RiskConfig model."""

    def test_valid_risk_config(self):
        """Create a valid risk config."""
        config = RiskConfig(dollars_per_trade=1000.0)
        assert config.dollars_per_trade == 1000.0
        assert config.max_positions == 5  # default

    def test_custom_max_positions(self):
        """Override default max positions."""
        config = RiskConfig(dollars_per_trade=1000.0, max_positions=10)
        assert config.max_positions == 10

    def test_dollars_per_trade_required(self):
        """dollars_per_trade is required."""
        with pytest.raises(ValidationError):
            RiskConfig(max_positions=5)

    def test_dollars_per_trade_must_be_positive(self):
        """dollars_per_trade must be greater than 0."""
        with pytest.raises(ValidationError) as exc_info:
            RiskConfig(dollars_per_trade=0.0)
        assert "greater than 0" in str(exc_info.value)

    def test_dollars_per_trade_negative_fails(self):
        """Negative dollars_per_trade fails validation."""
        with pytest.raises(ValidationError):
            RiskConfig(dollars_per_trade=-100.0)

    def test_max_positions_at_least_one(self):
        """max_positions must be at least 1."""
        with pytest.raises(ValidationError):
            RiskConfig(dollars_per_trade=1000.0, max_positions=0)


# =============================================================================
# ExecutionSpec Tests
# =============================================================================


class TestExecutionSpec:
    """Tests for ExecutionSpec model."""

    @pytest.fixture
    def valid_spec_data(self) -> dict:
        """Base data for a valid ExecutionSpec."""
        return {
            "strategy_id": "breakout_52w_high",
            "name": "Breakout Strategy - Tech Stocks",
            "workspace_id": uuid.uuid4(),
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "timeframe": "daily",
            "entry": {"type": "breakout_52w_high", "lookback_days": 252},
            "exit": {"type": "eod"},
            "risk": {"dollars_per_trade": 1000.0, "max_positions": 5},
        }

    def test_valid_execution_spec(self, valid_spec_data):
        """Create a valid execution spec."""
        spec = ExecutionSpec(**valid_spec_data)

        assert spec.strategy_id == "breakout_52w_high"
        assert spec.name == "Breakout Strategy - Tech Stocks"
        assert len(spec.symbols) == 3
        assert spec.timeframe == "daily"
        assert spec.entry.type == "breakout_52w_high"
        assert spec.exit.type == "eod"
        assert spec.risk.dollars_per_trade == 1000.0
        assert spec.enabled is True  # default

    def test_instance_id_auto_generated(self, valid_spec_data):
        """instance_id is auto-generated if not provided."""
        spec = ExecutionSpec(**valid_spec_data)
        assert spec.instance_id is not None
        assert isinstance(spec.instance_id, uuid.UUID)

    def test_different_specs_get_different_instance_ids(self, valid_spec_data):
        """Each spec gets a unique instance_id."""
        spec1 = ExecutionSpec(**valid_spec_data)
        spec2 = ExecutionSpec(**valid_spec_data)
        assert spec1.instance_id != spec2.instance_id

    def test_explicit_instance_id(self, valid_spec_data):
        """Can provide explicit instance_id."""
        explicit_id = uuid.uuid4()
        valid_spec_data["instance_id"] = explicit_id
        spec = ExecutionSpec(**valid_spec_data)
        assert spec.instance_id == explicit_id

    def test_created_at_defaults_to_now(self, valid_spec_data):
        """created_at defaults to current time."""
        before = datetime.now(timezone.utc)
        spec = ExecutionSpec(**valid_spec_data)
        after = datetime.now(timezone.utc)

        assert before <= spec.created_at <= after

    def test_symbols_required_nonempty(self, valid_spec_data):
        """symbols list cannot be empty."""
        valid_spec_data["symbols"] = []
        with pytest.raises(ValidationError) as exc_info:
            ExecutionSpec(**valid_spec_data)
        # Check for min_length validation
        assert "symbols" in str(exc_info.value).lower()

    def test_disabled_spec(self, valid_spec_data):
        """Spec can be disabled."""
        valid_spec_data["enabled"] = False
        spec = ExecutionSpec(**valid_spec_data)
        assert spec.enabled is False


# =============================================================================
# StrategyEvaluation Tests
# =============================================================================


class TestStrategyEvaluation:
    """Tests for StrategyEvaluation model."""

    @pytest.fixture
    def sample_intent(self) -> TradeIntent:
        """Create a sample trade intent."""
        return TradeIntent(
            correlation_id="test-correlation",
            workspace_id=uuid.uuid4(),
            action=IntentAction.OPEN_LONG,
            strategy_entity_id=uuid.uuid4(),
            symbol="AAPL",
            timeframe="daily",
            quantity=10,
        )

    def test_valid_evaluation_no_intents(self):
        """Create evaluation with no intents (no signal)."""
        eval_result = StrategyEvaluation(
            spec_id=str(uuid.uuid4()),
            symbol="AAPL",
            ts=datetime.now(timezone.utc),
        )

        assert len(eval_result.intents) == 0
        assert len(eval_result.signals) == 0
        assert eval_result.metadata == {}

    def test_valid_evaluation_with_intent(self, sample_intent):
        """Create evaluation with an intent."""
        eval_result = StrategyEvaluation(
            spec_id=str(uuid.uuid4()),
            symbol="AAPL",
            ts=datetime.now(timezone.utc),
            intents=[sample_intent],
            signals=["52-week high breakout detected"],
            metadata={"52w_high": 150.0, "current_price": 152.0},
        )

        assert len(eval_result.intents) == 1
        assert eval_result.intents[0].action == IntentAction.OPEN_LONG
        assert "52-week high breakout detected" in eval_result.signals
        assert eval_result.metadata["52w_high"] == 150.0

    def test_evaluation_id_auto_generated(self):
        """evaluation_id is auto-generated."""
        eval1 = StrategyEvaluation(
            spec_id=str(uuid.uuid4()),
            symbol="AAPL",
            ts=datetime.now(timezone.utc),
        )
        eval2 = StrategyEvaluation(
            spec_id=str(uuid.uuid4()),
            symbol="AAPL",
            ts=datetime.now(timezone.utc),
        )

        assert eval1.evaluation_id is not None
        assert eval2.evaluation_id is not None
        assert eval1.evaluation_id != eval2.evaluation_id

    def test_multiple_intents(self, sample_intent):
        """Evaluation can contain multiple intents."""
        # Create second intent (e.g., close existing + open new)
        close_intent = TradeIntent(
            correlation_id="test-correlation",
            workspace_id=sample_intent.workspace_id,
            action=IntentAction.CLOSE_LONG,
            strategy_entity_id=sample_intent.strategy_entity_id,
            symbol="MSFT",
            timeframe="daily",
        )

        eval_result = StrategyEvaluation(
            spec_id=str(uuid.uuid4()),
            symbol="AAPL",
            ts=datetime.now(timezone.utc),
            intents=[sample_intent, close_intent],
            signals=["Open AAPL", "Close MSFT"],
        )

        assert len(eval_result.intents) == 2
        assert len(eval_result.signals) == 2


# =============================================================================
# Import Tests
# =============================================================================


class TestPackageImports:
    """Test that the package exports work correctly."""

    def test_import_from_package(self):
        """All models can be imported from the package."""
        from app.services.strategy import (
            OHLCVBar,
            MarketSnapshot,
            EntryConfig,
            ExitConfig,
            RiskConfig,
            ExecutionSpec,
            StrategyEvaluation,
        )

        # Just verify imports work
        assert OHLCVBar is not None
        assert MarketSnapshot is not None
        assert EntryConfig is not None
        assert ExitConfig is not None
        assert RiskConfig is not None
        assert ExecutionSpec is not None
        assert StrategyEvaluation is not None


# =============================================================================
# UnicornConfig bar-quality guard validation
# =============================================================================


class TestUnicornConfigGuardValidation:
    """UnicornConfig must validate bar-quality guard parameters."""

    def test_max_wick_ratio_rejects_invalid(self):
        """max_wick_ratio=1.5 is out of (0.0, 1.0] range."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        with pytest.raises(ValueError, match="max_wick_ratio must be in"):
            UnicornConfig(max_wick_ratio=1.5)

    def test_max_wick_ratio_rejects_zero(self):
        """max_wick_ratio=0.0 is out of (0.0, 1.0] range (exclusive lower bound)."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        with pytest.raises(ValueError, match="max_wick_ratio must be in"):
            UnicornConfig(max_wick_ratio=0.0)

    def test_max_wick_ratio_accepts_none(self):
        """Default None is valid (guard disabled)."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        config = UnicornConfig()
        assert config.max_wick_ratio is None

    def test_max_wick_ratio_accepts_one(self):
        """max_wick_ratio=1.0 is the upper boundary, should be accepted."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        config = UnicornConfig(max_wick_ratio=1.0)
        assert config.max_wick_ratio == 1.0

    def test_max_range_atr_mult_rejects_zero(self):
        """max_range_atr_mult=0 must raise ValueError."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        with pytest.raises(ValueError, match="max_range_atr_mult must be > 0"):
            UnicornConfig(max_range_atr_mult=0)

    def test_max_range_atr_mult_rejects_negative(self):
        """max_range_atr_mult=-1.0 must raise ValueError."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        with pytest.raises(ValueError, match="max_range_atr_mult must be > 0"):
            UnicornConfig(max_range_atr_mult=-1.0)

    def test_max_range_atr_mult_accepts_none(self):
        """Default None is valid (guard disabled)."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        config = UnicornConfig()
        assert config.max_range_atr_mult is None

    def test_max_range_atr_mult_accepts_positive(self):
        """max_range_atr_mult=3.0 should be accepted."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        config = UnicornConfig(max_range_atr_mult=3.0)
        assert config.max_range_atr_mult == 3.0

    def test_min_displacement_atr_rejects_zero(self):
        """min_displacement_atr=0 must raise ValueError."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        with pytest.raises(ValueError, match="min_displacement_atr must be > 0"):
            UnicornConfig(min_displacement_atr=0)

    def test_min_displacement_atr_rejects_negative(self):
        """min_displacement_atr=-0.5 must raise ValueError."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        with pytest.raises(ValueError, match="min_displacement_atr must be > 0"):
            UnicornConfig(min_displacement_atr=-0.5)

    def test_min_displacement_atr_accepts_none(self):
        """Default None is valid (guard disabled)."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        config = UnicornConfig()
        assert config.min_displacement_atr is None

    def test_min_displacement_atr_accepts_positive(self):
        """min_displacement_atr=0.5 should be accepted."""
        from app.services.strategy.strategies.unicorn_model import UnicornConfig

        config = UnicornConfig(min_displacement_atr=0.5)
        assert config.min_displacement_atr == 0.5

    def test_session_profile_ny_open_valid(self):
        """UnicornConfig accepts SessionProfile.NY_OPEN without raising."""
        from app.services.strategy.strategies.unicorn_model import (
            UnicornConfig,
            SessionProfile,
        )

        config = UnicornConfig(session_profile=SessionProfile.NY_OPEN)
        assert config.session_profile == SessionProfile.NY_OPEN
