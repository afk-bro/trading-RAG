"""
Unit tests for sweep closure confirmation gate (recency + settlement).

Tests the max_sweep_age_bars and require_sweep_settlement config knobs
added to UnicornConfig and enforced in check_criteria().
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.services.strategy.models import OHLCVBar
from app.services.strategy.strategies.unicorn_model import UnicornConfig
from app.services.strategy.indicators.ict_patterns import LiquiditySweep
from app.services.strategy.indicators.tf_bias import BiasDirection
from app.services.backtest.engines.unicorn_runner import (
    check_criteria,
)


def _make_bar(
    idx: int, open_: float, high: float, low: float, close: float
) -> OHLCVBar:
    """Helper: create a bar at index-derived timestamp during NY AM macro window."""
    # Use 9:30 ET = 14:30 UTC as base, increment by minutes wrapping into hours
    base_hour = 14
    total_minutes = 30 + idx
    hour = base_hour + total_minutes // 60
    minute = total_minutes % 60
    ts = datetime(2024, 6, 3, hour, minute, tzinfo=timezone.utc)
    return OHLCVBar(ts=ts, open=open_, high=high, low=low, close=close, volume=1000.0)


def _make_bars(n: int = 60) -> list[OHLCVBar]:
    """Create N bars of flat price action at 100.0."""
    return [_make_bar(i % 60, 100.0, 101.0, 99.0, 100.0) for i in range(n)]


def _make_sweep(
    bar_index: int, sweep_type: str = "low", swept_level: float = 99.0
) -> LiquiditySweep:
    ts = datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc)
    return LiquiditySweep(
        sweep_type=sweep_type,
        swept_level=swept_level,
        sweep_high=101.0,
        sweep_low=98.0,
        bar_index=bar_index,
        timestamp=ts,
        reversal_strength=0.5,
    )


# Patch path for detect_liquidity_sweeps called from unicorn_runner
SWEEP_PATCH = "app.services.backtest.engines.unicorn_runner.detect_liquidity_sweeps"
# Patch bias to always return BULLISH so sweep logic is reached
BIAS_PATCH = "app.services.backtest.engines.unicorn_runner.compute_tf_bias"


def _mock_bias(*args, **kwargs):
    """Return a bullish bias so check_criteria reaches the sweep block."""
    from app.services.strategy.indicators.tf_bias import (
        TimeframeBias,
        TimeframeBiasComponent,
        BiasStrength,
    )

    m15 = TimeframeBiasComponent(
        timeframe="m15",
        direction=BiasDirection.BULLISH,
        strength=BiasStrength.STRONG,
        confidence=0.8,
    )
    return TimeframeBias(
        final_direction=BiasDirection.BULLISH,
        final_confidence=0.8,
        final_strength=BiasStrength.STRONG,
        timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
        m15_bias=m15,
        alignment_score=0.8,
    )


class TestSweepClosureGateDisabledByDefault:
    """Both None/False = any sweep counts (legacy behavior)."""

    def test_sweep_closure_gate_disabled_by_default(self):
        config = UnicornConfig()
        assert config.max_sweep_age_bars is None
        assert config.require_sweep_settlement is False

        bars = _make_bars(60)
        # Sweep at bar 10 (age=49 from last bar 59) — very stale but should pass
        old_sweep = _make_sweep(bar_index=10)

        with patch(SWEEP_PATCH, return_value=[old_sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is True
        assert check.sweep_type == "low"


class TestSweepRecencyGate:
    """max_sweep_age_bars rejects stale sweeps."""

    def test_sweep_closure_gate_rejects_stale_sweep(self):
        config = UnicornConfig(max_sweep_age_bars=10)
        bars = _make_bars(60)
        # Sweep at bar 10 → age = 59 - 10 = 49 bars — exceeds threshold of 10
        stale_sweep = _make_sweep(bar_index=10)

        with patch(SWEEP_PATCH, return_value=[stale_sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is False
        assert check.sweep_reject_reason == "stale"
        assert check.sweep_age_bars == 49

    def test_sweep_passes_when_within_age(self):
        config = UnicornConfig(max_sweep_age_bars=10)
        bars = _make_bars(60)
        # Sweep at bar 55 → age = 59 - 55 = 4 — within threshold
        fresh_sweep = _make_sweep(bar_index=55)

        with patch(SWEEP_PATCH, return_value=[fresh_sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is True
        assert check.sweep_reject_reason is None


class TestSweepSettlementGate:
    """require_sweep_settlement rejects unsettled sweeps."""

    def test_sweep_closure_gate_rejects_unsettled_sweep(self):
        config = UnicornConfig(require_sweep_settlement=True)
        bars = _make_bars(60)
        # Sweep at bar 50 (age=9), next bar close BELOW swept level → unsettled
        # swept_level=99.0, so bar 51 must close >= 99.0 for bullish
        sweep = _make_sweep(bar_index=50, swept_level=99.0)
        # Make bar 51 close below swept level
        bars[51] = _make_bar(51 % 60, 98.0, 99.0, 97.0, 98.0)

        with patch(SWEEP_PATCH, return_value=[sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is False
        assert check.sweep_settled is False
        assert check.sweep_reject_reason == "unsettled"

    def test_sweep_closure_gate_rejects_no_next_bar(self):
        """Sweep on last bar — settlement can't be verified."""
        config = UnicornConfig(require_sweep_settlement=True)
        bars = _make_bars(60)
        sweep = _make_sweep(bar_index=59)  # last bar

        with patch(SWEEP_PATCH, return_value=[sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is False
        assert check.sweep_reject_reason == "no_next_bar"


class TestSweepClosureGateCombined:
    """Both gates used together — fresh + settled → passes."""

    def test_sweep_closure_gate_passes_fresh_settled(self):
        config = UnicornConfig(max_sweep_age_bars=10, require_sweep_settlement=True)
        bars = _make_bars(60)
        # Sweep at bar 55 (age=4), next bar closes above swept_level
        sweep = _make_sweep(bar_index=55, swept_level=99.0)
        bars[56] = _make_bar(56 % 60, 99.0, 101.0, 98.5, 100.0)  # close=100 >= 99

        with patch(SWEEP_PATCH, return_value=[sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is True
        assert check.sweep_settled is True
        assert check.sweep_reject_reason is None


class TestSweepNewestFirst:
    """Engine picks most recent qualifying sweep when multiple exist."""

    def test_sweep_closure_gate_prefers_most_recent_qualifying(self):
        config = UnicornConfig(max_sweep_age_bars=20)
        bars = _make_bars(60)
        # Two sweeps: old at bar 30 (age=29, outside gate), new at bar 50 (age=9, inside)
        old_sweep = _make_sweep(bar_index=30)
        new_sweep = _make_sweep(bar_index=50)

        with patch(SWEEP_PATCH, return_value=[old_sweep, new_sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config,
            )

        assert check.liquidity_sweep_found is True
        # Age should reflect the winning sweep (bar 50, age=9)
        assert check.sweep_age_bars == 9


class TestSweepClosureDiagnostics:
    """Diagnostic fields populated correctly for each failure mode."""

    def test_sweep_closure_diagnostics(self):
        """All diagnostic fields set correctly across failure modes."""
        bars = _make_bars(60)

        # Case 1: stale sweep
        config_stale = UnicornConfig(max_sweep_age_bars=5)
        stale = _make_sweep(bar_index=10)
        with patch(SWEEP_PATCH, return_value=[stale]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config_stale,
            )
        assert check.sweep_age_bars == 49
        assert check.sweep_reject_reason == "stale"
        assert check.liquidity_sweep_found is False

        # Case 2: unsettled sweep
        config_settle = UnicornConfig(require_sweep_settlement=True)
        sweep = _make_sweep(bar_index=50, swept_level=99.0)
        bars_copy = _make_bars(60)
        bars_copy[51] = _make_bar(51 % 60, 98.0, 99.0, 97.0, 97.5)  # close < 99
        with patch(SWEEP_PATCH, return_value=[sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars_copy,
                htf_bars=bars_copy,
                ltf_bars=bars_copy,
                symbol="NQ",
                ts=bars_copy[-1].ts,
                config=config_settle,
            )
        assert check.sweep_settled is False
        assert check.sweep_reject_reason == "unsettled"

        # Case 3: no_next_bar
        config_settle2 = UnicornConfig(require_sweep_settlement=True)
        last_sweep = _make_sweep(bar_index=59)
        with patch(SWEEP_PATCH, return_value=[last_sweep]), patch(
            BIAS_PATCH, side_effect=_mock_bias
        ):
            check = check_criteria(
                bars=bars,
                htf_bars=bars,
                ltf_bars=bars,
                symbol="NQ",
                ts=bars[-1].ts,
                config=config_settle2,
            )
        assert check.sweep_settled is False
        assert check.sweep_reject_reason == "no_next_bar"


class TestSweepConfigValidation:
    """Config validation for max_sweep_age_bars."""

    def test_invalid_max_sweep_age_bars_zero(self):
        with pytest.raises(ValueError, match="max_sweep_age_bars must be >= 1"):
            UnicornConfig(max_sweep_age_bars=0)

    def test_invalid_max_sweep_age_bars_negative(self):
        with pytest.raises(ValueError, match="max_sweep_age_bars must be >= 1"):
            UnicornConfig(max_sweep_age_bars=-1)

    def test_valid_max_sweep_age_bars(self):
        config = UnicornConfig(max_sweep_age_bars=1)
        assert config.max_sweep_age_bars == 1
