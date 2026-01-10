"""Unit tests for trial document conversion utilities."""

import pytest
from uuid import uuid4

from app.services.kb.trial_doc import (
    regime_to_text,
    trial_to_text,
    trial_to_metadata,
)
from app.services.kb.types import TrialDoc, RegimeSnapshot


def make_regime(tags: list[str], **kwargs) -> RegimeSnapshot:
    """Helper to create RegimeSnapshot with tags."""
    return RegimeSnapshot(regime_tags=tags, **kwargs)


# =============================================================================
# Golden Tests - Lock the embedding text format
# =============================================================================


class TestRegimeToTextGolden:
    """Golden tests to lock regime_to_text format for embedding consistency."""

    def test_full_regime_snapshot(self):
        """Full regime should produce deterministic text."""
        regime = RegimeSnapshot(
            atr_pct=0.052,
            trend_strength=0.75,
            trend_dir=1,
            rsi=65,
            efficiency_ratio=0.72,
            regime_tags=["high_vol", "uptrend"],
            instrument="BTCUSDT",
            timeframe="1h",
        )

        result = regime_to_text(regime)

        # Lock the exact format
        assert (
            result
            == """Regime: high_vol, uptrend.
Market conditions for BTCUSDT 1h: ATR 5.2%, trend up (0.75), RSI 65, efficiency 0.72."""
        )

    def test_neutral_regime(self):
        """Neutral regime with defaults."""
        regime = RegimeSnapshot()

        result = regime_to_text(regime)

        assert (
            result
            == """Regime: neutral.
Market conditions for unknown: neutral conditions."""
        )

    def test_none_regime(self):
        """None regime should return unknown."""
        result = regime_to_text(None)

        assert result == "Regime: unknown."

    def test_downtrend_regime(self):
        """Downtrend direction formatting."""
        regime = RegimeSnapshot(
            atr_pct=0.03,
            trend_strength=0.6,
            trend_dir=-1,
            rsi=35,
            efficiency_ratio=0.5,
            regime_tags=["downtrend", "oversold"],
        )

        result = regime_to_text(regime)

        assert "trend down (0.60)" in result
        assert "RSI 35" in result
        assert "Regime: downtrend, oversold." in result

    def test_flat_trend_regime(self):
        """Flat trend direction."""
        regime = RegimeSnapshot(
            trend_strength=0.4,
            trend_dir=0,
        )

        result = regime_to_text(regime)

        assert "trend flat (0.40)" in result


class TestTrialToTextGolden:
    """Golden tests to lock trial_to_text format."""

    def test_full_trial_with_oos(self):
        """Full trial with OOS metrics."""
        trial = TrialDoc(
            tune_run_id=uuid4(),
            tune_id=uuid4(),
            workspace_id=uuid4(),
            dataset_id="btc_2023",
            instrument="BTCUSDT",
            timeframe="1h",
            strategy_name="ema_crossover",
            params={"fast": 12, "slow": 26},
            sharpe_is=1.5,
            sharpe_oos=1.2,
            return_frac_is=0.25,
            return_frac_oos=0.18,
            max_dd_frac_is=0.08,
            max_dd_frac_oos=0.10,
            has_oos=True,
            is_valid=True,
            objective_type="sharpe",
            objective_score=1.2,
            regime_oos=make_regime(["uptrend", "low_vol"]),
            warnings=[],
        )

        result = trial_to_text(trial)

        # Check key components
        assert "Dataset: btc_2023 1h." in result
        assert "OOS enabled." in result
        assert "Regime: uptrend, low_vol." in result
        assert "Strategy: ema_crossover with fast=12, slow=26." in result
        assert "OOS Sharpe 1.20" in result
        assert "return 18.0%" in result
        assert "max DD 10.0%" in result
        assert "Objective: sharpe (score 1.20)." in result

    def test_trial_without_oos(self):
        """Trial with IS-only metrics."""
        trial = TrialDoc(
            tune_run_id=uuid4(),
            tune_id=uuid4(),
            workspace_id=uuid4(),
            strategy_name="rsi_mean_reversion",
            params={"period": 14, "threshold": 30},
            sharpe_is=0.8,
            return_frac_is=0.12,
            max_dd_frac_is=0.15,
            has_oos=False,
            is_valid=True,
            objective_type="sharpe",
            objective_score=0.8,
            # No regime_is/oos = empty tags
            warnings=["low_sharpe"],
        )

        result = trial_to_text(trial)

        assert "OOS enabled." not in result
        assert "IS Sharpe 0.80" in result
        assert "Regime: neutral." in result
        assert "(low sharpe)" in result


# =============================================================================
# Regime/Trial Text Consistency
# =============================================================================


class TestTextFormatConsistency:
    """Ensure regime_to_text and trial_to_text use consistent formatting."""

    def test_regime_line_format_matches(self):
        """Regime line should use same format in both functions."""
        tags = ["high_vol", "uptrend"]

        # From regime_to_text
        regime = RegimeSnapshot(regime_tags=tags)
        regime_text = regime_to_text(regime)
        regime_line = regime_text.split("\n")[0]

        # From trial_to_text
        trial = TrialDoc(
            tune_run_id=uuid4(),
            tune_id=uuid4(),
            workspace_id=uuid4(),
            strategy_name="test",
            params={},
            regime_oos=make_regime(tags),  # Use regime_oos for tags
        )
        trial_text = trial_to_text(trial)

        # Find Regime line in trial text
        for line in trial_text.split("\n"):
            if line.startswith("Regime:"):
                trial_regime_line = line
                break

        # Both should have same regime line format
        assert regime_line == trial_regime_line

    def test_neutral_regime_format_matches(self):
        """Neutral/empty tags should format the same."""
        # From regime_to_text
        regime = RegimeSnapshot(regime_tags=[])
        regime_text = regime_to_text(regime)
        regime_line = regime_text.split("\n")[0]

        # From trial_to_text (no regime = empty tags)
        trial = TrialDoc(
            tune_run_id=uuid4(),
            tune_id=uuid4(),
            workspace_id=uuid4(),
            strategy_name="test",
            params={},
            # No regime_is/oos = empty tags
        )
        trial_text = trial_to_text(trial)

        for line in trial_text.split("\n"):
            if line.startswith("Regime:"):
                trial_regime_line = line
                break

        assert regime_line == trial_regime_line
        assert "neutral" in regime_line
