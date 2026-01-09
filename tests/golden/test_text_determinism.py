"""
Golden tests for text template determinism.

These tests lock down exact text output to catch any changes
that would affect embedding stability or reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
from uuid import UUID

from app.services.kb import (
    RegimeSnapshot,
    TrialDoc,
    compute_regime_snapshot,
    regime_snapshot_to_text,
    trial_to_text,
    trial_to_metadata,
    REGIME_SCHEMA_VERSION,
    TRIAL_DOC_SCHEMA_VERSION,
)


# =============================================================================
# Deterministic Test Data
# =============================================================================


def create_deterministic_ohlcv():
    """Create a deterministic OHLCV dataset for golden tests."""
    np.random.seed(12345)  # Fixed seed for reproducibility
    dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")

    # Generate deterministic price series
    base = 100.0
    returns = np.random.normal(0.0001, 0.01, 200)
    close = base * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": close * (1 - 0.002),
            "high": close * (1 + 0.005),
            "low": close * (1 - 0.005),
            "close": close,
            "volume": np.random.uniform(1000, 2000, 200),
        },
        index=dates,
    )
    return df


def create_deterministic_regime_snapshot():
    """Create a RegimeSnapshot with fixed values for golden tests."""
    return RegimeSnapshot(
        schema_version=REGIME_SCHEMA_VERSION,
        feature_params={
            "atr_period": 14,
            "rsi_period": 14,
            "bb_period": 20,
            "bb_k": 2.0,
            "z_period": 20,
            "trend_lookback": 50,
        },
        timeframe="1h",
        computed_at="2024-01-01T00:00:00Z",
        computation_source="test",
        n_bars=200,
        effective_n_bars=186,
        ts_start="2024-01-01T00:00:00+00:00",
        ts_end="2024-01-08T07:00:00+00:00",
        atr_pct=0.0123,
        std_pct=0.0098,
        bb_width_pct=0.034,
        range_pct=0.12,
        trend_strength=0.72,
        trend_dir=1,
        zscore=0.45,
        rsi=58.0,
        return_pct=0.05,
        drift_bps_per_bar=2.5,
        efficiency_ratio=0.65,
        instrument="BTCUSD",
        source="test",
        regime_tags=["efficient", "uptrend"],
        warnings=[],
    )


def create_deterministic_trial_doc():
    """Create a TrialDoc with fixed values for golden tests."""
    return TrialDoc(
        schema_version=TRIAL_DOC_SCHEMA_VERSION,
        doc_type="trial",
        tune_run_id=UUID("12345678-1234-5678-1234-567812345678"),
        tune_id=UUID("87654321-4321-8765-4321-876543218765"),
        workspace_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        dataset_id="btcusd_1h_2024",
        instrument="BTCUSD",
        timeframe="1h",
        strategy_name="mean_reversion",
        params={"period": 20, "threshold": 2.0, "stop_loss": 0.02},
        sharpe_is=1.45,
        sharpe_oos=1.12,
        return_frac_is=0.25,
        return_frac_oos=0.18,
        max_dd_frac_is=0.08,
        max_dd_frac_oos=0.12,
        n_trades_is=45,
        n_trades_oos=38,
        overfit_gap=0.33,
        regime_is=create_deterministic_regime_snapshot(),
        regime_oos=None,
        has_oos=True,
        is_valid=True,
        warnings=["moderate_overfit"],
        objective_type="sharpe",
        objective_score=1.12,
        created_at="2024-01-15T10:30:00Z",
    )


# =============================================================================
# Golden Text Tests
# =============================================================================


class TestRegimeSnapshotTextDeterminism:
    """Golden tests for regime_snapshot_to_text()."""

    def test_exact_text_match(self):
        """Text output should match golden reference exactly."""
        snapshot = create_deterministic_regime_snapshot()
        text = regime_snapshot_to_text(snapshot)

        # Golden reference
        expected = """Market regime (1h): efficient, uptrend.
Volatility: ATR 1.23%, BB width 3.4%.
Trend: strength 0.72, direction +1.
Position: z-score 0.45, RSI 58.
Efficiency: 0.65."""

        assert text == expected, f"Text mismatch:\nGot:\n{text}\n\nExpected:\n{expected}"

    def test_text_without_timeframe(self):
        """Text without timeframe should format correctly."""
        snapshot = create_deterministic_regime_snapshot()
        snapshot.timeframe = None
        text = regime_snapshot_to_text(snapshot)

        # Should not have timeframe in parentheses
        assert "(1h)" not in text
        assert "Market regime:" in text

    def test_text_neutral_tags(self):
        """Empty tags should show 'neutral'."""
        snapshot = create_deterministic_regime_snapshot()
        snapshot.regime_tags = []
        text = regime_snapshot_to_text(snapshot)

        assert "neutral" in text

    def test_text_multiple_tags(self):
        """Multiple tags should be comma-separated."""
        snapshot = create_deterministic_regime_snapshot()
        snapshot.regime_tags = ["high_vol", "noisy", "uptrend"]
        text = regime_snapshot_to_text(snapshot)

        assert "high_vol, noisy, uptrend" in text


class TestTrialDocTextDeterminism:
    """Golden tests for trial_to_text()."""

    def test_exact_text_match(self):
        """Text output should match golden reference exactly."""
        trial = create_deterministic_trial_doc()
        text = trial_to_text(trial)

        # Golden reference
        expected = """Dataset: btcusd_1h_2024 1h. OOS enabled.
Regime: efficient, uptrend.
Strategy: mean_reversion with period=20, stop_loss=0.02, threshold=2.0.
Performance: OOS Sharpe 1.12, return 18.0%, max DD 12.0%. Objective: sharpe (score 1.12). (moderate overfit)"""

        assert text == expected, f"Text mismatch:\nGot:\n{text}\n\nExpected:\n{expected}"

    def test_text_without_oos(self):
        """Text without OOS should use IS metrics."""
        trial = create_deterministic_trial_doc()
        trial.has_oos = False
        trial.sharpe_oos = None
        trial.return_frac_oos = None
        trial.max_dd_frac_oos = None
        trial.regime_oos = None
        trial.warnings = []
        text = trial_to_text(trial)

        assert "IS Sharpe 1.45" in text
        assert "OOS enabled" not in text

    def test_text_params_sorted(self):
        """Params should be sorted alphabetically."""
        trial = create_deterministic_trial_doc()
        trial.params = {"z_param": 3, "a_param": 1, "m_param": 2}
        text = trial_to_text(trial)

        # Should be alphabetical
        assert "a_param=1, m_param=2, z_param=3" in text


class TestTrialDocMetadataDeterminism:
    """Golden tests for trial_to_metadata()."""

    def test_metadata_keys_present(self):
        """All required metadata keys should be present."""
        trial = create_deterministic_trial_doc()
        meta = trial_to_metadata(trial)

        required_keys = [
            "schema_version",
            "doc_type",
            "tune_run_id",
            "tune_id",
            "workspace_id",
            "strategy_name",
            "params",
            "sharpe_is",
            "sharpe_oos",
            "regime_tags",
            "has_oos",
            "is_valid",
            "objective_type",
            "objective_score",
            "created_at",
        ]

        for key in required_keys:
            assert key in meta, f"Missing key: {key}"

    def test_metadata_values_match(self):
        """Metadata values should match trial doc."""
        trial = create_deterministic_trial_doc()
        meta = trial_to_metadata(trial)

        assert meta["doc_type"] == "trial"
        assert meta["strategy_name"] == "mean_reversion"
        assert meta["sharpe_oos"] == 1.12
        assert meta["regime_tags"] == ["efficient", "uptrend"]
        assert meta["has_oos"] is True
        assert meta["is_valid"] is True

    def test_metadata_uuids_as_strings(self):
        """UUIDs should be converted to strings."""
        trial = create_deterministic_trial_doc()
        meta = trial_to_metadata(trial)

        assert isinstance(meta["tune_run_id"], str)
        assert isinstance(meta["tune_id"], str)
        assert isinstance(meta["workspace_id"], str)

    def test_metadata_regime_snapshot_serialized(self):
        """Regime snapshots should be serialized as dicts."""
        trial = create_deterministic_trial_doc()
        meta = trial_to_metadata(trial)

        # regime_is should be a dict
        assert isinstance(meta["regime_snapshot_is"], dict)
        assert meta["regime_snapshot_is"]["schema_version"] == REGIME_SCHEMA_VERSION

        # regime_oos is None
        assert meta["regime_snapshot_oos"] is None


# =============================================================================
# Stability Tests
# =============================================================================


class TestStability:
    """Tests that output remains stable across runs."""

    def test_regime_text_stable_across_calls(self):
        """Same input should produce same text every time."""
        df = create_deterministic_ohlcv()

        texts = []
        for _ in range(5):
            snapshot = compute_regime_snapshot(df, source="test", timeframe="1h")
            # Reset computed_at for comparison
            snapshot.computed_at = "2024-01-01T00:00:00Z"
            texts.append(regime_snapshot_to_text(snapshot))

        # All texts should be identical
        assert len(set(texts)) == 1, "Text output not stable across calls"

    def test_trial_text_stable_across_calls(self):
        """Same trial should produce same text every time."""
        trial = create_deterministic_trial_doc()

        texts = [trial_to_text(trial) for _ in range(5)]

        assert len(set(texts)) == 1, "Trial text not stable across calls"

    def test_metadata_stable_across_calls(self):
        """Same trial should produce same metadata every time."""
        trial = create_deterministic_trial_doc()

        # Compare serialized metadata
        metas = [str(sorted(trial_to_metadata(trial).items())) for _ in range(5)]

        assert len(set(metas)) == 1, "Metadata not stable across calls"


# =============================================================================
# Schema Version Tests
# =============================================================================


class TestSchemaVersions:
    """Tests for schema versioning."""

    def test_regime_schema_version_in_snapshot(self):
        """Schema version should be included in snapshot."""
        df = create_deterministic_ohlcv()
        snapshot = compute_regime_snapshot(df, source="test")

        assert snapshot.schema_version == REGIME_SCHEMA_VERSION

    def test_regime_schema_version_in_dict(self):
        """Schema version should survive dict round-trip."""
        df = create_deterministic_ohlcv()
        snapshot = compute_regime_snapshot(df, source="test")
        d = snapshot.to_dict()
        restored = RegimeSnapshot.from_dict(d)

        assert restored.schema_version == REGIME_SCHEMA_VERSION

    def test_trial_schema_version_in_metadata(self):
        """Trial schema version should be in metadata."""
        trial = create_deterministic_trial_doc()
        meta = trial_to_metadata(trial)

        assert meta["schema_version"] == TRIAL_DOC_SCHEMA_VERSION
