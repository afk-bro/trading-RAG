"""Tests for WFO models and fold generation."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.services.backtest.wfo import (
    WFOConfig,
    Fold,
    WFOCandidateMetrics,
    WFOResult,
    InsufficientDataError,
    generate_folds,
    compute_params_hash,
)


class TestWFOConfig:
    """Tests for WFOConfig dataclass."""

    def test_valid_config(self):
        """Should create valid config."""
        config = WFOConfig(
            train_days=90,
            test_days=30,
            step_days=30,
            min_folds=5,
        )

        assert config.train_days == 90
        assert config.test_days == 30
        assert config.step_days == 30
        assert config.min_folds == 5
        assert config.leaderboard_top_k == 10  # default
        assert config.allow_partial is False  # default

    def test_config_with_date_bounds(self):
        """Should accept start/end timestamps."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        config = WFOConfig(
            train_days=60,
            test_days=20,
            step_days=20,
            start_ts=start,
            end_ts=end,
        )

        assert config.start_ts == start
        assert config.end_ts == end

    def test_invalid_train_days(self):
        """Should reject non-positive train_days."""
        with pytest.raises(ValueError, match="train_days must be positive"):
            WFOConfig(train_days=0, test_days=30, step_days=30)

    def test_invalid_test_days(self):
        """Should reject non-positive test_days."""
        with pytest.raises(ValueError, match="test_days must be positive"):
            WFOConfig(train_days=90, test_days=0, step_days=30)

    def test_invalid_step_days(self):
        """Should reject non-positive step_days."""
        with pytest.raises(ValueError, match="step_days must be positive"):
            WFOConfig(train_days=90, test_days=30, step_days=0)

    def test_invalid_min_folds(self):
        """Should reject min_folds < 1."""
        with pytest.raises(ValueError, match="min_folds must be at least 1"):
            WFOConfig(train_days=90, test_days=30, step_days=30, min_folds=0)


class TestFold:
    """Tests for Fold dataclass."""

    def test_valid_fold(self):
        """Should create valid fold."""
        fold = Fold(
            index=0,
            train_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2024, 4, 1, tzinfo=timezone.utc),
            test_start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            test_end=datetime(2024, 5, 1, tzinfo=timezone.utc),
        )

        assert fold.index == 0
        assert fold.train_days == 91  # Jan 1 to Apr 1
        assert fold.test_days == 30  # Apr 1 to May 1

    def test_contiguous_validation(self):
        """Should reject non-contiguous train/test windows."""
        with pytest.raises(ValueError, match="train_end must equal test_start"):
            Fold(
                index=0,
                train_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                train_end=datetime(2024, 4, 1, tzinfo=timezone.utc),
                test_start=datetime(2024, 4, 2, tzinfo=timezone.utc),  # Gap!
                test_end=datetime(2024, 5, 2, tzinfo=timezone.utc),
            )

    def test_invalid_train_window(self):
        """Should reject train_start >= train_end."""
        with pytest.raises(ValueError, match="train_start must be before train_end"):
            Fold(
                index=0,
                train_start=datetime(2024, 4, 1, tzinfo=timezone.utc),
                train_end=datetime(2024, 1, 1, tzinfo=timezone.utc),  # Before start!
                test_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                test_end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            )


class TestWFOCandidateMetrics:
    """Tests for WFOCandidateMetrics dataclass."""

    def test_coverage_calculation(self):
        """Should calculate coverage correctly."""
        candidate = WFOCandidateMetrics(
            params={"lookback": 20},
            params_hash="abc123",
            mean_oos=1.5,
            median_oos=1.4,
            worst_fold_oos=0.8,
            stddev_oos=0.3,
            pct_top_k=0.8,
            fold_count=4,
            total_folds=5,
        )

        assert candidate.coverage == 0.8
        assert candidate.meets_coverage_threshold is True

    def test_coverage_below_threshold(self):
        """Should detect coverage below threshold."""
        candidate = WFOCandidateMetrics(
            params={"lookback": 20},
            params_hash="abc123",
            mean_oos=1.5,
            median_oos=1.4,
            worst_fold_oos=0.8,
            stddev_oos=0.3,
            pct_top_k=0.4,
            fold_count=2,
            total_folds=5,
        )

        assert candidate.coverage == 0.4
        assert candidate.meets_coverage_threshold is False


class TestGenerateFolds:
    """Tests for generate_folds function."""

    def test_basic_fold_generation(self):
        """Should generate correct number of folds."""
        config = WFOConfig(
            train_days=90,
            test_days=30,
            step_days=30,
            min_folds=3,
        )

        # 360 days of data
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 26, tzinfo=timezone.utc)  # ~360 days

        folds = generate_folds(config, (start, end))

        assert len(folds) >= 3
        # First fold: train 90 days + test 30 days = 120 days
        # Then step 30 days each time
        # With 360 days: can fit ~9 folds

    def test_fold_boundaries(self):
        """Should create contiguous non-overlapping folds."""
        config = WFOConfig(
            train_days=60,
            test_days=20,
            step_days=20,
            min_folds=2,
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 30, tzinfo=timezone.utc)

        folds = generate_folds(config, (start, end))

        # Verify each fold is contiguous
        for fold in folds:
            assert fold.train_end == fold.test_start

        # Verify folds step correctly
        for i in range(1, len(folds)):
            expected_step = timedelta(days=config.step_days)
            actual_step = folds[i].train_start - folds[i - 1].train_start
            assert actual_step == expected_step

    def test_fold_indices(self):
        """Should assign sequential indices."""
        config = WFOConfig(
            train_days=30,
            test_days=10,
            step_days=10,
            min_folds=2,
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 3, 31, tzinfo=timezone.utc)

        folds = generate_folds(config, (start, end))

        for i, fold in enumerate(folds):
            assert fold.index == i

    def test_respects_config_start_end(self):
        """Should respect config start/end bounds."""
        config = WFOConfig(
            train_days=30,
            test_days=10,
            step_days=10,
            min_folds=2,
            start_ts=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end_ts=datetime(2024, 4, 30, tzinfo=timezone.utc),
        )

        # Available data is wider
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        folds = generate_folds(config, (start, end))

        # First fold should start at config.start_ts
        assert folds[0].train_start == config.start_ts

        # Last fold should end at or before config.end_ts
        assert folds[-1].test_end <= config.end_ts

    def test_insufficient_data_raises(self):
        """Should raise InsufficientDataError when data too short."""
        config = WFOConfig(
            train_days=90,
            test_days=30,
            step_days=30,
            min_folds=5,
        )

        # Only 100 days - not enough for even one fold
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 4, 10, tzinfo=timezone.utc)

        with pytest.raises(InsufficientDataError, match="Need 120 days"):
            generate_folds(config, (start, end))

    def test_insufficient_folds_raises(self):
        """Should raise InsufficientDataError when cannot produce min_folds."""
        config = WFOConfig(
            train_days=90,
            test_days=30,
            step_days=60,
            min_folds=10,  # Unrealistic for data range
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        with pytest.raises(InsufficientDataError, match="Only .* folds possible"):
            generate_folds(config, (start, end))

    def test_handles_naive_datetimes(self):
        """Should handle naive datetimes by assuming UTC."""
        config = WFOConfig(
            train_days=30,
            test_days=10,
            step_days=10,
            min_folds=2,
        )

        # Naive datetimes (no tzinfo)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)

        folds = generate_folds(config, (start, end))

        # Should succeed and produce timezone-aware folds
        assert len(folds) >= 2


class TestComputeParamsHash:
    """Tests for compute_params_hash function."""

    def test_deterministic(self):
        """Same params should produce same hash."""
        params = {"lookback": 20, "threshold": 0.5}

        hash1 = compute_params_hash(params)
        hash2 = compute_params_hash(params)

        assert hash1 == hash2

    def test_order_independent(self):
        """Hash should be independent of key order."""
        params1 = {"lookback": 20, "threshold": 0.5}
        params2 = {"threshold": 0.5, "lookback": 20}

        assert compute_params_hash(params1) == compute_params_hash(params2)

    def test_different_params_different_hash(self):
        """Different params should produce different hashes."""
        params1 = {"lookback": 20}
        params2 = {"lookback": 30}

        assert compute_params_hash(params1) != compute_params_hash(params2)

    def test_handles_nested_params(self):
        """Should handle nested param structures."""
        params = {
            "entry": {"lookback": 20, "threshold": 0.5},
            "exit": {"take_profit": 0.1},
        }

        hash_val = compute_params_hash(params)

        assert len(hash_val) == 16  # SHA256 truncated


class TestWFOResult:
    """Tests for WFOResult dataclass."""

    def test_create_result(self):
        """Should create valid result."""
        result = WFOResult(
            wfo_id=uuid4(),
            status="completed",
            n_folds=5,
            folds_completed=5,
            folds_failed=0,
            candidates=[],
            best_params={"lookback": 20},
            best_candidate=None,
        )

        assert result.status == "completed"
        assert result.n_folds == 5
        assert result.folds_failed == 0

    def test_result_with_candidates(self):
        """Should store candidates and best candidate."""
        candidate = WFOCandidateMetrics(
            params={"lookback": 20},
            params_hash="abc123",
            mean_oos=1.5,
            median_oos=1.4,
            worst_fold_oos=0.8,
            stddev_oos=0.3,
            pct_top_k=0.8,
            fold_count=4,
            total_folds=5,
        )

        result = WFOResult(
            wfo_id=uuid4(),
            status="completed",
            n_folds=5,
            folds_completed=5,
            folds_failed=0,
            candidates=[candidate],
            best_params=candidate.params,
            best_candidate=candidate,
        )

        assert len(result.candidates) == 1
        assert result.best_candidate == candidate
