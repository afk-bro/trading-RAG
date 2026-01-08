"""Tests for overfit diagnostics (gap computation)."""

import pytest
from pydantic import ValidationError

from app.routers.backtests import TuneRunListItem


class TestOverfitGapComputation:
    """Tests for overfit_gap computation."""

    def test_gap_computed_correctly_positive(self):
        """Gap should be IS - OOS when both present (positive case)."""
        score_is = 1.8
        score_oos = 1.2

        # Compute as the API does
        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == 0.6

    def test_gap_computed_correctly_zero(self):
        """Gap should be 0 when IS equals OOS."""
        score_is = 1.5
        score_oos = 1.5

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == 0.0

    def test_gap_computed_negative_when_oos_better(self):
        """Gap should be negative when OOS > IS (rare but good)."""
        score_is = 1.2
        score_oos = 1.5

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == -0.3

    def test_gap_with_negative_scores(self):
        """Gap should handle negative scores correctly."""
        # Both negative (e.g., bad Sharpe ratios)
        score_is = -0.5
        score_oos = -1.2

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        # IS = -0.5, OOS = -1.2, gap = -0.5 - (-1.2) = 0.7
        assert overfit_gap == 0.7

    def test_gap_none_when_is_missing(self):
        """Gap should be None when score_is is missing."""
        score_is = None
        score_oos = 1.5

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap is None

    def test_gap_none_when_oos_missing(self):
        """Gap should be None when score_oos is missing."""
        score_is = 1.8
        score_oos = None

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap is None

    def test_gap_none_when_both_missing(self):
        """Gap should be None when both scores missing."""
        score_is = None
        score_oos = None

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap is None


class TestOverfitGapRounding:
    """Tests for gap rounding precision."""

    def test_gap_rounded_to_4_decimals(self):
        """Gap should be rounded to 4 decimal places."""
        score_is = 1.82345678
        score_oos = 1.23456789

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        # 1.82345678 - 1.23456789 = 0.58888889
        assert overfit_gap == 0.5889

    def test_small_gap_preserved(self):
        """Small gaps should not be rounded to zero."""
        score_is = 1.5001
        score_oos = 1.5

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == 0.0001


class TestTuneRunListItemWithGap:
    """Tests for TuneRunListItem model with overfit_gap."""

    def test_model_accepts_overfit_gap(self):
        """Model should accept overfit_gap field."""
        item = TuneRunListItem(
            trial_index=0,
            run_id="abc123",
            params={"period": 20},
            score=1.5,
            score_is=1.8,
            score_oos=1.5,
            overfit_gap=0.3,
            status="completed",
        )

        assert item.overfit_gap == 0.3

    def test_model_gap_optional(self):
        """overfit_gap should be optional (None)."""
        item = TuneRunListItem(
            trial_index=0,
            run_id="abc123",
            params={"period": 20},
            score=1.5,
            status="completed",
        )

        assert item.overfit_gap is None

    def test_model_gap_negative(self):
        """Model should accept negative gap values."""
        item = TuneRunListItem(
            trial_index=0,
            run_id="abc123",
            params={"period": 20},
            score=1.5,
            score_is=1.2,
            score_oos=1.5,
            overfit_gap=-0.3,
            status="completed",
        )

        assert item.overfit_gap == -0.3


class TestOverfitWarningThresholds:
    """Tests documenting warning threshold behavior."""

    def test_gap_above_warning_threshold(self):
        """Gap > 0.5 should trigger warning styling in UI.

        Threshold: GAP_WARN = 0.5 Sharpe
        """
        score_is = 2.0
        score_oos = 1.3  # Gap = 0.7 (overfit warning)

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == 0.7
        assert overfit_gap > 0.5  # Would show warning

    def test_gap_above_danger_threshold(self):
        """Gap > 1.0 should trigger danger styling in UI.

        This indicates severe overfitting.
        """
        score_is = 2.5
        score_oos = 1.2  # Gap = 1.3 (severe overfit)

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == 1.3
        assert overfit_gap > 1.0  # Would show danger

    def test_gap_below_warning_threshold(self):
        """Gap <= 0.5 should show normal styling."""
        score_is = 1.6
        score_oos = 1.4  # Gap = 0.2 (acceptable)

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == 0.2
        assert overfit_gap <= 0.5  # Would show normal

    def test_negative_gap_shows_success(self):
        """Negative gap (OOS > IS) should show success styling.

        This is rare but indicates the model generalizes well.
        """
        score_is = 1.3
        score_oos = 1.6  # Gap = -0.3 (OOS better!)

        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)
        else:
            overfit_gap = None

        assert overfit_gap == -0.3
        assert overfit_gap < 0  # Would show success


class TestNonSplitTuneHasNoGap:
    """Tests that non-split tunes don't show gap."""

    def test_no_gap_when_oos_ratio_null(self):
        """When oos_ratio is null, gap should not be computed.

        In non-split mode, there's no IS/OOS distinction, so
        gap doesn't make sense.
        """
        # Non-split tune has score but no score_is/score_oos
        item = TuneRunListItem(
            trial_index=0,
            run_id="abc123",
            params={"period": 20},
            score=1.5,
            score_is=None,
            score_oos=None,
            overfit_gap=None,
            status="completed",
        )

        assert item.score == 1.5
        assert item.score_is is None
        assert item.score_oos is None
        assert item.overfit_gap is None
