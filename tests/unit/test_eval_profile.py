"""Unit tests for EvalAccountProfile and compute_r_day."""

import pytest
from dataclasses import FrozenInstanceError

from app.services.backtest.engines.eval_profile import (
    EvalAccountProfile,
    compute_r_day,
)


@pytest.fixture
def profile():
    """Standard 50k / $2k DD profile."""
    return EvalAccountProfile(
        account_size=50_000,
        max_drawdown_dollars=2_000,
        max_daily_loss_dollars=1_000,
        risk_fraction=0.15,
        r_min_dollars=100.0,
        r_max_dollars=300.0,
    )


class TestComputeRDay:
    """Tests for the compute_r_day pure function."""

    def test_r_day_nominal_at_cap(self, profile):
        """Full room ($2000) * 0.15 = $300 -> capped at r_max."""
        r = compute_r_day(profile, current_equity=50_000, peak_equity=50_000)
        assert r == 300.0

    def test_r_day_at_start(self, profile):
        """At start of run, current == peak == account_size -> full room."""
        r = compute_r_day(profile, current_equity=50_000, peak_equity=50_000)
        assert r == 300.0

    def test_r_day_partial_drawdown(self, profile):
        """Peak=52000, current=51000 -> trailing DD=$1000, room=$1000 -> $150."""
        r = compute_r_day(profile, current_equity=51_000, peak_equity=52_000)
        assert r == 150.0

    def test_r_day_floor_clamp(self, profile):
        """Room=500, fraction=0.15 -> raw=$75, clamped to r_min=$100."""
        # trailing DD = 1500, room = 2000-1500 = 500
        r = compute_r_day(profile, current_equity=50_500, peak_equity=52_000)
        assert r == 100.0

    def test_r_day_blown_exact(self, profile):
        """Room exactly 0 -> eval blown."""
        # trailing DD = 2000, room = 0
        r = compute_r_day(profile, current_equity=48_000, peak_equity=50_000)
        assert r == 0.0

    def test_r_day_negative_room(self, profile):
        """Over max DD -> still 0."""
        r = compute_r_day(profile, current_equity=47_000, peak_equity=50_000)
        assert r == 0.0

    def test_r_day_small_room_above_floor(self, profile):
        """Room=800 -> raw=$120, between min and max."""
        # trailing DD = 1200, room = 800
        r = compute_r_day(profile, current_equity=48_800, peak_equity=50_000)
        assert r == 120.0

    def test_r_day_peak_above_account_size(self, profile):
        """Peak can exceed account_size (profitable run)."""
        # peak=55000, current=55000 -> trailing DD=0, room=2000 -> cap at 300
        r = compute_r_day(profile, current_equity=55_000, peak_equity=55_000)
        assert r == 300.0

    def test_r_day_peak_above_account_with_drawdown(self, profile):
        """Peak=55000, current=53500 -> trailing DD=1500, room=500 -> raw=75 -> floor 100."""
        r = compute_r_day(profile, current_equity=53_500, peak_equity=55_000)
        assert r == 100.0


class TestProfileFrozen:
    """EvalAccountProfile should be immutable."""

    def test_profile_frozen(self, profile):
        with pytest.raises(FrozenInstanceError):
            profile.account_size = 100_000

    def test_profile_fields(self, profile):
        assert profile.account_size == 50_000
        assert profile.max_drawdown_dollars == 2_000
        assert profile.max_daily_loss_dollars == 1_000
        assert profile.risk_fraction == 0.15
        assert profile.r_min_dollars == 100.0
        assert profile.r_max_dollars == 300.0
