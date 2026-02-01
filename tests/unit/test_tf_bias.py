"""Tests for multi-timeframe bias computation."""

from datetime import datetime, timedelta, timezone

from app.services.strategy.indicators.tf_bias import (
    BiasDirection,
    BiasStrength,
    TimeframeBiasComponent,
    _is_component_usable,
    compute_tf_bias,
)
from app.services.strategy.models import OHLCVBar


# ---------------------------------------------------------------------------
# Helper to build a minimal component
# ---------------------------------------------------------------------------


def _make_component(
    tf: str,
    direction: BiasDirection = BiasDirection.BULLISH,
    confidence: float = 0.8,
    factors: dict | None = None,
) -> TimeframeBiasComponent:
    return TimeframeBiasComponent(
        timeframe=tf,
        direction=direction,
        strength=BiasStrength.STRONG if confidence > 0.75 else BiasStrength.MODERATE,
        confidence=confidence,
        factors=factors or {},
    )


# ---------------------------------------------------------------------------
# _is_component_usable
# ---------------------------------------------------------------------------


class TestIsComponentUsable:
    def test_none_is_unusable(self):
        assert _is_component_usable(None) is False

    def test_insufficient_data_is_unusable(self):
        comp = _make_component("4h", factors={"error": "insufficient_data"})
        assert _is_component_usable(comp) is False

    def test_normal_component_is_usable(self):
        comp = _make_component("4h")
        assert _is_component_usable(comp) is True

    def test_other_error_is_still_usable(self):
        """Only the specific 'insufficient_data' sentinel makes a component unusable."""
        comp = _make_component("4h", factors={"error": "some_other_error"})
        assert _is_component_usable(comp) is True

    def test_empty_factors_is_usable(self):
        comp = _make_component("4h", factors={})
        assert _is_component_usable(comp) is True


# ---------------------------------------------------------------------------
# compute_tf_bias: insufficient_data exclusion
# ---------------------------------------------------------------------------


def _make_bars(n: int, tf_minutes: int = 15) -> list[OHLCVBar]:
    """Generate trivially-bullish bars for a given timeframe."""
    base = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
    bars = []
    for i in range(n):
        ts = base + timedelta(minutes=i * tf_minutes)
        bars.append(
            OHLCVBar(
                ts=ts,
                open=100.0 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.5 + i * 0.1,
                volume=1000.0,
            )
        )
    return bars


class TestInsufficientDataExclusion:
    """Verify that an insufficient_data component does not tank confidence."""

    def test_insufficient_h4_does_not_contribute_to_denominator(self):
        """Core regression test: H4 with insufficient_data must not inflate
        total_weight while contributing 0.0 to the numerator."""

        # Provide enough bars for H1, M15, M5 to produce real components,
        # but too few for H4 (needs 210 bars for EMA(200)+10).
        h4_bars = _make_bars(20, tf_minutes=240)  # 20 bars — insufficient
        h1_bars = _make_bars(100, tf_minutes=60)
        m15_bars = _make_bars(200, tf_minutes=15)
        m5_bars = _make_bars(500, tf_minutes=5)

        result = compute_tf_bias(
            h4_bars=h4_bars,
            h1_bars=h1_bars,
            m15_bars=m15_bars,
            m5_bars=m5_bars,
        )

        # H4 component should still be on the result (transparency)
        assert result.h4_bias is not None
        assert result.h4_bias.factors.get("error") == "insufficient_data"

        # The bias should still be computable from the remaining TFs.
        # Without the fix, confidence would be ~0.47 (below 0.5 = WEAK).
        # With the fix, H4's 0.30 weight is removed from the denominator.
        assert result.final_confidence > 0.0
        # The key assertion: confidence should not be dragged to WEAK
        # by a zero-confidence phantom component.
        assert result.final_strength != BiasStrength.WEAK or result.final_direction == BiasDirection.NEUTRAL

    def test_all_insufficient_yields_zero_confidence(self):
        """If every component has insufficient data, confidence should be 0."""
        # Only H4 provided, and it's insufficient
        h4_bars = _make_bars(5, tf_minutes=240)

        result = compute_tf_bias(h4_bars=h4_bars)

        # No usable components → zero confidence, WEAK
        assert result.final_confidence == 0.0
        assert result.final_strength == BiasStrength.WEAK

    def test_insufficient_component_excluded_from_alignment(self):
        """Insufficient component should not count toward alignment score."""
        h4_bars = _make_bars(20, tf_minutes=240)
        m15_bars = _make_bars(200, tf_minutes=15)

        result = compute_tf_bias(h4_bars=h4_bars, m15_bars=m15_bars)

        # Only M15 should contribute to alignment (1 TF)
        # H4 is insufficient → excluded from total_count
        assert result.h4_bias is not None
        assert result.h4_bias.factors.get("error") == "insufficient_data"
        # alignment_score = agreeing / total; with 1 usable TF, should be 1.0
        assert result.alignment_score == 1.0

    def test_insufficient_component_excluded_from_conflicts(self):
        """Insufficient component should not appear in conflicting_timeframes."""
        h4_bars = _make_bars(20, tf_minutes=240)
        m15_bars = _make_bars(200, tf_minutes=15)

        result = compute_tf_bias(h4_bars=h4_bars, m15_bars=m15_bars)

        assert "4h" not in result.conflicting_timeframes

    def test_h4_transitions_from_excluded_to_included(self):
        """Proves H4 becomes usable once warmup is satisfied.

        Early window: 50 H4 bars → insufficient (needs 210).
        Late window:  220 H4 bars → usable, contributes to scoring.
        """
        m15_bars = _make_bars(200, tf_minutes=15)

        # Early: insufficient H4
        early_h4 = _make_bars(50, tf_minutes=240)
        early = compute_tf_bias(h4_bars=early_h4, m15_bars=m15_bars)
        assert early.h4_bias is not None
        assert early.h4_bias.factors.get("error") == "insufficient_data"

        # Late: sufficient H4
        late_h4 = _make_bars(220, tf_minutes=240)
        late = compute_tf_bias(h4_bars=late_h4, m15_bars=m15_bars)
        assert late.h4_bias is not None
        assert late.h4_bias.factors.get("error") is None
        assert late.h4_bias.confidence > 0.0
