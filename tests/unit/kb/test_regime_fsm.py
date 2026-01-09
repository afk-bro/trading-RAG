"""Tests for regime stability FSM (hysteresis guard)."""

import pytest
from app.services.kb.regime_fsm import (
    RegimeFSM,
    FSMConfig,
    FSMState,
    RegimeTransitionEvent,
)


class TestFSMBasicBehavior:
    """Tests for basic FSM state transitions."""

    def test_initial_state(self):
        """FSM starts with no stable regime until first update."""
        fsm = RegimeFSM(config=FSMConfig())
        state = fsm.get_state()
        assert state.stable_regime_key is None
        assert state.candidate_regime_key is None
        assert state.candidate_count == 0

    def test_first_regime_immediately_stable(self):
        """First regime becomes stable immediately (no history to compare)."""
        fsm = RegimeFSM(config=FSMConfig(M=20))
        event = fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        state = fsm.get_state()
        assert state.stable_regime_key == "trend=uptrend|vol=high_vol"
        assert event is None  # No transition event for initialization

    def test_same_regime_clears_candidate(self):
        """Receiving same regime as stable clears any candidate."""
        fsm = RegimeFSM(config=FSMConfig(M=5))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        # Start building candidate
        fsm.update("trend=flat|vol=high_vol", confidence=0.7)
        fsm.update("trend=flat|vol=high_vol", confidence=0.7)
        assert fsm.get_state().candidate_count == 2

        # Back to stable regime
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        assert fsm.get_state().candidate_count == 0
        assert fsm.get_state().candidate_regime_key is None


class TestFSMHysteresis:
    """Tests for hysteresis behavior (C_enter vs C_exit)."""

    def test_low_confidence_ignored(self):
        """Regime changes below C_exit are ignored as noise."""
        fsm = RegimeFSM(config=FSMConfig(M=5, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        # Low confidence change should be ignored
        fsm.update("trend=flat|vol=high_vol", confidence=0.50)
        assert fsm.get_state().candidate_count == 0

    def test_candidate_builds_above_c_exit(self):
        """Regime changes above C_exit start building candidate."""
        fsm = RegimeFSM(config=FSMConfig(M=5, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        # Above C_exit, candidate builds
        fsm.update("trend=flat|vol=high_vol", confidence=0.60)
        assert fsm.get_state().candidate_count == 1
        assert fsm.get_state().candidate_regime_key == "trend=flat|vol=high_vol"


class TestFSMTransitionConfirmation:
    """Tests for transition confirmation logic."""

    def test_transition_requires_m_bars(self):
        """Transition only confirms after M consecutive bars."""
        fsm = RegimeFSM(config=FSMConfig(M=3, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        # 2 bars is not enough
        fsm.update("trend=flat|vol=high_vol", confidence=0.75)
        fsm.update("trend=flat|vol=high_vol", confidence=0.75)
        assert fsm.get_state().stable_regime_key == "trend=uptrend|vol=high_vol"

        # 3rd bar triggers transition
        event = fsm.update("trend=flat|vol=high_vol", confidence=0.75)
        assert fsm.get_state().stable_regime_key == "trend=flat|vol=high_vol"
        assert event is not None
        assert event.from_key == "trend=uptrend|vol=high_vol"
        assert event.to_key == "trend=flat|vol=high_vol"

    def test_transition_requires_median_confidence(self):
        """Transition requires median confidence >= C_enter over M bars."""
        fsm = RegimeFSM(config=FSMConfig(M=3, C_enter=0.75, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        # Candidate builds but median confidence too low
        fsm.update("trend=flat|vol=high_vol", confidence=0.60)
        fsm.update("trend=flat|vol=high_vol", confidence=0.60)
        fsm.update("trend=flat|vol=high_vol", confidence=0.60)

        # Still on original regime (median 0.60 < 0.75)
        assert fsm.get_state().stable_regime_key == "trend=uptrend|vol=high_vol"

    def test_different_candidate_resets_count(self):
        """Switching to different candidate resets the count."""
        fsm = RegimeFSM(config=FSMConfig(M=5, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        # Build candidate A
        fsm.update("trend=flat|vol=high_vol", confidence=0.75)
        fsm.update("trend=flat|vol=high_vol", confidence=0.75)
        assert fsm.get_state().candidate_count == 2

        # Switch to candidate B
        fsm.update("trend=downtrend|vol=high_vol", confidence=0.75)
        assert fsm.get_state().candidate_count == 1
        assert fsm.get_state().candidate_regime_key == "trend=downtrend|vol=high_vol"


class TestFSMRegimeAge:
    """Tests for regime age tracking."""

    def test_age_increments_on_same_regime(self):
        """Age increments each bar while in same stable regime."""
        fsm = RegimeFSM(config=FSMConfig(M=20))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        for i in range(5):
            fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        assert fsm.get_state().regime_age_bars == 6  # 1 initial + 5 updates

    def test_age_resets_on_transition(self):
        """Age resets to 1 after regime transition."""
        fsm = RegimeFSM(config=FSMConfig(M=2, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        assert fsm.get_state().regime_age_bars == 3

        # Transition
        fsm.update("trend=flat|vol=high_vol", confidence=0.8)
        fsm.update("trend=flat|vol=high_vol", confidence=0.8)

        assert fsm.get_state().regime_age_bars == 1  # Reset after transition


class TestFSMReset:
    """Tests for FSM reset behavior."""

    def test_reset_clears_state(self):
        """Reset clears all FSM state."""
        fsm = RegimeFSM(config=FSMConfig(M=20))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        fsm.update("trend=flat|vol=high_vol", confidence=0.7)

        fsm.reset()
        state = fsm.get_state()

        assert state.stable_regime_key is None
        assert state.candidate_regime_key is None
        assert state.candidate_count == 0
        assert state.regime_age_bars == 0


class TestFSMTransitionEvent:
    """Tests for transition event details."""

    def test_transition_event_contains_confidence_history(self):
        """Transition event includes confidence history from candidate period."""
        fsm = RegimeFSM(config=FSMConfig(M=3, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        fsm.update("trend=flat|vol=high_vol", confidence=0.72)
        fsm.update("trend=flat|vol=high_vol", confidence=0.78)
        event = fsm.update("trend=flat|vol=high_vol", confidence=0.75)

        assert event is not None
        assert event.confidence_history == [0.72, 0.78, 0.75]
        assert event.median_confidence == 0.75  # Median of [0.72, 0.75, 0.78]

    def test_transition_event_includes_previous_age(self):
        """Transition event includes age of previous regime."""
        fsm = RegimeFSM(config=FSMConfig(M=2, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        # Age is now 3

        fsm.update("trend=flat|vol=high_vol", confidence=0.8)
        event = fsm.update("trend=flat|vol=high_vol", confidence=0.8)

        assert event is not None
        assert event.previous_regime_age == 3


class TestFSMEdgeCases:
    """Tests for edge cases."""

    def test_m_equals_one(self):
        """With M=1, transition happens immediately after first candidate bar."""
        fsm = RegimeFSM(config=FSMConfig(M=1, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        event = fsm.update("trend=flat|vol=high_vol", confidence=0.75)

        assert event is not None
        assert fsm.get_state().stable_regime_key == "trend=flat|vol=high_vol"

    def test_confidence_exactly_at_threshold(self):
        """Confidence exactly at C_exit should trigger candidate building."""
        fsm = RegimeFSM(config=FSMConfig(M=5, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        fsm.update("trend=flat|vol=high_vol", confidence=0.55)  # Exactly at threshold
        assert fsm.get_state().candidate_count == 1

    def test_confidence_exactly_at_c_enter(self):
        """Median confidence exactly at C_enter should allow transition."""
        fsm = RegimeFSM(config=FSMConfig(M=3, C_enter=0.70, C_exit=0.55))
        fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)

        fsm.update("trend=flat|vol=high_vol", confidence=0.70)
        fsm.update("trend=flat|vol=high_vol", confidence=0.70)
        event = fsm.update("trend=flat|vol=high_vol", confidence=0.70)

        assert event is not None
        assert fsm.get_state().stable_regime_key == "trend=flat|vol=high_vol"

    def test_no_transition_without_initialization(self):
        """Update without prior state returns None event."""
        fsm = RegimeFSM(config=FSMConfig())
        event = fsm.update("trend=uptrend|vol=high_vol", confidence=0.8)
        assert event is None  # Initialization, not transition
