"""
Regime stability FSM (hysteresis guard) for v1.5 Live Intelligence.

Prevents regime flicker by requiring M consecutive bars of a candidate
regime with median confidence >= C_enter before confirming a transition.

Key parameters:
- M: persistence bars required (default: 20)
- C_enter: confidence threshold to confirm transition (default: 0.75)
- C_exit: confidence threshold to consider change (default: 0.55)

State machine:
1. First regime becomes immediately stable (no history)
2. Different regime with confidence >= C_exit starts candidate tracking
3. Same candidate for M bars with median confidence >= C_enter -> transition
4. Back to stable regime or different candidate -> reset candidate state
"""

from dataclasses import dataclass, field
from statistics import median
from typing import Optional


@dataclass
class FSMConfig:
    """Configuration for regime FSM."""

    M: int = 20  # Persistence bars required
    C_enter: float = 0.75  # Confidence threshold to confirm transition
    C_exit: float = 0.55  # Confidence threshold to consider regime change


@dataclass
class FSMState:
    """Current state of the regime FSM."""

    stable_regime_key: Optional[str] = None
    candidate_regime_key: Optional[str] = None
    candidate_count: int = 0
    regime_age_bars: int = 0


@dataclass
class RegimeTransitionEvent:
    """Event emitted when regime transition is confirmed."""

    from_key: str
    to_key: str
    confidence_history: list[float]
    median_confidence: float
    previous_regime_age: int


class RegimeFSM:
    """
    Finite state machine for regime stability with hysteresis.

    Implements a guard against regime flicker by requiring persistence
    of candidate regime over M bars with sufficient confidence.
    """

    def __init__(self, config: Optional[FSMConfig] = None):
        """
        Initialize FSM.

        Args:
            config: FSM configuration. Uses defaults if not provided.
        """
        self._config = config or FSMConfig()
        self._state = FSMState()
        self._candidate_confidences: list[float] = []

    def update(
        self, regime_key: str, confidence: float
    ) -> Optional[RegimeTransitionEvent]:
        """
        Update FSM with new regime observation.

        Args:
            regime_key: Canonical regime key (e.g., "trend=uptrend|vol=high_vol")
            confidence: Confidence score for this observation (0-1)

        Returns:
            RegimeTransitionEvent if transition confirmed, None otherwise
        """
        # Case 1: First observation - initialize stable regime
        if self._state.stable_regime_key is None:
            self._state.stable_regime_key = regime_key
            self._state.regime_age_bars = 1
            return None

        # Case 2: Same as stable regime - increment age and clear candidate
        if regime_key == self._state.stable_regime_key:
            self._state.regime_age_bars += 1
            self._clear_candidate()
            return None

        # Case 3: Different regime - check if confidence meets C_exit threshold
        if confidence < self._config.C_exit:
            # Below C_exit: ignore as noise, but still increment age
            self._state.regime_age_bars += 1
            return None

        # Case 4: Different candidate than current candidate - reset tracking
        if (
            self._state.candidate_regime_key is not None
            and regime_key != self._state.candidate_regime_key
        ):
            self._clear_candidate()

        # Track this candidate
        if self._state.candidate_regime_key is None:
            self._state.candidate_regime_key = regime_key

        self._state.candidate_count += 1
        self._candidate_confidences.append(confidence)

        # Check if candidate has persisted for M bars
        if self._state.candidate_count >= self._config.M:
            # Calculate median confidence
            med_conf = median(self._candidate_confidences)

            # Check if median meets C_enter threshold
            if med_conf >= self._config.C_enter:
                return self._confirm_transition(regime_key, med_conf)

        return None

    def _confirm_transition(
        self, new_key: str, med_conf: float
    ) -> RegimeTransitionEvent:
        """Confirm transition to new regime and emit event."""
        event = RegimeTransitionEvent(
            from_key=self._state.stable_regime_key,
            to_key=new_key,
            confidence_history=list(self._candidate_confidences),
            median_confidence=med_conf,
            previous_regime_age=self._state.regime_age_bars,
        )

        # Update state
        self._state.stable_regime_key = new_key
        self._state.regime_age_bars = 1
        self._clear_candidate()

        return event

    def _clear_candidate(self) -> None:
        """Clear candidate tracking state."""
        self._state.candidate_regime_key = None
        self._state.candidate_count = 0
        self._candidate_confidences.clear()

    def get_state(self) -> FSMState:
        """
        Get current FSM state.

        Returns:
            Copy of current state (read-only snapshot)
        """
        return FSMState(
            stable_regime_key=self._state.stable_regime_key,
            candidate_regime_key=self._state.candidate_regime_key,
            candidate_count=self._state.candidate_count,
            regime_age_bars=self._state.regime_age_bars,
        )

    def reset(self) -> None:
        """Reset FSM to initial state."""
        self._state = FSMState()
        self._candidate_confidences.clear()
