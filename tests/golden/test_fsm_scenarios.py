"""
Golden tests for FSM determinism.

These tests use JSON fixtures that define:
1. FSM configuration (M, C_enter, C_exit)
2. A sequence of input steps (raw_key, confidence)
3. Expected state after each step
4. Whether a transition event should fire

Golden tests ensure the FSM produces consistent, documented behavior
across all edge cases including:
- Stable regime persistence (age incrementing, candidate clearing)
- Clean transitions (event firing, age reset)
- Flicker suppression (candidate count resetting)
- Low confidence noise filtering
- Hysteresis boundary behavior (C_exit vs C_enter)
"""

import json
import pytest
from pathlib import Path

from app.services.kb.regime_fsm import RegimeFSM, FSMConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_scenario(name: str) -> dict:
    """Load golden scenario from JSON fixture file."""
    path = FIXTURES_DIR / f"fsm_{name}.json"
    with open(path) as f:
        return json.load(f)


class TestFSMGoldenScenarios:
    """Golden tests for FSM behavior using recorded scenarios."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "stable_regime",
            "clean_transition",
            "flicker_suppressed",
            "low_confidence_ignored",
            "hysteresis_boundary",
        ],
    )
    def test_scenario(self, scenario_name: str):
        """
        Run golden scenario and verify outputs match expected values.

        Each scenario defines a sequence of FSM updates with expected
        state after each step, allowing deterministic verification of
        FSM behavior.
        """
        scenario = load_scenario(scenario_name)

        config = FSMConfig(**scenario["config"])
        fsm = RegimeFSM(config=config)

        for i, step in enumerate(scenario["steps"]):
            step_comment = step.get("comment", f"Step {i}")

            event = fsm.update(
                regime_key=step["raw_key"],
                confidence=step["confidence"],
            )

            state = fsm.get_state()
            expected = step["expected_state"]

            # Verify stable regime key
            assert state.stable_regime_key == expected["stable_key"], (
                f"{step_comment}: stable_key mismatch. "
                f"Got '{state.stable_regime_key}', expected '{expected['stable_key']}'"
            )

            # Verify candidate regime key
            assert state.candidate_regime_key == expected.get("candidate_key"), (
                f"{step_comment}: candidate_key mismatch. "
                f"Got '{state.candidate_regime_key}', "
                f"expected '{expected.get('candidate_key')}'"
            )

            # Verify candidate count
            assert state.candidate_count == expected["candidate_count"], (
                f"{step_comment}: candidate_count mismatch. "
                f"Got {state.candidate_count}, expected {expected['candidate_count']}"
            )

            # Verify regime age
            assert state.regime_age_bars == expected["regime_age_bars"], (
                f"{step_comment}: regime_age_bars mismatch. "
                f"Got {state.regime_age_bars}, expected {expected['regime_age_bars']}"
            )

            # Verify transition event
            expected_transition = step.get("transition", False)
            assert (event is not None) == expected_transition, (
                f"{step_comment}: transition mismatch. "
                f"Got event={event is not None}, expected transition={expected_transition}"
            )

            # If transition occurred, verify event details
            if event is not None and "transition_event" in step:
                expected_event = step["transition_event"]

                assert event.from_key == expected_event["from_key"], (
                    f"{step_comment}: transition from_key mismatch. "
                    f"Got '{event.from_key}', expected '{expected_event['from_key']}'"
                )

                assert event.to_key == expected_event["to_key"], (
                    f"{step_comment}: transition to_key mismatch. "
                    f"Got '{event.to_key}', expected '{expected_event['to_key']}'"
                )

                if "median_confidence" in expected_event:
                    assert (
                        abs(
                            event.median_confidence
                            - expected_event["median_confidence"]
                        )
                        < 0.01
                    ), (
                        f"{step_comment}: median_confidence mismatch. "
                        f"Got {event.median_confidence}, "
                        f"expected {expected_event['median_confidence']}"
                    )


class TestFSMGoldenFixtureIntegrity:
    """Tests to verify golden fixtures are well-formed."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "stable_regime",
            "clean_transition",
            "flicker_suppressed",
            "low_confidence_ignored",
            "hysteresis_boundary",
        ],
    )
    def test_fixture_has_required_fields(self, scenario_name: str):
        """Verify each fixture has all required fields."""
        scenario = load_scenario(scenario_name)

        # Top-level required fields
        assert "name" in scenario, "Missing 'name' field"
        assert "description" in scenario, "Missing 'description' field"
        assert "config" in scenario, "Missing 'config' field"
        assert "steps" in scenario, "Missing 'steps' field"

        # Config required fields
        config = scenario["config"]
        assert "M" in config, "Missing 'M' in config"
        assert "C_enter" in config, "Missing 'C_enter' in config"
        assert "C_exit" in config, "Missing 'C_exit' in config"

        # Steps required fields
        assert (
            len(scenario["steps"]) >= 10
        ), f"Fixture should have at least 10 steps, got {len(scenario['steps'])}"

        for i, step in enumerate(scenario["steps"]):
            assert "raw_key" in step, f"Step {i}: missing 'raw_key'"
            assert "confidence" in step, f"Step {i}: missing 'confidence'"
            assert "expected_state" in step, f"Step {i}: missing 'expected_state'"

            expected = step["expected_state"]
            assert "stable_key" in expected, f"Step {i}: missing 'stable_key'"
            assert "candidate_count" in expected, f"Step {i}: missing 'candidate_count'"
            assert "regime_age_bars" in expected, f"Step {i}: missing 'regime_age_bars'"

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "stable_regime",
            "clean_transition",
            "flicker_suppressed",
            "low_confidence_ignored",
            "hysteresis_boundary",
        ],
    )
    def test_fixture_config_valid(self, scenario_name: str):
        """Verify fixture config has valid values."""
        scenario = load_scenario(scenario_name)
        config = scenario["config"]

        assert config["M"] >= 1, "M must be >= 1"
        assert (
            0 < config["C_exit"] < config["C_enter"] <= 1.0
        ), "Must have 0 < C_exit < C_enter <= 1.0"

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "stable_regime",
            "clean_transition",
            "flicker_suppressed",
            "low_confidence_ignored",
            "hysteresis_boundary",
        ],
    )
    def test_fixture_confidence_values_valid(self, scenario_name: str):
        """Verify all confidence values are in valid range."""
        scenario = load_scenario(scenario_name)

        for i, step in enumerate(scenario["steps"]):
            conf = step["confidence"]
            assert 0 <= conf <= 1.0, f"Step {i}: confidence {conf} out of range [0, 1]"


class TestFSMGoldenScenarioDocumentation:
    """Tests that verify each scenario tests what it claims to test."""

    def test_stable_regime_scenario_tests_stability(self):
        """Verify stable_regime scenario never has a transition."""
        scenario = load_scenario("stable_regime")

        transitions = [s for s in scenario["steps"] if s.get("transition", False)]
        assert (
            len(transitions) == 0
        ), "stable_regime scenario should have no transitions"

        # Verify age increments
        ages = [s["expected_state"]["regime_age_bars"] for s in scenario["steps"]]
        assert ages[-1] > ages[0], "Age should increment over time"

    def test_clean_transition_scenario_has_transitions(self):
        """Verify clean_transition scenario has at least one transition."""
        scenario = load_scenario("clean_transition")

        transitions = [s for s in scenario["steps"] if s.get("transition", False)]
        assert (
            len(transitions) >= 1
        ), "clean_transition scenario should have at least one transition"

        # Verify transition resets age to 1
        for step in scenario["steps"]:
            if step.get("transition", False):
                assert (
                    step["expected_state"]["regime_age_bars"] == 1
                ), "After transition, regime_age_bars should be 1"

    def test_flicker_suppressed_scenario_has_no_transitions(self):
        """Verify flicker_suppressed scenario has no transitions despite candidates."""
        scenario = load_scenario("flicker_suppressed")

        transitions = [s for s in scenario["steps"] if s.get("transition", False)]
        assert (
            len(transitions) == 0
        ), "flicker_suppressed scenario should have no transitions"

        # Verify candidates appear and are cleared
        candidates = [
            s
            for s in scenario["steps"]
            if s["expected_state"].get("candidate_key") is not None
        ]
        assert (
            len(candidates) >= 3
        ), "flicker_suppressed should have multiple candidate appearances"

    def test_low_confidence_ignored_scenario_filters_noise(self):
        """Verify low_confidence_ignored scenario properly filters low-confidence signals."""
        scenario = load_scenario("low_confidence_ignored")
        config = scenario["config"]
        c_exit = config["C_exit"]

        # Find steps with low confidence different regimes
        low_conf_different = [
            s
            for s in scenario["steps"]
            if s["confidence"] < c_exit
            and s["raw_key"]
            != scenario["steps"][0]["raw_key"]  # Different from initial
        ]

        assert (
            len(low_conf_different) >= 3
        ), "low_confidence_ignored should have multiple low-confidence different regimes"

        # These should all have candidate_count = 0
        for step in low_conf_different:
            assert (
                step["expected_state"]["candidate_count"] == 0
            ), "Low confidence different regime should not build candidate"

    def test_hysteresis_boundary_tests_c_enter_threshold(self):
        """Verify hysteresis_boundary tests the C_enter threshold behavior."""
        scenario = load_scenario("hysteresis_boundary")
        config = scenario["config"]
        c_enter = config["C_enter"]
        c_exit = config["C_exit"]

        # Find steps with confidence in hysteresis band
        hysteresis_band = [
            s for s in scenario["steps"] if c_exit < s["confidence"] < c_enter
        ]

        assert (
            len(hysteresis_band) >= 3
        ), "hysteresis_boundary should have steps in the (C_exit, C_enter) band"

        # Verify there's at least one case where candidate builds but no transition
        # due to median being below C_enter
        candidate_no_transition = [
            s
            for s in scenario["steps"]
            if s["expected_state"]["candidate_count"] >= config["M"]
            and not s.get("transition", False)
        ]

        assert len(candidate_no_transition) >= 1, (
            "hysteresis_boundary should show candidate persistence without transition "
            "when median < C_enter"
        )
