"""Unit tests for the Policy Engine and rules."""

import uuid
from datetime import datetime

import pytest

from app.schemas import (
    TradeIntent,
    IntentAction,
    PolicyReason,
    CurrentState,
)
from app.services.policy_engine import (
    PolicyEngine,
    PolicyRule,
    RuleResult,
    KillSwitchRule,
    DriftGuardRule,
    MaxDrawdownRule,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_intent() -> TradeIntent:
    """Create a sample trade intent for testing."""
    return TradeIntent(
        correlation_id="test-correlation-123",
        workspace_id=uuid.uuid4(),
        action=IntentAction.OPEN_LONG,
        strategy_entity_id=uuid.uuid4(),
        symbol="BTCUSDT",
        timeframe="1h",
        quantity=0.1,
        signal_strength=0.8,
        reason="Test intent",
    )


@pytest.fixture
def default_state() -> CurrentState:
    """Create default (safe) current state."""
    return CurrentState(
        kill_switch_active=False,
        trading_enabled=True,
    )


# =============================================================================
# KillSwitchRule Tests
# =============================================================================


class TestKillSwitchRule:
    """Tests for the KillSwitchRule."""

    def test_passes_when_kill_switch_inactive(self, sample_intent, default_state):
        """Rule passes when kill switch is not active."""
        rule = KillSwitchRule()
        result = rule.evaluate(sample_intent, default_state)

        assert result.passed is True
        assert result.reason is None

    def test_rejects_when_kill_switch_active(self, sample_intent):
        """Rule rejects when kill switch is active."""
        state = CurrentState(kill_switch_active=True)
        rule = KillSwitchRule()
        result = rule.evaluate(sample_intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.KILL_SWITCH_ACTIVE
        assert "kill switch" in result.reason_details.lower()

    def test_rejects_when_trading_disabled(self, sample_intent):
        """Rule rejects when trading is disabled."""
        state = CurrentState(kill_switch_active=False, trading_enabled=False)
        rule = KillSwitchRule()
        result = rule.evaluate(sample_intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.KILL_SWITCH_ACTIVE
        assert "disabled" in result.reason_details.lower()

    def test_has_highest_priority(self):
        """Kill switch should have priority 0 (highest)."""
        rule = KillSwitchRule()
        assert rule.priority == 0
        assert rule.name == "kill_switch"


# =============================================================================
# DriftGuardRule Tests
# =============================================================================


class TestDriftGuardRule:
    """Tests for the DriftGuardRule."""

    def test_passes_when_no_regime_data(self, sample_intent, default_state):
        """Rule passes with warning when no regime data available."""
        rule = DriftGuardRule()
        result = rule.evaluate(sample_intent, default_state)

        assert result.passed is True
        assert len(result.warnings) == 1
        assert "skipped" in result.warnings[0].lower()

    def test_passes_when_z_score_low(self, sample_intent):
        """Rule passes when z-score is below warning threshold."""
        state = CurrentState(regime_distance_z=1.5)
        rule = DriftGuardRule(warning_threshold=2.0, reject_threshold=3.0)
        result = rule.evaluate(sample_intent, state)

        assert result.passed is True
        assert len(result.warnings) == 0

    def test_warns_when_z_score_moderate(self, sample_intent):
        """Rule passes with warning when z-score is moderate."""
        state = CurrentState(regime_distance_z=2.5)
        rule = DriftGuardRule(warning_threshold=2.0, reject_threshold=3.0)
        result = rule.evaluate(sample_intent, state)

        assert result.passed is True
        assert len(result.warnings) == 1
        assert "drift" in result.warnings[0].lower()

    def test_rejects_when_z_score_high(self, sample_intent):
        """Rule rejects when z-score exceeds reject threshold."""
        state = CurrentState(regime_distance_z=3.5)
        rule = DriftGuardRule(warning_threshold=2.0, reject_threshold=3.0)
        result = rule.evaluate(sample_intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.REGIME_DRIFT
        assert "3.50" in result.reason_details

    def test_rejects_at_exact_threshold(self, sample_intent):
        """Rule rejects at exactly the threshold."""
        state = CurrentState(regime_distance_z=3.0)
        rule = DriftGuardRule(reject_threshold=3.0)
        result = rule.evaluate(sample_intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.REGIME_DRIFT

    def test_custom_thresholds(self, sample_intent):
        """Custom thresholds are respected."""
        state = CurrentState(regime_distance_z=1.5)

        # With default thresholds (2.0, 3.0) - should pass
        rule_default = DriftGuardRule()
        assert rule_default.evaluate(sample_intent, state).passed is True

        # With stricter thresholds - should reject
        rule_strict = DriftGuardRule(warning_threshold=1.0, reject_threshold=1.2)
        assert rule_strict.evaluate(sample_intent, state).passed is False


# =============================================================================
# MaxDrawdownRule Tests
# =============================================================================


class TestMaxDrawdownRule:
    """Tests for the MaxDrawdownRule."""

    def test_passes_when_no_drawdown_data(self, sample_intent, default_state):
        """Rule passes with warning when no drawdown data."""
        rule = MaxDrawdownRule()
        result = rule.evaluate(sample_intent, default_state)

        assert result.passed is True
        assert len(result.warnings) == 1

    def test_passes_for_closing_positions(self):
        """Rule always passes for closing positions."""
        state = CurrentState(max_drawdown_today=10.0)  # Over limit
        rule = MaxDrawdownRule(max_drawdown_pct=5.0)

        close_actions = [
            IntentAction.CLOSE_LONG,
            IntentAction.CLOSE_SHORT,
            IntentAction.SCALE_OUT,
            IntentAction.CANCEL_ORDER,
        ]

        for action in close_actions:
            intent = TradeIntent(
                correlation_id="test",
                workspace_id=uuid.uuid4(),
                action=action,
                strategy_entity_id=uuid.uuid4(),
                symbol="BTCUSDT",
                timeframe="1h",
            )
            result = rule.evaluate(intent, state)
            assert result.passed is True, f"Should pass for {action}"

    def test_rejects_opening_when_drawdown_exceeded(self, sample_intent):
        """Rule rejects opening positions when drawdown exceeds limit."""
        state = CurrentState(max_drawdown_today=6.0)
        rule = MaxDrawdownRule(max_drawdown_pct=5.0)
        result = rule.evaluate(sample_intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.MAX_DRAWDOWN_EXCEEDED

    def test_passes_when_drawdown_below_limit(self, sample_intent):
        """Rule passes when drawdown is below limit."""
        state = CurrentState(max_drawdown_today=3.0)
        rule = MaxDrawdownRule(max_drawdown_pct=5.0)
        result = rule.evaluate(sample_intent, state)

        assert result.passed is True


# =============================================================================
# PolicyEngine Tests
# =============================================================================


class TestPolicyEngine:
    """Tests for the PolicyEngine."""

    def test_default_rules_loaded(self):
        """Engine loads default rules on init."""
        engine = PolicyEngine()
        rule_names = [r.name for r in engine.rules]

        assert "kill_switch" in rule_names
        assert "drift_guard" in rule_names
        assert "max_drawdown" in rule_names

    def test_rules_sorted_by_priority(self):
        """Rules are sorted by priority (lowest first)."""
        engine = PolicyEngine()
        priorities = [r.priority for r in engine.rules]

        assert priorities == sorted(priorities)

    def test_approves_when_all_rules_pass(self, sample_intent, default_state):
        """Engine approves when all rules pass."""
        engine = PolicyEngine()
        decision = engine.evaluate(sample_intent, default_state)

        assert decision.approved is True
        assert decision.reason == PolicyReason.ALL_RULES_PASSED
        assert len(decision.rules_failed) == 0
        assert len(decision.rules_passed) > 0

    def test_rejects_when_kill_switch_active(self, sample_intent):
        """Engine rejects immediately when kill switch is active."""
        state = CurrentState(kill_switch_active=True)
        engine = PolicyEngine()
        decision = engine.evaluate(sample_intent, state)

        assert decision.approved is False
        assert decision.reason == PolicyReason.KILL_SWITCH_ACTIVE
        # Should short-circuit - only kill_switch evaluated
        assert "kill_switch" in decision.rules_failed
        assert len(decision.rules_evaluated) == 1

    def test_short_circuits_on_first_failure(self, sample_intent):
        """Engine stops evaluating after first rule failure."""
        state = CurrentState(
            kill_switch_active=False,
            regime_distance_z=5.0,  # High drift - will fail
        )
        engine = PolicyEngine()
        decision = engine.evaluate(sample_intent, state)

        assert decision.approved is False
        assert decision.reason == PolicyReason.REGIME_DRIFT
        # kill_switch passed, drift_guard failed, max_drawdown not evaluated
        assert "kill_switch" in decision.rules_passed
        assert "drift_guard" in decision.rules_failed
        # max_drawdown shouldn't be in evaluated list (short-circuit)

    def test_collects_warnings(self, sample_intent):
        """Engine collects warnings from all rules."""
        state = CurrentState(
            kill_switch_active=False,
            regime_distance_z=2.5,  # Warning level
            max_drawdown_today=None,  # Will generate warning
        )
        engine = PolicyEngine()
        decision = engine.evaluate(sample_intent, state)

        assert decision.approved is True
        assert len(decision.warnings) >= 1

    def test_custom_rules(self, sample_intent, default_state):
        """Engine works with custom rules."""

        class AlwaysRejectRule(PolicyRule):
            @property
            def name(self):
                return "always_reject"

            @property
            def priority(self):
                return 50

            def evaluate(self, intent, state):
                return RuleResult(
                    passed=False,
                    reason=PolicyReason.STRATEGY_DISABLED,
                    reason_details="Test rejection",
                )

        engine = PolicyEngine(rules=[KillSwitchRule(), AlwaysRejectRule()])
        decision = engine.evaluate(sample_intent, default_state)

        assert decision.approved is False
        assert decision.reason == PolicyReason.STRATEGY_DISABLED

    def test_evaluation_timing_recorded(self, sample_intent, default_state):
        """Engine records evaluation time."""
        engine = PolicyEngine()
        decision = engine.evaluate(sample_intent, default_state)

        assert decision.evaluation_ms is not None
        assert decision.evaluation_ms >= 0

    def test_handles_rule_exception(self, sample_intent, default_state):
        """Engine handles exceptions in rules safely."""

        class BrokenRule(PolicyRule):
            @property
            def name(self):
                return "broken"

            @property
            def priority(self):
                return 5  # After kill_switch

            def evaluate(self, intent, state):
                raise RuntimeError("Rule exploded!")

        engine = PolicyEngine(rules=[KillSwitchRule(), BrokenRule()])
        decision = engine.evaluate(sample_intent, default_state)

        # Should reject safely on exception
        assert decision.approved is False
        assert "exception" in decision.reason_details.lower()
        assert "broken" in decision.rules_failed

    def test_add_rule(self, sample_intent, default_state):
        """Can add rules dynamically."""
        engine = PolicyEngine(rules=[KillSwitchRule()])
        assert len(engine.rules) == 1

        engine.add_rule(DriftGuardRule())
        assert len(engine.rules) == 2

        # Rules should be re-sorted
        assert engine.rules[0].name == "kill_switch"  # priority 0
        assert engine.rules[1].name == "drift_guard"  # priority 10
