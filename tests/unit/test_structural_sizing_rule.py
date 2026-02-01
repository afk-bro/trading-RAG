"""Unit tests for the StructuralSizingRule."""

import json
import uuid

import pytest

from app.schemas import (
    TradeIntent,
    IntentAction,
    PolicyReason,
    CurrentState,
)
from app.services.policy_engine import StructuralSizingRule


@pytest.fixture
def rule():
    return StructuralSizingRule()


def _make_intent(
    action=IntentAction.OPEN_LONG,
    symbol="MNQ",
    price=None,
    stop_loss=None,
    quantity=1.0,
    metadata=None,
):
    return TradeIntent(
        correlation_id="test-sizing",
        workspace_id=uuid.uuid4(),
        action=action,
        strategy_entity_id=uuid.uuid4(),
        symbol=symbol,
        timeframe="5m",
        quantity=quantity,
        price=price,
        stop_loss=stop_loss,
        metadata=metadata or {},
    )


def _make_state(risk_budget_dollars=None, risk_multiplier=1.0):
    return CurrentState(
        risk_budget_dollars=risk_budget_dollars,
        risk_multiplier=risk_multiplier,
    )


class TestStructuralSizingRule:
    def test_normal_sizing_mnq(self, rule):
        """MNQ: entry=18000, stop=17990, budget=$500 → 25 contracts.
        risk_per_contract = 10 * $2 = $20, 500/20 = 25
        """
        intent = _make_intent(price=18000.0, stop_loss=17990.0, symbol="MNQ")
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity == 25.0

    def test_normal_sizing_nq(self, rule):
        """NQ: entry=18000, stop=17990, budget=$500 → 2 contracts.
        risk_per_contract = 10 * $20 = $200, floor(500/200) = 2
        """
        intent = _make_intent(price=18000.0, stop_loss=17990.0, symbol="NQ")
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity == 2.0

    def test_reject_risk_too_high(self, rule):
        """Wide stop + small budget → 0 contracts → reject."""
        # 100 point stop on NQ = $2000 risk, budget $500 → 0
        intent = _make_intent(price=18000.0, stop_loss=17900.0, symbol="NQ")
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.RISK_TOO_HIGH_FOR_ACCOUNT

    def test_pass_through_close_action(self, rule):
        """CLOSE_LONG bypasses sizing regardless of state."""
        intent = _make_intent(action=IntentAction.CLOSE_LONG)
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity is None

    def test_reject_missing_stop_when_sizing_enabled(self, rule):
        """Budget set but no stop → reject (not pass-through)."""
        intent = _make_intent(price=18000.0, stop_loss=None)
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.RISK_TOO_HIGH_FOR_ACCOUNT
        assert "stop loss" in result.reason_details.lower()

    def test_reject_missing_price_when_sizing_enabled(self, rule):
        """Budget set but no price → reject."""
        intent = _make_intent(price=None, stop_loss=17990.0)
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.RISK_TOO_HIGH_FOR_ACCOUNT
        assert "entry price" in result.reason_details.lower()

    def test_pass_through_no_budget(self, rule):
        """Budget None → pass-through (sizing not enabled)."""
        intent = _make_intent(price=18000.0, stop_loss=17990.0)
        state = _make_state(risk_budget_dollars=None)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity is None

    def test_half_size_multiplier(self, rule):
        """risk_multiplier=0.5 halves contracts.
        MNQ: 10pt stop * $2 = $20/contract, budget $500 * 0.5 = $250, floor(250/20) = 12
        """
        intent = _make_intent(price=18000.0, stop_loss=17990.0, symbol="MNQ")
        state = _make_state(risk_budget_dollars=500.0, risk_multiplier=0.5)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity == 12.0

    def test_multiplier_causes_reject(self, rule):
        """Low multiplier pushes contracts to 0 → reject."""
        # NQ: 10pt * $20 = $200/contract, budget $500 * 0.1 = $50, floor(50/200) = 0
        intent = _make_intent(price=18000.0, stop_loss=17990.0, symbol="NQ")
        state = _make_state(risk_budget_dollars=500.0, risk_multiplier=0.1)
        result = rule.evaluate(intent, state)

        assert result.passed is False
        assert result.reason == PolicyReason.RISK_TOO_HIGH_FOR_ACCOUNT

    def test_debug_details_in_rejection(self, rule):
        """reason_details contains all debug fields."""
        intent = _make_intent(price=18000.0, stop_loss=17900.0, symbol="NQ")
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is False
        debug = json.loads(result.reason_details)
        expected_keys = {
            "entry",
            "stop",
            "stop_points",
            "point_value",
            "risk_per_contract",
            "risk_budget",
            "risk_multiplier",
            "effective_budget",
            "computed_contracts",
        }
        assert expected_keys == set(debug.keys())

    def test_zero_stop_distance(self, rule):
        """Entry == stop passes with warning."""
        intent = _make_intent(price=18000.0, stop_loss=18000.0, symbol="MNQ")
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert len(result.warnings) == 1
        assert "zero" in result.warnings[0].lower()

    def test_modified_quantity_is_whole_number(self, rule):
        """Contracts are always integer-valued (as float)."""
        # MNQ: 7pt stop * $2 = $14/contract, budget $100, floor(100/14) = 7
        intent = _make_intent(price=18000.0, stop_loss=17993.0, symbol="MNQ")
        state = _make_state(risk_budget_dollars=100.0)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity == float(int(result.modified_quantity))

    def test_priority(self, rule):
        assert rule.priority == 50
        assert rule.name == "structural_sizing"

    def test_short_side_open_short(self, rule):
        """OPEN_SHORT also gets sized (not a passthrough action)."""
        intent = _make_intent(
            action=IntentAction.OPEN_SHORT,
            price=18000.0,
            stop_loss=18010.0,
            symbol="MNQ",
        )
        state = _make_state(risk_budget_dollars=500.0)
        result = rule.evaluate(intent, state)

        assert result.passed is True
        assert result.modified_quantity == 25.0  # 10pt * $2 = $20, 500/20 = 25
