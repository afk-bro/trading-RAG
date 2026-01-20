"""Unit tests for strategy confidence low alerts (v1.5)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.ops_alerts.models import (
    EvalContext,
    OpsRuleType,
    Severity,
    get_rule,
)
from app.services.ops_alerts.evaluator import OpsAlertEvaluator
from app.repositories.ops_alerts import OpsAlertsRepository, UpsertResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create mock repository."""
    repo = AsyncMock(spec=OpsAlertsRepository)
    repo.upsert.return_value = UpsertResult(
        id=uuid4(),
        is_new=True,
        previous_severity=None,
        current_severity="medium",
        escalated=False,
    )
    repo.resolve_by_dedupe_key.return_value = None
    repo.get_active_dedupe_keys.return_value = set()
    return repo


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock()
    pool.acquire.return_value.__aexit__ = AsyncMock()
    return pool


@pytest.fixture
def evaluator(mock_repo, mock_pool):
    """Create evaluator with mocks."""
    return OpsAlertEvaluator(mock_repo, mock_pool)


@pytest.fixture
def sample_version_id():
    """Sample strategy version ID."""
    return uuid4()


@pytest.fixture
def sample_strategy_id():
    """Sample strategy ID."""
    return uuid4()


def make_snapshot(confidence_score: float, as_of_ts: datetime = None):
    """Helper to create snapshot dict."""
    if as_of_ts is None:
        as_of_ts = datetime.now(timezone.utc)
    return {
        "as_of_ts": as_of_ts,
        "computed_at": as_of_ts,
        "regime": "trend-up|volatility-normal",
        "confidence_score": confidence_score,
        "confidence_components": {
            "performance": 0.8,
            "drawdown": 0.7,
            "stability": 0.6,
            "data_freshness": 0.5,
            "regime_fit": confidence_score,
        },
    }


def make_intel_data(
    version_id,
    strategy_id,
    snapshots: list[dict],
    strategy_name: str = "Test Strategy",
):
    """Helper to create strategy intel dict."""
    return {
        "version_id": version_id,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "version_number": 1,
        "version_tag": "v1.0",
        "snapshots": snapshots,
    }


# =============================================================================
# Model Tests
# =============================================================================


class TestStrategyConfidenceRule:
    """Test rule definition for strategy confidence."""

    def test_rule_exists(self):
        """Strategy confidence rule is defined."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)
        assert rule is not None
        assert rule.rule_type == OpsRuleType.STRATEGY_CONFIDENCE_LOW

    def test_rule_properties(self):
        """Rule has correct properties."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)
        assert rule.requires_strategy_intel is True
        assert rule.persistence_count == 2
        assert rule.default_severity == Severity.MEDIUM


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestStrategyConfidenceEvaluation:
    """Test strategy confidence evaluation logic."""

    @pytest.mark.asyncio
    async def test_no_intel_data_skips(self, evaluator):
        """Returns skip when no strategy intel data."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=None,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is False
        assert conditions[0].skip_reason == "strategy_intel_unavailable"

    @pytest.mark.asyncio
    async def test_no_active_versions_skips(self, evaluator):
        """Returns skip when no active versions."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=[],
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is False
        assert conditions[0].skip_reason == "no_active_versions"

    @pytest.mark.asyncio
    async def test_healthy_score_no_trigger(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Healthy confidence scores don't trigger alert."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [
            make_snapshot(0.75),  # Above warn threshold
            make_snapshot(0.80),
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is False

    @pytest.mark.asyncio
    async def test_single_low_score_no_trigger(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Single low score doesn't trigger (persistence gate)."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [
            make_snapshot(0.30),  # Below warn threshold
            make_snapshot(0.75),  # Healthy - breaks streak
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is False

    @pytest.mark.asyncio
    async def test_consecutive_warn_triggers(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Two consecutive low scores trigger warn alert."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [
            make_snapshot(0.30),  # Below warn (0.35)
            make_snapshot(0.32),  # Below warn
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is True
        assert conditions[0].severity == Severity.MEDIUM
        assert "warn" in conditions[0].dedupe_key
        assert conditions[0].payload["confidence_score"] == 0.30
        assert conditions[0].payload["consecutive_low_count"] == 2

    @pytest.mark.asyncio
    async def test_consecutive_critical_triggers(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Two consecutive critical scores trigger high severity alert."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [
            make_snapshot(0.15),  # Below critical (0.20)
            make_snapshot(0.18),  # Below critical
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is True
        assert conditions[0].severity == Severity.HIGH
        assert "critical" in conditions[0].dedupe_key
        assert conditions[0].payload["consecutive_low_count"] == 2

    @pytest.mark.asyncio
    async def test_mixed_critical_warn_triggers_warn(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Critical followed by warn still triggers warn (not critical)."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [
            make_snapshot(0.25),  # Between warn and critical
            make_snapshot(0.15),  # Below critical
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert len(conditions) == 1
        assert conditions[0].triggered is True
        # Should be warn because latest is not critical
        assert conditions[0].severity == Severity.MEDIUM
        assert "warn" in conditions[0].dedupe_key

    @pytest.mark.asyncio
    async def test_multiple_versions_evaluated(self, evaluator):
        """Multiple active versions are evaluated independently."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        version1_id = uuid4()
        version2_id = uuid4()
        strategy_id = uuid4()

        intel_data = [
            make_intel_data(
                version1_id,
                strategy_id,
                [make_snapshot(0.30), make_snapshot(0.28)],
                "Strategy A",
            ),
            make_intel_data(
                version2_id,
                strategy_id,
                [make_snapshot(0.75), make_snapshot(0.80)],
                "Strategy B",
            ),
        ]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        # Only version1 should trigger
        triggered = [c for c in conditions if c.triggered]
        assert len(triggered) == 1
        assert str(version1_id) in triggered[0].dedupe_key

    @pytest.mark.asyncio
    async def test_payload_contains_weak_components(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Payload includes weakest confidence components."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [
            {
                "as_of_ts": datetime.now(timezone.utc),
                "computed_at": datetime.now(timezone.utc),
                "regime": "range|volatility-high",
                "confidence_score": 0.25,
                "confidence_components": {
                    "performance": 0.8,
                    "drawdown": 0.2,  # Weakest
                    "stability": 0.3,  # Second weakest
                    "data_freshness": 0.4,  # Third weakest
                    "regime_fit": 0.7,
                },
            },
            make_snapshot(0.28),
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert conditions[0].triggered is True
        weak = conditions[0].payload["weak_components"]
        assert len(weak) == 3
        # Weakest first
        assert weak[0]["name"] == "drawdown"
        assert weak[0]["score"] == 0.2

    @pytest.mark.asyncio
    async def test_dedupe_key_format(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Dedupe key has correct format."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        snapshots = [make_snapshot(0.30), make_snapshot(0.28)]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        now = datetime(2026, 1, 19, 14, 30, 0, tzinfo=timezone.utc)
        ctx = EvalContext(
            workspace_id=uuid4(),
            now=now,
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        expected_key = f"strategy_confidence_low:{sample_version_id}:warn:2026-01-19"
        assert conditions[0].dedupe_key == expected_key

    @pytest.mark.asyncio
    async def test_weak_components_golden_contract(
        self, evaluator, sample_version_id, sample_strategy_id
    ):
        """Golden test: weak_components format is sorted, capped to 3, has name+score."""
        rule = get_rule(OpsRuleType.STRATEGY_CONFIDENCE_LOW)

        # All 5 components with distinct scores for clear ordering
        snapshots = [
            {
                "as_of_ts": datetime.now(timezone.utc),
                "computed_at": datetime.now(timezone.utc),
                "regime": "trend-down|volatility-high",
                "confidence_score": 0.18,  # Critical level
                "confidence_components": {
                    "performance": 0.9,  # Strongest (excluded)
                    "regime_fit": 0.7,  # Second strongest (excluded)
                    "data_freshness": 0.5,  # Third (included as #3)
                    "stability": 0.3,  # Fourth (included as #2)
                    "drawdown": 0.1,  # Weakest (included as #1)
                },
            },
            {
                "as_of_ts": datetime.now(timezone.utc),
                "computed_at": datetime.now(timezone.utc),
                "regime": "trend-down|volatility-high",
                "confidence_score": 0.15,
                "confidence_components": {
                    "performance": 0.9,
                    "regime_fit": 0.7,
                    "data_freshness": 0.5,
                    "stability": 0.3,
                    "drawdown": 0.1,
                },
            },
        ]
        intel_data = [make_intel_data(sample_version_id, sample_strategy_id, snapshots)]

        ctx = EvalContext(
            workspace_id=uuid4(),
            now=datetime.now(timezone.utc),
            strategy_intel=intel_data,
        )

        conditions = await evaluator._eval_strategy_confidence_low(rule, ctx)

        assert conditions[0].triggered is True
        weak = conditions[0].payload["weak_components"]

        # Contract: exactly 3 components
        assert len(weak) == 3, "weak_components must be capped to 3"

        # Contract: sorted by score ascending (weakest first)
        assert (
            weak[0]["score"] <= weak[1]["score"] <= weak[2]["score"]
        ), "weak_components must be sorted by score ascending"

        # Contract: each entry has name and score keys
        for entry in weak:
            assert "name" in entry, "Each weak_component must have 'name'"
            assert "score" in entry, "Each weak_component must have 'score'"
            assert isinstance(entry["name"], str)
            assert isinstance(entry["score"], (int, float))

        # Contract: verify exact order for this test case
        assert weak[0]["name"] == "drawdown"
        assert weak[0]["score"] == 0.1
        assert weak[1]["name"] == "stability"
        assert weak[1]["score"] == 0.3
        assert weak[2]["name"] == "data_freshness"
        assert weak[2]["score"] == 0.5


# =============================================================================
# Resolution Tests
# =============================================================================


class TestStrategyConfidenceResolution:
    """Test resolution of strategy confidence alerts."""

    @pytest.mark.asyncio
    async def test_resolves_when_score_recovers(self, evaluator, mock_repo):
        """Alert resolves when confidence score recovers."""
        workspace_id = uuid4()
        version_id = uuid4()
        now = datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc)

        # Simulate active alert from previous evaluation
        dedupe_key = f"strategy_confidence_low:{version_id}:warn:2026-01-19"
        mock_repo.get_active_dedupe_keys.return_value = {dedupe_key}

        # Score has recovered - key not in triggered set
        triggered_keys = set()

        await evaluator._resolution_pass(workspace_id, triggered_keys, now)

        mock_repo.resolve_by_dedupe_key.assert_called_with(workspace_id, dedupe_key)

    @pytest.mark.asyncio
    async def test_no_resolve_when_still_triggered(self, evaluator, mock_repo):
        """Alert not resolved when condition still active."""
        workspace_id = uuid4()
        version_id = uuid4()
        now = datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc)

        dedupe_key = f"strategy_confidence_low:{version_id}:warn:2026-01-19"
        mock_repo.get_active_dedupe_keys.return_value = {dedupe_key}

        # Key still triggered this pass
        triggered_keys = {dedupe_key}

        await evaluator._resolution_pass(workspace_id, triggered_keys, now)

        # Should not call resolve for this key
        resolve_calls = [
            call
            for call in mock_repo.resolve_by_dedupe_key.call_args_list
            if call.args[1] == dedupe_key
        ]
        assert len(resolve_calls) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestStrategyConfidenceIntegration:
    """Integration tests for full evaluation flow."""

    @pytest.mark.asyncio
    async def test_full_evaluation_creates_alert(self, mock_repo, mock_pool):
        """Full evaluation flow creates alert for low confidence."""
        workspace_id = uuid4()
        version_id = uuid4()
        strategy_id = uuid4()

        evaluator = OpsAlertEvaluator(mock_repo, mock_pool)

        # Mock the context building
        intel_data = [
            make_intel_data(
                version_id,
                strategy_id,
                [make_snapshot(0.30), make_snapshot(0.28)],
            )
        ]

        with patch.object(
            evaluator, "_build_context", return_value=AsyncMock()
        ) as mock_build:
            ctx = EvalContext(
                workspace_id=workspace_id,
                now=datetime.now(timezone.utc),
                strategy_intel=intel_data,
                # Other data sources can be None
                health_snapshot=None,
                coverage_stats=None,
                match_run_stats=None,
            )
            mock_build.return_value = ctx

            result = await evaluator.evaluate(workspace_id)

        # Check that strategy confidence rule was evaluated
        assert "strategy_confidence_low" in result.by_rule_type
        rule_result = result.by_rule_type["strategy_confidence_low"]
        assert rule_result["triggered"] is True

    @pytest.mark.asyncio
    async def test_escalation_warn_to_critical(self, mock_repo, mock_pool):
        """Alert escalates from warn to critical when score drops further."""
        workspace_id = uuid4()
        version_id = uuid4()
        strategy_id = uuid4()

        # First upsert returns existing warn
        mock_repo.upsert.return_value = UpsertResult(
            id=uuid4(),
            is_new=False,
            previous_severity="medium",
            current_severity="high",
            escalated=True,
        )

        evaluator = OpsAlertEvaluator(mock_repo, mock_pool)

        # Scores dropped to critical level
        intel_data = [
            make_intel_data(
                version_id,
                strategy_id,
                [make_snapshot(0.15), make_snapshot(0.12)],
            )
        ]

        with patch.object(evaluator, "_build_context") as mock_build:
            ctx = EvalContext(
                workspace_id=workspace_id,
                now=datetime.now(timezone.utc),
                strategy_intel=intel_data,
                health_snapshot=None,
                coverage_stats=None,
                match_run_stats=None,
            )
            mock_build.return_value = ctx

            result = await evaluator.evaluate(workspace_id)

        # Should have escalated
        rule_result = result.by_rule_type["strategy_confidence_low"]
        assert rule_result.get("escalated") is True
        assert rule_result["severity"] == "high"
