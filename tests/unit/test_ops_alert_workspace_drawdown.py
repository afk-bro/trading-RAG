"""Unit tests for workspace drawdown ops alert rule."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.ops_alerts.evaluator import OpsAlertEvaluator
from app.services.ops_alerts.models import (
    EvalContext,
    OpsRuleType,
    Severity,
    get_rule,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repo():
    """Mock OpsAlertsRepository."""
    repo = AsyncMock()
    repo.upsert = AsyncMock(
        return_value=MagicMock(id=uuid4(), is_new=True, escalated=False)
    )
    repo.resolve_by_dedupe_key = AsyncMock(return_value=None)
    repo.get_active_dedupe_keys = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Setup acquire as async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm

    return pool, conn


@pytest.fixture
def evaluator(mock_repo, mock_pool):
    """Create evaluator with mocked dependencies."""
    pool, conn = mock_pool
    return OpsAlertEvaluator(mock_repo, pool)


@pytest.fixture
def sample_workspace_id():
    """Sample workspace ID."""
    return uuid4()


@pytest.fixture
def base_context(sample_workspace_id):
    """Base evaluation context."""
    return EvalContext(
        workspace_id=sample_workspace_id,
        now=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        job_run_id=uuid4(),
    )


# =============================================================================
# _eval_workspace_drawdown_high Tests
# =============================================================================


class TestWorkspaceDrawdownRule:
    """Tests for workspace_drawdown_high rule evaluation."""

    def test_skips_when_no_equity_data(self, evaluator, base_context):
        """Test that rule skips when equity data is unavailable."""
        base_context.equity_data = None
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is False
        assert result.skip_reason == "equity_data_unavailable"

    def test_skips_when_insufficient_snapshots(self, evaluator, base_context):
        """Test that rule skips when fewer than 2 snapshots exist."""
        base_context.equity_data = {
            "drawdown_pct": 0.15,
            "snapshot_count": 1,  # Less than required
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is False
        assert "insufficient_snapshots" in result.skip_reason

    def test_no_trigger_when_below_warn_threshold(self, evaluator, base_context):
        """Test that rule doesn't trigger when DD < 12%."""
        base_context.equity_data = {
            "drawdown_pct": 0.08,  # 8% - below warn threshold
            "peak_equity": 100000.0,
            "current_equity": 92000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is False
        assert result.skip_reason is None

    def test_triggers_warn_at_12_percent(self, evaluator, base_context):
        """Test that rule triggers warn severity at 12% DD."""
        base_context.equity_data = {
            "drawdown_pct": 0.12,  # Exactly at warn threshold
            "peak_equity": 100000.0,
            "current_equity": 88000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is True
        assert result.severity == Severity.MEDIUM
        assert ":warn:" in result.dedupe_key

    def test_triggers_warn_at_15_percent(self, evaluator, base_context):
        """Test that rule triggers warn severity at 15% DD."""
        base_context.equity_data = {
            "drawdown_pct": 0.15,  # Between warn and critical
            "peak_equity": 100000.0,
            "current_equity": 85000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is True
        assert result.severity == Severity.MEDIUM
        assert ":warn:" in result.dedupe_key

    def test_triggers_critical_at_20_percent(self, evaluator, base_context):
        """Test that rule triggers critical (HIGH) severity at 20% DD."""
        base_context.equity_data = {
            "drawdown_pct": 0.20,  # At critical threshold
            "peak_equity": 100000.0,
            "current_equity": 80000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is True
        assert result.severity == Severity.HIGH
        assert ":critical:" in result.dedupe_key

    def test_triggers_critical_at_25_percent(self, evaluator, base_context):
        """Test that rule triggers critical severity at 25% DD."""
        base_context.equity_data = {
            "drawdown_pct": 0.25,  # Above critical threshold
            "peak_equity": 100000.0,
            "current_equity": 75000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is True
        assert result.severity == Severity.HIGH
        assert ":critical:" in result.dedupe_key

    def test_payload_contains_required_fields(self, evaluator, base_context):
        """Test that triggered alert payload contains all required fields."""
        base_context.equity_data = {
            "drawdown_pct": 0.15,
            "peak_equity": 100000.0,
            "current_equity": 85000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is True
        payload = result.payload

        # Required payload fields
        assert "workspace_id" in payload
        assert "drawdown_pct" in payload
        assert "peak_equity" in payload
        assert "current_equity" in payload
        assert "peak_ts" in payload
        assert "current_ts" in payload
        assert "window_days" in payload
        assert "snapshot_count" in payload
        assert "thresholds" in payload

        # Threshold values
        thresholds = payload["thresholds"]
        assert "warn" in thresholds
        assert "critical" in thresholds
        assert "clear_warn" in thresholds
        assert "clear_critical" in thresholds

        # Values match
        assert payload["drawdown_pct"] == 0.15
        assert payload["peak_equity"] == 100000.0
        assert payload["current_equity"] == 85000.0

    def test_dedupe_key_format(self, evaluator, base_context):
        """Test dedupe key has correct format."""
        base_context.equity_data = {
            "drawdown_pct": 0.15,
            "peak_equity": 100000.0,
            "current_equity": 85000.0,
            "peak_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
            "current_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "window_days": 30,
            "snapshot_count": 10,
        }
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)

        result = evaluator._eval_workspace_drawdown_high(rule, base_context)

        assert result.triggered is True
        # Format: workspace_drawdown_high:{severity_bucket}:{date}
        assert result.dedupe_key.startswith("workspace_drawdown_high:")
        assert ":warn:" in result.dedupe_key
        assert "2024-01-15" in result.dedupe_key


class TestWorkspaceDrawdownResolution:
    """Tests for workspace drawdown alert resolution."""

    @pytest.mark.asyncio
    async def test_resolves_when_dd_recovers(
        self, evaluator, mock_repo, sample_workspace_id
    ):
        """Test that alert resolves when drawdown recovers below threshold."""
        # Active drawdown alert exists
        existing_alert = MagicMock(id=uuid4(), rule_type="workspace_drawdown_high")
        mock_repo.get_active_dedupe_keys.return_value = [
            "workspace_drawdown_high:warn:2024-01-15"
        ]
        mock_repo.resolve_by_dedupe_key.return_value = existing_alert

        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Empty triggered_keys means condition cleared
        resolved = await evaluator._resolution_pass(
            sample_workspace_id,
            triggered_keys=set(),  # Nothing triggered
            now=now,
        )

        # Should have resolved the drawdown alert
        assert len(resolved) > 0
        mock_repo.resolve_by_dedupe_key.assert_called()

    @pytest.mark.asyncio
    async def test_no_resolve_when_still_triggered(
        self, evaluator, mock_repo, sample_workspace_id
    ):
        """Test that alert stays active when drawdown is still high."""
        mock_repo.get_active_dedupe_keys.return_value = [
            "workspace_drawdown_high:warn:2024-01-15"
        ]

        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Triggered keys include the drawdown alert
        await evaluator._resolution_pass(
            sample_workspace_id,
            triggered_keys={"workspace_drawdown_high:warn:2024-01-15"},
            now=now,
        )

        # Should not have resolved (filtered out by triggered_keys)
        # The drawdown-specific resolve_by_dedupe_key should not be called for this key
        drawdown_resolve_calls = [
            call
            for call in mock_repo.resolve_by_dedupe_key.call_args_list
            if "drawdown" in str(call)
        ]
        assert len(drawdown_resolve_calls) == 0


class TestWorkspaceDrawdownRuleDefinition:
    """Tests for workspace drawdown rule definition."""

    def test_rule_exists(self):
        """Test that WORKSPACE_DRAWDOWN_HIGH rule is defined."""
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)
        assert rule is not None
        assert rule.rule_type == OpsRuleType.WORKSPACE_DRAWDOWN_HIGH

    def test_rule_requires_equity_data(self):
        """Test that rule requires equity data."""
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)
        assert rule.requires_equity_data is True

    def test_rule_has_persistence_count(self):
        """Test that rule has persistence gating."""
        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)
        assert rule.persistence_count == 2

    def test_rule_is_daily(self):
        """Test that rule uses daily deduplication."""
        from app.services.ops_alerts.models import DedupePeriod

        rule = get_rule(OpsRuleType.WORKSPACE_DRAWDOWN_HIGH)
        assert rule.dedupe_period == DedupePeriod.DAILY


class TestDrawdownThresholds:
    """Tests for drawdown threshold constants."""

    def test_warn_threshold_is_12_percent(self, evaluator):
        """Test that warn threshold is 12%."""
        assert evaluator.DRAWDOWN_WARN == 0.12

    def test_critical_threshold_is_20_percent(self, evaluator):
        """Test that critical threshold is 20%."""
        assert evaluator.DRAWDOWN_CRITICAL == 0.20

    def test_clear_warn_is_10_percent(self, evaluator):
        """Test that clear warn threshold is 10%."""
        assert evaluator.DRAWDOWN_CLEAR_WARN == 0.10

    def test_clear_critical_is_16_percent(self, evaluator):
        """Test that clear critical threshold is 16%."""
        assert evaluator.DRAWDOWN_CLEAR_CRITICAL == 0.16

    def test_window_is_30_days(self, evaluator):
        """Test that default window is 30 days."""
        assert evaluator.DRAWDOWN_WINDOW_DAYS == 30

    def test_hysteresis_gap_for_warn(self, evaluator):
        """Test that there's a gap between warn and clear_warn."""
        # Warn at 12%, clear at 10% = 2% hysteresis
        assert evaluator.DRAWDOWN_WARN > evaluator.DRAWDOWN_CLEAR_WARN
        assert evaluator.DRAWDOWN_WARN - evaluator.DRAWDOWN_CLEAR_WARN == pytest.approx(
            0.02
        )

    def test_hysteresis_gap_for_critical(self, evaluator):
        """Test that there's a gap between critical and clear_critical."""
        # Critical at 20%, clear at 16% = 4% hysteresis
        assert evaluator.DRAWDOWN_CRITICAL > evaluator.DRAWDOWN_CLEAR_CRITICAL
        assert (
            evaluator.DRAWDOWN_CRITICAL - evaluator.DRAWDOWN_CLEAR_CRITICAL
            == pytest.approx(0.04)
        )
