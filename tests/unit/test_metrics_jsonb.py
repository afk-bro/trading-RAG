"""Tests for metrics JSONB persistence."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.services.backtest.tuner import serialize_metrics, METRICS_KEYS


class TestSerializeMetrics:
    """Tests for metrics serializer function."""

    def test_serializes_all_canonical_keys(self):
        """Serializer should include all canonical keys."""
        summary = {
            "return_pct": 12.5678,
            "sharpe": 1.8234,
            "max_drawdown_pct": -8.2345,
            "win_rate": 0.6234,
            "trades": 45,
            "profit_factor": 2.1234,
        }

        metrics = serialize_metrics(summary)

        assert set(metrics.keys()) == set(METRICS_KEYS)

    def test_rounds_percentages_to_2_decimals(self):
        """return_pct and max_drawdown_pct should be rounded to 2 decimals."""
        summary = {
            "return_pct": 12.5678,
            "max_drawdown_pct": -8.2345,
        }

        metrics = serialize_metrics(summary)

        assert metrics["return_pct"] == 12.57
        assert metrics["max_drawdown_pct"] == -8.23

    def test_rounds_ratios_to_4_decimals(self):
        """sharpe, win_rate, profit_factor should be rounded to 4 decimals."""
        summary = {
            "sharpe": 1.82345678,
            "win_rate": 0.62345678,
            "profit_factor": 2.12345678,
        }

        metrics = serialize_metrics(summary)

        assert metrics["sharpe"] == 1.8235
        assert metrics["win_rate"] == 0.6235
        assert metrics["profit_factor"] == 2.1235

    def test_trades_as_integer(self):
        """trades should be converted to int."""
        summary = {"trades": 45.0}

        metrics = serialize_metrics(summary)

        assert metrics["trades"] == 45
        assert isinstance(metrics["trades"], int)

    def test_converts_decimal_to_float(self):
        """Decimal values should be converted to float."""
        summary = {
            "return_pct": Decimal("12.5678"),
            "sharpe": Decimal("1.8234"),
        }

        metrics = serialize_metrics(summary)

        assert isinstance(metrics["return_pct"], float)
        assert isinstance(metrics["sharpe"], float)

    def test_converts_nan_to_none(self):
        """NaN values should become None."""
        summary = {
            "sharpe": float("nan"),
            "profit_factor": float("nan"),
        }

        metrics = serialize_metrics(summary)

        assert metrics["sharpe"] is None
        assert metrics["profit_factor"] is None

    def test_converts_inf_to_none(self):
        """Infinity values should become None."""
        summary = {
            "sharpe": float("inf"),
            "profit_factor": float("-inf"),
        }

        metrics = serialize_metrics(summary)

        assert metrics["sharpe"] is None
        assert metrics["profit_factor"] is None

    def test_missing_keys_are_none(self):
        """Missing keys should be None, not omitted."""
        summary = {"return_pct": 10.0}

        metrics = serialize_metrics(summary)

        assert metrics["sharpe"] is None
        assert metrics["max_drawdown_pct"] is None
        assert metrics["win_rate"] is None
        assert metrics["trades"] is None
        assert metrics["profit_factor"] is None

    def test_extra_keys_are_ignored(self):
        """Keys not in METRICS_KEYS should be ignored."""
        summary = {
            "return_pct": 10.0,
            "some_other_metric": 999,
            "internal_value": "ignored",
        }

        metrics = serialize_metrics(summary)

        assert "some_other_metric" not in metrics
        assert "internal_value" not in metrics


class TestMetricsRepositoryPersistence:
    """Tests for metrics persistence in repository."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool, conn

    @pytest.mark.asyncio
    async def test_update_tune_run_result_with_metrics(self, mock_pool):
        """update_tune_run_result should persist metrics as JSONB."""
        from app.repositories.backtests import TuneRepository

        pool, conn = mock_pool
        repo = TuneRepository(pool)
        tune_id = uuid4()

        conn.execute.return_value = None

        await repo.update_tune_run_result(
            tune_id=tune_id,
            trial_index=0,
            run_id=uuid4(),
            score=1.5,
            status="completed",
            metrics_is={"return_pct": 10.0, "sharpe": 1.2},
            metrics_oos={"return_pct": 8.0, "sharpe": 1.0},
        )

        # Verify metrics are in the query
        call_args = conn.execute.call_args[0][0]
        assert "metrics_is" in call_args
        assert "metrics_oos" in call_args

    @pytest.mark.asyncio
    async def test_non_split_persists_to_metrics_oos(self, mock_pool):
        """Non-split tunes should persist to metrics_oos (Option A)."""
        from app.repositories.backtests import TuneRepository

        pool, conn = mock_pool
        repo = TuneRepository(pool)
        tune_id = uuid4()

        conn.execute.return_value = None

        # Non-split: only metrics_oos, no metrics_is
        await repo.update_tune_run_result(
            tune_id=tune_id,
            trial_index=0,
            run_id=uuid4(),
            score=1.5,
            status="completed",
            metrics_oos={"return_pct": 10.0, "sharpe": 1.2},
        )

        # Verify the call was made
        assert conn.execute.called


class TestMetricsAPIResponse:
    """Tests for metrics in API response."""

    def test_tune_run_list_item_includes_metrics(self):
        """TuneRunListItem should accept metrics_is and metrics_oos."""
        from app.routers.backtests.schemas import TuneRunListItem

        item = TuneRunListItem(
            trial_index=0,
            run_id="abc123",
            params={"period": 20},
            score=1.5,
            score_is=1.2,
            score_oos=1.5,
            metrics_is={"return_pct": 10.0, "sharpe": 1.2},
            metrics_oos={"return_pct": 8.0, "sharpe": 1.0},
            status="completed",
        )

        assert item.metrics_is == {"return_pct": 10.0, "sharpe": 1.2}
        assert item.metrics_oos == {"return_pct": 8.0, "sharpe": 1.0}

    def test_tune_run_list_item_metrics_optional(self):
        """metrics_is and metrics_oos should be optional."""
        from app.routers.backtests.schemas import TuneRunListItem

        item = TuneRunListItem(
            trial_index=0,
            run_id="abc123",
            params={"period": 20},
            score=1.5,
            status="completed",
        )

        assert item.metrics_is is None
        assert item.metrics_oos is None


class TestMetricsDataContract:
    """Tests for metrics data contract compliance."""

    def test_metrics_are_numbers_only(self):
        """All metrics values should be numeric or None."""
        summary = {
            "return_pct": 12.5,
            "sharpe": 1.8,
            "max_drawdown_pct": -8.2,
            "win_rate": 0.62,
            "trades": 45,
            "profit_factor": 2.1,
        }

        metrics = serialize_metrics(summary)

        for key, value in metrics.items():
            assert value is None or isinstance(
                value, (int, float)
            ), f"{key} should be numeric or None, got {type(value)}"

    def test_no_string_values(self):
        """Metrics should never contain string values."""
        summary = {
            "return_pct": "12.5%",  # Bad: string with %
            "sharpe": "1.8",  # Bad: string number
        }

        # These would fail to convert, resulting in None
        metrics = serialize_metrics(summary)

        # String values should not persist as strings
        for key, value in metrics.items():
            assert not isinstance(value, str), f"{key} should not be a string"
