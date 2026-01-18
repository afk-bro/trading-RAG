"""Tests for TuneJob artifact generation functions."""

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.jobs.handlers.tune import (
    _generate_equity_csv,
    _generate_trials_csv,
    _generate_tune_json,
    _write_file_atomic,
)
from app.services.backtest.tuner import TuneResult


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock get_settings for all tests in this module."""
    mock = MagicMock()
    mock.git_sha = "abc123test"
    with patch("app.jobs.handlers.tune.get_settings", return_value=mock):
        yield mock


class TestWriteFileAtomic:
    """Tests for _write_file_atomic."""

    def test_creates_parent_directories(self, tmp_path: Path):
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "a" / "b" / "c" / "test.txt"
        content = b"hello world"

        size = _write_file_atomic(nested_path, content)

        assert nested_path.exists()
        assert nested_path.read_bytes() == content
        assert size == len(content)

    def test_returns_correct_size(self, tmp_path: Path):
        """Should return the correct file size."""
        path = tmp_path / "test.txt"
        content = b"0" * 1024

        size = _write_file_atomic(path, content)

        assert size == 1024

    def test_atomic_write_no_temp_file_remains(self, tmp_path: Path):
        """Should not leave temp files after successful write."""
        path = tmp_path / "test.txt"
        content = b"test content"

        _write_file_atomic(path, content)

        # Check no .tmp files remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestGenerateTuneJson:
    """Tests for _generate_tune_json."""

    def test_generates_valid_json(self):
        """Should generate valid JSON."""
        tune = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "strategy_name": "test_strategy",
            "param_space": {"lookback": [10, 20, 30]},
            "search_type": "grid",
            "n_trials": 10,
            "seed": 42,
            "objective_metric": "sharpe",
            "objective_type": "sharpe_dd_penalty",
            "objective_params": {"dd_lambda": 0.5},
            "oos_ratio": 0.2,
            "gates": {"max_drawdown_pct": 20, "min_trades": 5},
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }
        result = TuneResult(
            tune_id=tune["id"],
            status="completed",
            n_trials=10,
            trials_completed=8,
            best_run_id=uuid4(),
            best_params={"lookback": 20},
            best_score=1.5,
            leaderboard=[],
        )

        content = _generate_tune_json(tune, result, None)
        parsed = json.loads(content)

        assert parsed["identifiers"]["tune_id"] == str(tune["id"])
        assert parsed["identifiers"]["strategy_name"] == "test_strategy"
        assert parsed["param_space"]["lookback"] == [10, 20, 30]
        assert parsed["results"]["status"] == "completed"
        assert parsed["results"]["trials_completed"] == 8
        assert parsed["results"]["best_score"] == 1.5

    def test_includes_data_revision(self):
        """Should include data revision when provided."""
        tune = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "created_at": datetime.now(timezone.utc),
        }
        result = TuneResult(
            tune_id=tune["id"],
            status="completed",
            n_trials=5,
            trials_completed=5,
            best_run_id=None,
            best_params=None,
            best_score=None,
            leaderboard=[],
        )
        data_revision = {
            "exchange_id": "kucoin",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "row_count": 1000,
        }

        content = _generate_tune_json(tune, result, data_revision)
        parsed = json.loads(content)

        assert parsed["data_lineage"] == data_revision

    def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields gracefully."""
        tune = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "strategy_entity_id": uuid4(),
        }
        result = TuneResult(
            tune_id=tune["id"],
            status="failed",
            n_trials=5,
            trials_completed=0,
            best_run_id=None,
            best_params=None,
            best_score=None,
            leaderboard=[],
        )

        content = _generate_tune_json(tune, result, None)
        parsed = json.loads(content)

        assert parsed["identifiers"]["strategy_name"] is None
        assert parsed["data_lineage"] is None
        assert parsed["results"]["best_run_id"] is None


class TestGenerateTrialsCsv:
    """Tests for _generate_trials_csv."""

    def test_generates_valid_csv_header(self):
        """Should generate CSV with correct header."""
        runs = []

        content = _generate_trials_csv(runs)
        lines = content.decode("utf-8").strip().split("\n")

        assert len(lines) == 1  # Just header
        header = lines[0].split(",")
        assert "trial_index" in header
        assert "params_json" in header
        assert "score" in header
        assert "score_is" in header
        assert "score_oos" in header
        assert "objective_score" in header

    def test_includes_all_trial_data(self):
        """Should include all trial data."""
        runs = [
            {
                "trial_index": 0,
                "run_id": uuid4(),
                "params": {"lookback": 10},
                "score": 1.2,
                "score_is": 1.4,
                "score_oos": 1.0,
                "objective_score": 0.9,
                "status": "completed",
                "skip_reason": None,
                "failed_reason": None,
                "metrics_is": {
                    "return_pct": 15.5, "sharpe": 1.4, "max_drawdown_pct": 8.2, "num_trades": 25
                },
                "metrics_oos": {
                    "return_pct": 10.0, "sharpe": 1.0, "max_drawdown_pct": 10.1, "num_trades": 12
                },
            },
            {
                "trial_index": 1,
                "run_id": uuid4(),
                "params": {"lookback": 20},
                "score": 0.8,
                "score_is": None,
                "score_oos": None,
                "objective_score": None,
                "status": "skipped",
                "skip_reason": "duplicate",
                "failed_reason": None,
                "metrics_is": None,
                "metrics_oos": None,
            },
        ]

        content = _generate_trials_csv(runs)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)

        assert len(rows) == 2

        # Check first row
        assert rows[0]["trial_index"] == "0"
        assert rows[0]["score"] == "1.2"
        assert rows[0]["score_is"] == "1.4"
        assert rows[0]["score_oos"] == "1.0"
        assert rows[0]["status"] == "completed"
        assert rows[0]["return_pct_is"] == "15.5"
        assert rows[0]["sharpe_oos"] == "1.0"

        # Check second row (skipped)
        assert rows[1]["trial_index"] == "1"
        assert rows[1]["status"] == "skipped"
        assert rows[1]["skip_reason"] == "duplicate"

    def test_serializes_params_as_json(self):
        """Should serialize params as JSON string."""
        runs = [
            {
                "trial_index": 0,
                "run_id": uuid4(),
                "params": {"lookback": 10, "threshold": 0.5},
                "score": 1.0,
                "status": "completed",
            },
        ]

        content = _generate_trials_csv(runs)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)

        params = json.loads(rows[0]["params_json"])
        assert params == {"lookback": 10, "threshold": 0.5}


class TestGenerateEquityCsv:
    """Tests for _generate_equity_csv."""

    def test_generates_valid_csv_header(self):
        """Should generate CSV with correct header."""
        equity_curve = []

        content = _generate_equity_csv(equity_curve)
        lines = content.decode("utf-8").strip().split("\n")

        assert len(lines) == 1  # Just header
        header = lines[0].split(",")
        assert header == ["ts", "equity", "drawdown_pct"]

    def test_includes_equity_data(self):
        """Should include equity data with timestamps."""
        equity_curve = [
            {"t": "2024-01-01T00:00:00Z", "equity": 10000},
            {"t": "2024-01-02T00:00:00Z", "equity": 10500},
            {"t": "2024-01-03T00:00:00Z", "equity": 10200},
        ]

        content = _generate_equity_csv(equity_curve)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["ts"] == "2024-01-01T00:00:00Z"
        assert float(rows[0]["equity"]) == 10000.0
        assert float(rows[0]["drawdown_pct"]) == 0.0

    def test_calculates_drawdown_correctly(self):
        """Should calculate drawdown from peak correctly."""
        equity_curve = [
            {"t": "2024-01-01T00:00:00Z", "equity": 10000},
            {"t": "2024-01-02T00:00:00Z", "equity": 10500},  # New peak
            {"t": "2024-01-03T00:00:00Z", "equity": 9975},  # 5% drawdown from 10500
            {"t": "2024-01-04T00:00:00Z", "equity": 10500},  # Recovery
            {"t": "2024-01-05T00:00:00Z", "equity": 11000},  # New peak
            {"t": "2024-01-06T00:00:00Z", "equity": 10450},  # 5% drawdown from 11000
        ]

        content = _generate_equity_csv(equity_curve)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)

        # First point - no drawdown
        assert float(rows[0]["drawdown_pct"]) == 0.0

        # New peak - no drawdown
        assert float(rows[1]["drawdown_pct"]) == 0.0

        # Drawdown from 10500 to 9975 = 5%
        assert float(rows[2]["drawdown_pct"]) == pytest.approx(5.0, rel=0.01)

        # Recovery to peak - 0%
        assert float(rows[3]["drawdown_pct"]) == 0.0

        # New peak - 0%
        assert float(rows[4]["drawdown_pct"]) == 0.0

        # Drawdown from 11000 to 10450 = 5%
        assert float(rows[5]["drawdown_pct"]) == pytest.approx(5.0, rel=0.01)

    def test_handles_empty_curve(self):
        """Should handle empty equity curve."""
        content = _generate_equity_csv([])
        lines = content.decode("utf-8").strip().split("\n")

        assert len(lines) == 1  # Just header
