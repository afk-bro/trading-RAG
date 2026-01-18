"""Tests for TuneJob handler."""

import base64
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4, UUID

import pytest

from app.jobs.models import Job
from app.jobs.types import JobType, JobStatus


class TestHandleTune:
    """Tests for handle_tune handler."""

    @pytest.fixture
    def sample_tune_id(self) -> UUID:
        """Create a sample tune ID."""
        return uuid4()

    @pytest.fixture
    def sample_workspace_id(self) -> UUID:
        """Create a sample workspace ID."""
        return uuid4()

    @pytest.fixture
    def sample_strategy_entity_id(self) -> UUID:
        """Create a sample strategy entity ID."""
        return uuid4()

    @pytest.fixture
    def sample_ohlcv_csv(self) -> bytes:
        """Create sample OHLCV CSV content."""
        csv_lines = [
            "date,open,high,low,close,volume",
            "2024-01-01T00:00:00Z,42000.0,42500.0,41800.0,42200.0,100.0",
            "2024-01-02T00:00:00Z,42200.0,42700.0,42000.0,42400.0,110.0",
            "2024-01-03T00:00:00Z,42400.0,42900.0,42200.0,42600.0,120.0",
        ]
        return "\n".join(csv_lines).encode("utf-8")

    @pytest.fixture
    def sample_job_with_file_content(
        self,
        sample_tune_id,
        sample_workspace_id,
        sample_strategy_entity_id,
        sample_ohlcv_csv,
    ):
        """Create a sample TUNE job with inline file content."""
        return Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.RUNNING,
            workspace_id=sample_workspace_id,
            payload={
                "workspace_id": str(sample_workspace_id),
                "tune_id": str(sample_tune_id),
                "strategy_entity_id": str(sample_strategy_entity_id),
                "ohlcv_file_content": base64.b64encode(sample_ohlcv_csv).decode(
                    "ascii"
                ),
                "filename": "BTC-USDT_1h.csv",
                "param_space": {"period": [10, 14, 20]},
                "search_type": "grid",
                "objective_type": "sharpe",
                "gates": {"max_drawdown_pct": 20, "min_trades": 5},
                "oos_ratio": 0.2,
                "seed": 42,
                "n_trials": 3,
                "objective_metric": "sharpe",
                "min_trades": 5,
                "initial_cash": 10000.0,
                "commission_bps": 10,
                "slippage_bps": 5,
            },
        )

    @pytest.fixture
    def sample_job_with_data_source(
        self, sample_tune_id, sample_workspace_id, sample_strategy_entity_id
    ):
        """Create a sample TUNE job with data_source (stored OHLCV)."""
        return Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.RUNNING,
            workspace_id=sample_workspace_id,
            payload={
                "workspace_id": str(sample_workspace_id),
                "tune_id": str(sample_tune_id),
                "strategy_entity_id": str(sample_strategy_entity_id),
                "data_source": {
                    "exchange_id": "kucoin",
                    "symbol": "BTC-USDT",
                    "timeframe": "1h",
                    "start_ts": "2024-01-01T00:00:00Z",
                    "end_ts": "2024-07-01T00:00:00Z",
                },
                "param_space": {"period": [10, 14, 20]},
                "search_type": "grid",
                "objective_type": "sharpe",
                "gates": {"max_drawdown_pct": 20, "min_trades": 5},
                "oos_ratio": 0.2,
                "seed": 42,
                "n_trials": 3,
                "objective_metric": "sharpe",
                "min_trades": 5,
                "initial_cash": 10000.0,
                "commission_bps": 10,
                "slippage_bps": 5,
            },
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        mock_pool = MagicMock()
        mock_events_repo = AsyncMock()
        mock_events_repo.info = AsyncMock()
        mock_events_repo.error = AsyncMock()
        return {
            "pool": mock_pool,
            "events_repo": mock_events_repo,
            "worker_id": "test-worker",
        }

    @pytest.fixture
    def mock_tune_result(self, sample_tune_id):
        """Create a mock TuneResult."""
        from app.services.backtest.tuner import TuneResult

        best_run_id = uuid4()
        return TuneResult(
            tune_id=sample_tune_id,
            status="completed",
            n_trials=3,
            trials_completed=3,
            best_run_id=best_run_id,
            best_params={"period": 14},
            best_score=1.5,
            leaderboard=[
                {
                    "rank": 1,
                    "run_id": str(best_run_id),
                    "params": {"period": 14},
                    "score": 1.5,
                    "objective_score": 1.45,
                    "summary": {"sharpe": 1.5, "return_pct": 15.0},
                }
            ],
            warnings=[],
        )

    @pytest.mark.asyncio
    async def test_handler_registered_with_registry(self):
        """Test that handle_tune is registered with the registry."""
        # Import to trigger registration
        from app.jobs.handlers import tune  # noqa: F401
        from app.jobs.registry import default_registry
        from app.jobs.types import JobType

        handler = default_registry.get_handler(JobType.TUNE)
        assert handler is not None
        assert handler.__name__ == "handle_tune"

    @pytest.mark.asyncio
    async def test_handler_calls_param_tuner_with_file_content(
        self,
        sample_job_with_file_content,
        mock_context,
        mock_tune_result,
        sample_ohlcv_csv,
    ):
        """Test handler calls ParamTuner.run with inline file content."""
        from app.jobs.handlers.tune import handle_tune

        with patch("app.jobs.handlers.tune.ParamTuner") as MockTuner:
            mock_tuner_instance = AsyncMock()
            mock_tuner_instance.run = AsyncMock(return_value=mock_tune_result)
            MockTuner.return_value = mock_tuner_instance

            with patch("app.jobs.handlers.tune.TuneRepository") as MockTuneRepo:
                mock_tune_repo = AsyncMock()
                MockTuneRepo.return_value = mock_tune_repo

                with patch(
                    "app.jobs.handlers.tune.KnowledgeBaseRepository"
                ) as MockKBRepo:
                    mock_kb_repo = AsyncMock()
                    MockKBRepo.return_value = mock_kb_repo

                    with patch(
                        "app.jobs.handlers.tune.BacktestRepository"
                    ) as MockBTRepo:
                        mock_bt_repo = AsyncMock()
                        MockBTRepo.return_value = mock_bt_repo

                        with patch(
                            "app.jobs.handlers.tune.generate_tune_artifacts"
                        ) as mock_gen_artifacts:
                            mock_gen_artifacts.return_value = [
                                "tune.json",
                                "trials.csv",
                                "equity_best.csv",
                            ]

                            result = await handle_tune(
                                sample_job_with_file_content, mock_context
                            )

            # Verify result
            assert result["status"] == "completed"
            assert result["tune_id"] == str(mock_tune_result.tune_id)
            assert result["n_trials"] == 3
            assert result["trials_completed"] == 3
            assert result["best_score"] == 1.5
            assert "artifacts" in result

            # Verify ParamTuner.run was called
            mock_tuner_instance.run.assert_called_once()
            call_kwargs = mock_tuner_instance.run.call_args[1]
            assert call_kwargs["file_content"] == sample_ohlcv_csv
            assert call_kwargs["search_type"] == "grid"
            assert call_kwargs["objective_type"] == "sharpe"

    @pytest.mark.asyncio
    async def test_handler_calls_ensure_ohlcv_range_with_data_source(
        self,
        sample_job_with_data_source,
        mock_context,
        mock_tune_result,
        sample_ohlcv_csv,
    ):
        """Test handler calls ensure_ohlcv_range when data_source provided."""
        from app.jobs.handlers.tune import handle_tune
        from app.services.market_data.ensure_range import EnsureRangeResult
        from app.repositories.ohlcv import Candle

        # Mock ensure_ohlcv_range
        mock_ensure_result = EnsureRangeResult(
            total_candles=100,
            fetched_candles=0,
            gaps_filled=[],
            was_cached=True,
        )

        # Mock candles for loading from DB (spread across multiple days)
        mock_candles = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1 + (i // 24), i % 24, 0, 0, tzinfo=timezone.utc),
                open=42000.0 + i * 10,
                high=42500.0 + i * 10,
                low=41800.0 + i * 10,
                close=42200.0 + i * 10,
                volume=100.0 + i,
            )
            for i in range(100)
        ]

        with patch("app.jobs.handlers.tune.ensure_ohlcv_range") as mock_ensure:
            mock_ensure.return_value = mock_ensure_result

            with patch("app.jobs.handlers.tune.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.get_range = AsyncMock(return_value=mock_candles)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch("app.jobs.handlers.tune.ParamTuner") as MockTuner:
                    mock_tuner_instance = AsyncMock()
                    mock_tuner_instance.run = AsyncMock(return_value=mock_tune_result)
                    MockTuner.return_value = mock_tuner_instance

                    with patch("app.jobs.handlers.tune.TuneRepository") as MockTuneRepo:
                        mock_tune_repo = AsyncMock()
                        MockTuneRepo.return_value = mock_tune_repo

                        with patch(
                            "app.jobs.handlers.tune.KnowledgeBaseRepository"
                        ) as MockKBRepo:
                            mock_kb_repo = AsyncMock()
                            MockKBRepo.return_value = mock_kb_repo

                            with patch(
                                "app.jobs.handlers.tune.BacktestRepository"
                            ) as MockBTRepo:
                                mock_bt_repo = AsyncMock()
                                MockBTRepo.return_value = mock_bt_repo

                                with patch(
                                    "app.jobs.handlers.tune.generate_tune_artifacts"
                                ) as mock_gen:
                                    mock_gen.return_value = ["tune.json"]

                                    result = await handle_tune(
                                        sample_job_with_data_source, mock_context
                                    )

        # Verify ensure_ohlcv_range was called
        mock_ensure.assert_called_once()
        call_kwargs = mock_ensure.call_args[1]
        assert call_kwargs["exchange_id"] == "kucoin"
        assert call_kwargs["symbol"] == "BTC-USDT"
        assert call_kwargs["timeframe"] == "1h"

        # Verify OHLCVRepository.get_range was called to load data
        mock_ohlcv_repo.get_range.assert_called_once()

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handler_returns_correct_result_schema(
        self, sample_job_with_file_content, mock_context, mock_tune_result
    ):
        """Test handler returns correct result schema."""
        from app.jobs.handlers.tune import handle_tune

        with patch("app.jobs.handlers.tune.ParamTuner") as MockTuner:
            mock_tuner_instance = AsyncMock()
            mock_tuner_instance.run = AsyncMock(return_value=mock_tune_result)
            MockTuner.return_value = mock_tuner_instance

            with patch("app.jobs.handlers.tune.TuneRepository"):
                with patch("app.jobs.handlers.tune.KnowledgeBaseRepository"):
                    with patch("app.jobs.handlers.tune.BacktestRepository"):
                        with patch(
                            "app.jobs.handlers.tune.generate_tune_artifacts"
                        ) as mock_gen:
                            mock_gen.return_value = [
                                "tune.json",
                                "trials.csv",
                                "equity_best.csv",
                            ]

                            result = await handle_tune(
                                sample_job_with_file_content, mock_context
                            )

        # Verify result schema
        assert "status" in result
        assert "tune_id" in result
        assert "n_trials" in result
        assert "trials_completed" in result
        assert "best_run_id" in result
        assert "best_score" in result
        assert "artifacts" in result

        assert result["status"] == "completed"
        assert isinstance(result["artifacts"], list)

    @pytest.mark.asyncio
    async def test_handler_records_artifacts(
        self, sample_job_with_file_content, mock_context, mock_tune_result
    ):
        """Test handler calls generate_tune_artifacts."""
        from app.jobs.handlers.tune import handle_tune

        with patch("app.jobs.handlers.tune.ParamTuner") as MockTuner:
            mock_tuner_instance = AsyncMock()
            mock_tuner_instance.run = AsyncMock(return_value=mock_tune_result)
            MockTuner.return_value = mock_tuner_instance

            with patch("app.jobs.handlers.tune.TuneRepository"):
                with patch("app.jobs.handlers.tune.KnowledgeBaseRepository"):
                    with patch("app.jobs.handlers.tune.BacktestRepository"):
                        with patch(
                            "app.jobs.handlers.tune.generate_tune_artifacts"
                        ) as mock_gen:
                            mock_gen.return_value = [
                                "tune.json",
                                "trials.csv",
                                "equity_best.csv",
                            ]

                            with patch(
                                "app.jobs.handlers.tune.ArtifactRepository"
                            ) as MockArtifactRepo:
                                mock_artifact_repo = AsyncMock()
                                MockArtifactRepo.return_value = mock_artifact_repo

                                result = await handle_tune(
                                    sample_job_with_file_content, mock_context
                                )

        # Verify generate_tune_artifacts was called
        mock_gen.assert_called_once()

        assert result["artifacts"] == ["tune.json", "trials.csv", "equity_best.csv"]

    @pytest.mark.asyncio
    async def test_handler_missing_tune_id_raises(self, mock_context):
        """Test handler raises ValueError if tune_id missing."""
        from app.jobs.handlers.tune import handle_tune

        job = Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.RUNNING,
            payload={
                "workspace_id": str(uuid4()),
                # Missing tune_id
                "strategy_entity_id": str(uuid4()),
                "ohlcv_file_content": base64.b64encode(b"test").decode("ascii"),
            },
        )

        with pytest.raises(ValueError, match="tune_id"):
            await handle_tune(job, mock_context)

    @pytest.mark.asyncio
    async def test_handler_missing_ohlcv_data_raises(self, mock_context):
        """Test handler raises ValueError if neither ohlcv_file_content nor data_source."""
        from app.jobs.handlers.tune import handle_tune

        job = Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.RUNNING,
            payload={
                "workspace_id": str(uuid4()),
                "tune_id": str(uuid4()),
                "strategy_entity_id": str(uuid4()),
                # Missing both ohlcv_file_content and data_source
            },
        )

        with pytest.raises(ValueError, match="ohlcv_file_content.*data_source"):
            await handle_tune(job, mock_context)


class TestGenerateTuneArtifacts:
    """Tests for generate_tune_artifacts helper."""

    @pytest.fixture
    def sample_tune_result(self):
        """Create a sample TuneResult."""
        from app.services.backtest.tuner import TuneResult

        tune_id = uuid4()
        best_run_id = uuid4()
        return TuneResult(
            tune_id=tune_id,
            status="completed",
            n_trials=3,
            trials_completed=3,
            best_run_id=best_run_id,
            best_params={"period": 14},
            best_score=1.5,
            leaderboard=[],
            warnings=[],
        )

    @pytest.mark.asyncio
    async def test_generate_artifacts_returns_artifact_paths(
        self, sample_tune_result, tmp_path, monkeypatch
    ):
        """Test that generate_tune_artifacts returns list of artifact paths."""
        from app.jobs.handlers.tune import generate_tune_artifacts

        # Mock the artifacts directory setting
        monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))

        mock_artifact_repo = AsyncMock()
        mock_artifact_repo.create = AsyncMock()

        # Mock tune_repo.get_tune() to return tune metadata
        mock_tune_repo = AsyncMock()
        mock_tune_repo.get_tune = AsyncMock(
            return_value={
                "id": sample_tune_result.tune_id,
                "workspace_id": uuid4(),
                "strategy_entity_id": uuid4(),
                "strategy_name": "test_strategy",
                "param_space": {"period": [10, 14, 20]},
                "search_type": "grid",
                "n_trials": 3,
                "seed": 42,
                "objective_metric": "sharpe",
                "objective_type": "sharpe",
                "objective_params": None,
                "oos_ratio": 0.2,
                "gates": {"max_drawdown_pct": 20},
                "created_at": datetime.now(timezone.utc),
            }
        )
        mock_tune_repo.list_tune_runs = AsyncMock(
            return_value=(
                [
                    {
                        "trial_index": 0,
                        "run_id": uuid4(),
                        "params": {"period": 14},
                        "score": 1.5,
                        "score_is": 1.6,
                        "score_oos": 1.4,
                        "objective_score": 1.5,
                        "status": "completed",
                        "metrics_is": {"return_pct": 10.0},
                        "metrics_oos": {"return_pct": 8.0},
                    }
                ],
                1,
            )
        )

        # Mock backtest_repo.get_run() for equity curve
        mock_backtest_repo = AsyncMock()
        mock_backtest_repo.get_run = AsyncMock(
            return_value={
                "equity_curve": [
                    {"t": "2024-01-01T00:00:00Z", "equity": 10000},
                    {"t": "2024-01-02T00:00:00Z", "equity": 10500},
                ],
            }
        )

        workspace_id = uuid4()

        # Patch get_settings to use tmp_path as artifacts_dir
        from unittest.mock import patch

        mock_settings = MagicMock()
        mock_settings.artifacts_dir = str(tmp_path)
        with patch("app.jobs.handlers.tune.get_settings", return_value=mock_settings):
            artifacts = await generate_tune_artifacts(
                tune_id=sample_tune_result.tune_id,
                result=sample_tune_result,
                workspace_id=workspace_id,
                data_revision=None,
                artifact_repo=mock_artifact_repo,
                tune_repo=mock_tune_repo,
                backtest_repo=mock_backtest_repo,
            )

        # Should return list of artifact paths
        assert isinstance(artifacts, list)
        assert len(artifacts) == 3  # tune.json, trials.csv, equity_best.csv
        assert mock_artifact_repo.create.call_count == 3

        # Check files were created
        tune_dir = tmp_path / "tunes" / str(sample_tune_result.tune_id)
        assert (tune_dir / "tune.json").exists()
        assert (tune_dir / "trials.csv").exists()
        assert (tune_dir / "equity_best.csv").exists()
