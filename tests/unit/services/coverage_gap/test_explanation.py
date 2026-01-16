"""Unit tests for strategy explanation generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.coverage_gap.explanation import (
    ExplanationError,
    StrategyExplanation,
    generate_strategy_explanation,
)


class TestGenerateStrategyExplanation:
    """Tests for generate_strategy_explanation function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=MagicMock(
                text="This strategy matches your breakout intent because it monitors 52-week highs.",
                model="claude-3-haiku",
                provider="anthropic",
                latency_ms=250.0,
            )
        )
        return llm

    @pytest.fixture
    def sample_intent(self):
        """Sample intent JSON."""
        return {
            "strategy_archetypes": ["breakout", "momentum"],
            "indicators": ["volume", "atr"],
            "timeframe_buckets": ["swing"],
            "timeframe_explicit": ["1d"],
            "symbols": ["BTC", "ETH"],
            "risk_terms": ["stop_loss"],
            "topics": ["crypto", "trading"],
        }

    @pytest.fixture
    def sample_strategy(self):
        """Sample strategy dict."""
        return {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "52W Breakout Strategy",
            "description": "Enters when price breaks 52-week high",
            "tags": {
                "strategy_archetypes": ["breakout"],
                "indicators": ["volume"],
                "timeframe_buckets": ["swing"],
            },
            "backtest_summary": {
                "status": "validated",
                "best_oos_score": 1.25,
                "max_drawdown": 0.15,
            },
        }

    @pytest.mark.asyncio
    async def test_generates_explanation_successfully(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Successfully generates explanation with all data."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout", "volume", "swing"],
                match_score=0.75,
            )

        assert isinstance(result, StrategyExplanation)
        assert result.strategy_id == "123e4567-e89b-12d3-a456-426614174000"
        assert result.strategy_name == "52W Breakout Strategy"
        assert "breakout" in result.explanation.lower()
        assert result.model == "claude-3-haiku"
        assert result.provider == "anthropic"
        assert result.latency_ms == 250.0

    @pytest.mark.asyncio
    async def test_raises_error_when_llm_not_configured(
        self, sample_intent, sample_strategy
    ):
        """Raises ExplanationError when LLM is not configured."""
        mock_status = MagicMock(enabled=False, provider_config="auto")
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=None
        ), patch(
            "app.services.coverage_gap.explanation.get_llm_status",
            return_value=mock_status,
        ):
            with pytest.raises(ExplanationError) as exc_info:
                await generate_strategy_explanation(
                    intent_json=sample_intent,
                    strategy=sample_strategy,
                    matched_tags=["breakout"],
                )

            assert "LLM not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_empty_intent(self, mock_llm, sample_strategy):
        """Handles empty intent gracefully."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json={},
                strategy=sample_strategy,
                matched_tags=[],
            )

        assert isinstance(result, StrategyExplanation)
        # LLM generate was still called
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_minimal_strategy(self, mock_llm, sample_intent):
        """Handles minimal strategy data."""
        minimal_strategy = {
            "id": "abc123",
            "name": "Simple Strategy",
        }

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=minimal_strategy,
                matched_tags=[],
            )

        assert result.strategy_name == "Simple Strategy"
        assert result.strategy_id == "abc123"

    @pytest.mark.asyncio
    async def test_includes_matched_tags_in_prompt(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Matched tags are included in the LLM prompt."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout", "volume"],
                match_score=0.85,
            )

        # Check the prompt includes matched tags
        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs["messages"]
        user_prompt = messages[1]["content"]
        assert "breakout" in user_prompt
        assert "volume" in user_prompt
        assert "85%" in user_prompt  # match_score formatted

    @pytest.mark.asyncio
    async def test_includes_backtest_summary_in_prompt(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Backtest summary is included when available."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=[],
            )

        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs["messages"]
        user_prompt = messages[1]["content"]
        assert "validated" in user_prompt.lower()
        assert "1.25" in user_prompt  # best_oos_score

    @pytest.mark.asyncio
    async def test_handles_llm_error(self, mock_llm, sample_intent, sample_strategy):
        """Wraps LLM errors in ExplanationError."""
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM API timeout"))

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            with pytest.raises(ExplanationError) as exc_info:
                await generate_strategy_explanation(
                    intent_json=sample_intent,
                    strategy=sample_strategy,
                    matched_tags=[],
                )

            assert "Failed to generate explanation" in str(exc_info.value)
            assert "LLM API timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_matched_tags_uses_semantic_message(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """When no matched tags, indicates semantic similarity."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=[],  # No explicit matches
            )

        call_args = mock_llm.generate.call_args
        messages = call_args.kwargs["messages"]
        user_prompt = messages[1]["content"]
        assert "semantic similarity" in user_prompt.lower()
