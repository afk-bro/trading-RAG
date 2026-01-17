"""Unit tests for strategy explanation generation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.coverage_gap.explanation import (
    CachedExplanation,
    StrategyExplanation,
    compute_confidence_qualifier,
    generate_strategy_explanation,
    _make_fallback_explanation,
    USER_SAFE_ERRORS,
)
from app.services.llm_base import (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAPIError,
)


class TestGenerateStrategyExplanation:
    """Tests for generate_strategy_explanation function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=MagicMock(
                text="This strategy matches your breakout intent via 52-week highs.",
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
    async def test_returns_fallback_when_llm_not_configured(
        self, sample_intent, sample_strategy
    ):
        """Returns fallback when LLM is not configured."""
        mock_status = MagicMock(enabled=False, provider_config="auto")
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=None
        ), patch(
            "app.services.coverage_gap.explanation.get_llm_status",
            return_value=mock_status,
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
            )

        assert result.degraded is True
        assert result.reason_code == "llm_unconfigured"
        assert result.error == USER_SAFE_ERRORS["llm_unconfigured"]
        assert result.model == "fallback"
        assert result.provider == "fallback"

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
    async def test_handles_llm_error_with_fallback(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Returns fallback on LLM errors instead of raising."""
        mock_llm.generate = AsyncMock(
            side_effect=LLMAPIError(
                message="Internal server error",
                provider="openai",
                status_code=500,
            )
        )

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=[],
            )

        assert result.degraded is True
        assert result.reason_code == "llm_error"
        assert result.error == USER_SAFE_ERRORS["llm_error"]

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

    @pytest.mark.asyncio
    async def test_verbosity_short_uses_brief_prompt(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Short verbosity uses brief prompt instructions."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
                verbosity="short",
            )

        assert result.verbosity == "short"
        call_args = mock_llm.generate.call_args
        assert call_args.kwargs["max_tokens"] == 300

    @pytest.mark.asyncio
    async def test_verbosity_detailed_uses_longer_prompt(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Detailed verbosity uses longer prompt instructions."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
                verbosity="detailed",
            )

        assert result.verbosity == "detailed"
        call_args = mock_llm.generate.call_args
        assert call_args.kwargs["max_tokens"] == 600

    @pytest.mark.asyncio
    async def test_returns_confidence_qualifier(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Result includes deterministic confidence qualifier."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout", "volume", "swing"],
                match_score=0.75,
            )

        assert result.confidence_qualifier
        assert "Confidence:" in result.confidence_qualifier
        assert "High" in result.confidence_qualifier  # High due to 3 tags + backtest

    @pytest.mark.asyncio
    async def test_includes_generated_at_timestamp(
        self, mock_llm, sample_intent, sample_strategy
    ):
        """Result includes generated_at timestamp."""
        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=[],
            )

        assert result.generated_at
        assert "T" in result.generated_at  # ISO format

    @pytest.mark.asyncio
    async def test_returns_fallback_on_timeout(self, sample_intent, sample_strategy):
        """Returns fallback when LLM request times out."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
                timeout_seconds=0.1,
            )

        assert result.degraded is True
        assert result.reason_code == "llm_timeout"
        assert result.error == USER_SAFE_ERRORS["llm_timeout"]

    @pytest.mark.asyncio
    async def test_returns_fallback_on_rate_limit(self, sample_intent, sample_strategy):
        """Returns fallback when rate limited."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            side_effect=LLMRateLimitError(
                message="Rate limit exceeded",
                provider="openai",
                retry_after_seconds=60,
            )
        )

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
            )

        assert result.degraded is True
        assert result.reason_code == "llm_rate_limit"
        assert result.error == USER_SAFE_ERRORS["llm_rate_limit"]

    @pytest.mark.asyncio
    async def test_returns_fallback_on_llm_timeout_error(
        self, sample_intent, sample_strategy
    ):
        """Returns fallback on LLMTimeoutError from provider."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            side_effect=LLMTimeoutError(
                message="Provider timeout",
                provider="openai",
                timeout_seconds=30.0,
            )
        )

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
            )

        assert result.degraded is True
        assert result.reason_code == "llm_error"  # Mapped to generic llm_error

    @pytest.mark.asyncio
    async def test_returns_fallback_on_unexpected_error(
        self, sample_intent, sample_strategy
    ):
        """Returns fallback on unexpected exception (never exposes raw error)."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            side_effect=ValueError("Unexpected internal error")
        )

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout"],
            )

        assert result.degraded is True
        assert result.reason_code == "llm_error"
        # User never sees raw exception message
        assert result.error == USER_SAFE_ERRORS["llm_error"]
        assert "Unexpected" not in result.error

    @pytest.mark.asyncio
    async def test_fallback_still_computes_confidence(
        self, sample_intent, sample_strategy
    ):
        """Fallback responses still have deterministic confidence qualifier."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch(
            "app.services.coverage_gap.explanation.get_llm", return_value=mock_llm
        ):
            result = await generate_strategy_explanation(
                intent_json=sample_intent,
                strategy=sample_strategy,
                matched_tags=["breakout", "volume", "swing"],
                match_score=0.8,
            )

        # Confidence qualifier should be computed despite degraded mode
        assert "Confidence:" in result.confidence_qualifier
        assert result.degraded is True


class TestMakeFallbackExplanation:
    """Tests for _make_fallback_explanation helper function."""

    def test_fallback_with_matched_tags(self):
        """Fallback explanation includes tag information."""
        result = _make_fallback_explanation(
            strategy_id="strat-123",
            strategy_name="Test Strategy",
            reason_code="llm_timeout",
            matched_tags=["momentum", "RSI", "daily"],
            match_score=0.8,
            verbosity="short",
        )

        assert result.degraded is True
        assert result.reason_code == "llm_timeout"
        assert result.error == USER_SAFE_ERRORS["llm_timeout"]
        assert result.model == "fallback"
        assert result.provider == "fallback"
        assert "3 overlapping tag" in result.explanation
        assert "momentum" in result.explanation

    def test_fallback_with_match_score_only(self):
        """Fallback shows match score when no tags."""
        result = _make_fallback_explanation(
            strategy_id="strat-123",
            strategy_name="Test Strategy",
            reason_code="llm_rate_limit",
            matched_tags=[],
            match_score=0.75,
            verbosity="short",
        )

        assert result.degraded is True
        assert "75%" in result.explanation

    def test_fallback_generic_when_no_data(self):
        """Generic fallback when no tags or score."""
        result = _make_fallback_explanation(
            strategy_id="strat-123",
            strategy_name="Test Strategy",
            reason_code="llm_unconfigured",
            matched_tags=[],
            match_score=None,
            verbosity="short",
        )

        assert result.degraded is True
        assert "potential match" in result.explanation

    def test_fallback_truncates_many_tags(self):
        """Fallback truncates when more than 3 tags."""
        result = _make_fallback_explanation(
            strategy_id="strat-123",
            strategy_name="Test Strategy",
            reason_code="llm_error",
            matched_tags=["tag1", "tag2", "tag3", "tag4", "tag5"],
            match_score=0.9,
            verbosity="short",
        )

        assert "5 overlapping tag" in result.explanation
        assert "..." in result.explanation  # Truncation indicator


class TestComputeConfidenceQualifier:
    """Tests for compute_confidence_qualifier function."""

    def test_high_confidence_many_tags_validated_backtest(self):
        """High confidence with many matched tags and validated backtest."""
        result = compute_confidence_qualifier(
            matched_tags=["breakout", "volume", "swing"],
            match_score=0.8,
            has_backtest=True,
            backtest_validated=True,
        )

        assert "Confidence: High" in result
        assert "3 matching tags" in result
        assert "validated backtest" in result

    def test_medium_confidence_some_tags(self):
        """Medium confidence with partial tag overlap."""
        # Calculate: tag=2/3=0.67*0.4=0.27 + score=0.6*0.4=0.24 + backtest=0.1*0.2=0.02 = 0.53
        result = compute_confidence_qualifier(
            matched_tags=["breakout", "momentum"],
            match_score=0.6,
            has_backtest=True,
            backtest_validated=False,
        )

        assert "Confidence: Medium" in result
        assert "2 matching tags" in result
        assert "backtest available" in result

    def test_low_confidence_no_tags_no_backtest(self):
        """Low confidence with no tags and no backtest."""
        result = compute_confidence_qualifier(
            matched_tags=[],
            match_score=0.3,
            has_backtest=False,
            backtest_validated=False,
        )

        assert "Confidence: Low" in result
        assert "general alignment" in result

    def test_semantic_similarity_reason_when_no_tags(self):
        """Uses semantic similarity reason when no matched tags but good score."""
        result = compute_confidence_qualifier(
            matched_tags=[],
            match_score=0.7,
            has_backtest=False,
            backtest_validated=False,
        )

        assert "semantic similarity" in result

    def test_handles_none_match_score(self):
        """Handles None match score gracefully."""
        result = compute_confidence_qualifier(
            matched_tags=["breakout"],
            match_score=None,
            has_backtest=False,
            backtest_validated=False,
        )

        assert "Confidence:" in result


class TestCachedExplanation:
    """Tests for CachedExplanation dataclass."""

    def test_to_dict_serialization(self):
        """to_dict() returns correct structure for JSONB."""
        cached = CachedExplanation(
            explanation="Test explanation",
            confidence_qualifier="Confidence: High — based on 3 matching tags.",
            model="claude-3-haiku",
            provider="anthropic",
            verbosity="short",
            latency_ms=250.0,
            generated_at="2025-01-15T12:00:00Z",
            strategy_updated_at="2025-01-14T10:00:00Z",
        )

        result = cached.to_dict()

        assert result["explanation"] == "Test explanation"
        assert (
            result["confidence_qualifier"]
            == "Confidence: High — based on 3 matching tags."
        )
        assert result["model"] == "claude-3-haiku"
        assert result["provider"] == "anthropic"
        assert result["verbosity"] == "short"
        assert result["latency_ms"] == 250.0
        assert result["generated_at"] == "2025-01-15T12:00:00Z"
        assert result["strategy_updated_at"] == "2025-01-14T10:00:00Z"

    def test_from_dict_deserialization(self):
        """from_dict() creates correct CachedExplanation instance."""
        data = {
            "explanation": "Test explanation",
            "confidence_qualifier": "Confidence: Medium",
            "model": "gpt-4",
            "provider": "openai",
            "verbosity": "detailed",
            "latency_ms": 500.0,
            "generated_at": "2025-01-15T12:00:00Z",
            "strategy_updated_at": None,
        }

        result = CachedExplanation.from_dict(data)

        assert result.explanation == "Test explanation"
        assert result.confidence_qualifier == "Confidence: Medium"
        assert result.model == "gpt-4"
        assert result.provider == "openai"
        assert result.verbosity == "detailed"
        assert result.latency_ms == 500.0
        assert result.generated_at == "2025-01-15T12:00:00Z"
        assert result.strategy_updated_at is None

    def test_from_dict_handles_missing_fields(self):
        """from_dict() handles missing fields with defaults."""
        data = {
            "explanation": "Minimal data",
        }

        result = CachedExplanation.from_dict(data)

        assert result.explanation == "Minimal data"
        assert result.confidence_qualifier == ""
        assert result.model == ""
        assert result.provider == ""
        assert result.verbosity == "short"  # default
        assert result.latency_ms is None
        assert result.generated_at == ""
        assert result.strategy_updated_at is None

    def test_roundtrip_serialization(self):
        """to_dict() -> from_dict() roundtrip preserves all data."""
        original = CachedExplanation(
            explanation="Roundtrip test",
            confidence_qualifier="Confidence: Low",
            model="claude-3-sonnet",
            provider="anthropic",
            verbosity="detailed",
            latency_ms=123.45,
            generated_at="2025-01-15T12:00:00Z",
            strategy_updated_at="2025-01-10T08:00:00Z",
        )

        roundtrip = CachedExplanation.from_dict(original.to_dict())

        assert roundtrip.explanation == original.explanation
        assert roundtrip.confidence_qualifier == original.confidence_qualifier
        assert roundtrip.model == original.model
        assert roundtrip.provider == original.provider
        assert roundtrip.verbosity == original.verbosity
        assert roundtrip.latency_ms == original.latency_ms
        assert roundtrip.generated_at == original.generated_at
        assert roundtrip.strategy_updated_at == original.strategy_updated_at
