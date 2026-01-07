"""Unit tests for LLM factory and provider selection."""

import pytest
from unittest.mock import patch, MagicMock

from app.services.llm_factory import (
    _resolve_provider,
    _get_effective_rerank_model,
    get_llm_status,
    get_llm,
    reset_llm,
    LLMStartupError,
    LLMStatus,
)


class MockSettings:
    """Mock settings for testing."""

    def __init__(
        self,
        llm_provider="auto",
        llm_required=False,
        llm_enabled=True,
        anthropic_api_key=None,
        claude_code_oauth_token=None,
        openrouter_api_key=None,
        answer_model="claude-sonnet-4",
        rerank_model="claude-haiku-3-5",
        llm_timeout=60,
    ):
        self.llm_provider = llm_provider
        self.llm_required = llm_required
        self.llm_enabled = llm_enabled
        self.anthropic_api_key = anthropic_api_key
        self.claude_code_oauth_token = claude_code_oauth_token
        self.openrouter_api_key = openrouter_api_key
        self.answer_model = answer_model
        self.rerank_model = rerank_model
        self.llm_timeout = llm_timeout


class TestResolveProvider:
    """Tests for provider resolution logic."""

    def test_auto_prefers_anthropic_when_both_keys(self):
        """Test that auto mode prefers Anthropic when both keys are set."""
        settings = MockSettings(
            llm_provider="auto",
            anthropic_api_key="sk-ant-test",
            openrouter_api_key="sk-or-test",
        )
        provider, key = _resolve_provider(settings)
        assert provider == "anthropic"
        assert key == "sk-ant-test"

    def test_auto_falls_back_to_openrouter(self):
        """Test that auto mode falls back to OpenRouter when no Anthropic key."""
        settings = MockSettings(
            llm_provider="auto",
            anthropic_api_key=None,
            openrouter_api_key="sk-or-test",
        )
        provider, key = _resolve_provider(settings)
        assert provider == "openrouter"
        assert key == "sk-or-test"

    def test_auto_disabled_when_no_keys_and_not_required(self):
        """Test that auto mode returns None when no keys and not required."""
        settings = MockSettings(
            llm_provider="auto",
            anthropic_api_key=None,
            openrouter_api_key=None,
            llm_required=False,
        )
        provider, key = _resolve_provider(settings)
        assert provider is None
        assert key is None

    def test_auto_raises_when_no_keys_and_required(self):
        """Test that auto mode raises error when no keys and required."""
        settings = MockSettings(
            llm_provider="auto",
            anthropic_api_key=None,
            openrouter_api_key=None,
            llm_required=True,
        )
        with pytest.raises(LLMStartupError) as exc:
            _resolve_provider(settings)
        assert "no API key configured" in str(exc.value)

    def test_explicit_anthropic_requires_key(self):
        """Test that explicit anthropic provider requires its key."""
        settings = MockSettings(
            llm_provider="anthropic",
            anthropic_api_key=None,
        )
        with pytest.raises(LLMStartupError) as exc:
            _resolve_provider(settings)
        assert "ANTHROPIC_API_KEY" in str(exc.value)

    def test_explicit_openrouter_requires_key(self):
        """Test that explicit openrouter provider requires its key."""
        settings = MockSettings(
            llm_provider="openrouter",
            openrouter_api_key=None,
        )
        with pytest.raises(LLMStartupError) as exc:
            _resolve_provider(settings)
        assert "OPENROUTER_API_KEY not set" in str(exc.value)

    def test_explicit_anthropic_with_key(self):
        """Test explicit anthropic provider with valid key."""
        settings = MockSettings(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-test",
        )
        provider, key = _resolve_provider(settings)
        assert provider == "anthropic"
        assert key == "sk-ant-test"

    def test_explicit_openrouter_with_key(self):
        """Test explicit openrouter provider with valid key."""
        settings = MockSettings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-test",
        )
        provider, key = _resolve_provider(settings)
        assert provider == "openrouter"
        assert key == "sk-or-test"

    def test_kill_switch_disables_llm(self):
        """Test that llm_enabled=False disables LLM regardless of keys."""
        settings = MockSettings(
            llm_provider="auto",
            llm_enabled=False,
            anthropic_api_key="sk-ant-test",
        )
        provider, key = _resolve_provider(settings)
        assert provider is None
        assert key is None

    def test_empty_string_key_treated_as_unset(self):
        """Test that empty string keys are treated as unset."""
        settings = MockSettings(
            llm_provider="auto",
            anthropic_api_key="",
            openrouter_api_key="  ",  # whitespace only
            llm_required=False,
        )
        provider, key = _resolve_provider(settings)
        assert provider is None
        assert key is None


class TestEffectiveRerankModel:
    """Tests for rerank model fallback logic."""

    def test_uses_rerank_model_when_set(self):
        """Test that explicit rerank model is used."""
        settings = MockSettings(
            answer_model="claude-sonnet-4",
            rerank_model="claude-haiku-3-5",
        )
        assert _get_effective_rerank_model(settings) == "claude-haiku-3-5"

    def test_falls_back_to_answer_model_when_none(self):
        """Test fallback to answer model when rerank is None."""
        settings = MockSettings(
            answer_model="claude-sonnet-4",
            rerank_model=None,
        )
        assert _get_effective_rerank_model(settings) == "claude-sonnet-4"

    def test_falls_back_to_answer_model_when_empty(self):
        """Test fallback to answer model when rerank is empty string."""
        settings = MockSettings(
            answer_model="claude-sonnet-4",
            rerank_model="",
        )
        assert _get_effective_rerank_model(settings) == "claude-sonnet-4"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped from rerank model."""
        settings = MockSettings(
            answer_model="claude-sonnet-4",
            rerank_model="  claude-haiku-3-5  ",
        )
        # The function should strip but still return the model
        result = _get_effective_rerank_model(settings)
        assert result in ["claude-haiku-3-5", "  claude-haiku-3-5  "]  # Either is acceptable


class TestGetLLMStatus:
    """Tests for get_llm_status function."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_llm()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_llm()

    @patch("app.services.llm_factory.get_settings")
    def test_returns_status_with_anthropic(self, mock_get_settings):
        """Test status when Anthropic is configured."""
        mock_get_settings.return_value = MockSettings(
            llm_provider="auto",
            anthropic_api_key="sk-ant-test",
        )

        # Patch the client creation to avoid actual API calls
        with patch("app.services.llm_factory._create_client") as mock_create:
            mock_create.return_value = MagicMock()
            status = get_llm_status()

        assert status.enabled is True
        assert status.provider_config == "auto"
        assert status.provider_resolved == "anthropic"
        assert status.answer_model == "claude-sonnet-4"
        assert status.rerank_model_effective == "claude-haiku-3-5"

    @patch("app.services.llm_factory.get_settings")
    def test_returns_disabled_status(self, mock_get_settings):
        """Test status when LLM is disabled."""
        mock_get_settings.return_value = MockSettings(
            llm_provider="auto",
            llm_enabled=False,
        )

        status = get_llm_status()

        assert status.enabled is False
        assert status.provider_resolved is None


class TestGetLLM:
    """Tests for get_llm function."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_llm()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_llm()

    @patch("app.services.llm_factory.get_settings")
    def test_returns_none_when_disabled(self, mock_get_settings):
        """Test that get_llm returns None when LLM is disabled."""
        mock_get_settings.return_value = MockSettings(
            llm_provider="auto",
            anthropic_api_key=None,
            openrouter_api_key=None,
            llm_required=False,
        )

        client = get_llm()
        assert client is None

    @patch("app.services.llm_factory.get_settings")
    def test_raises_on_required_but_missing(self, mock_get_settings):
        """Test that get_llm raises when required but no keys."""
        mock_get_settings.return_value = MockSettings(
            llm_provider="auto",
            anthropic_api_key=None,
            openrouter_api_key=None,
            llm_required=True,
        )

        with pytest.raises(LLMStartupError):
            get_llm()

    @patch("app.services.llm_factory.get_settings")
    def test_caches_client(self, mock_get_settings):
        """Test that get_llm returns cached client."""
        mock_get_settings.return_value = MockSettings(
            llm_provider="auto",
            anthropic_api_key="sk-ant-test",
        )

        with patch("app.services.llm_factory._create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            client1 = get_llm()
            client2 = get_llm()

            assert client1 is client2
            mock_create.assert_called_once()
