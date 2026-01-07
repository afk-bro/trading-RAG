"""Unit tests for LLM grounding contract behavior."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm_base import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing generate_answer behavior."""

    def __init__(self):
        super().__init__(answer_model="test-model", rerank_model=None)
        self.provider = "mock"
        self.last_messages = None

    async def generate(self, *, messages, model=None, max_tokens=2000):
        """Capture messages for inspection and return mock response."""
        self.last_messages = messages
        return LLMResponse(
            text="Mock response",
            model=model or self.answer_model,
            provider=self.provider,
        )


class TestGroundingContract:
    """Tests for grounding contract in generate_answer."""

    @pytest.fixture
    def client(self):
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_grounding_contract_in_system_prompt(self, client):
        """Verify the grounding contract is included in system prompt."""
        chunks = [{"content": "AAPL stock went up today.", "title": "News"}]

        await client.generate_answer(
            question="What is a stock?",
            chunks=chunks,
        )

        system_message = client.last_messages[0]
        assert system_message["role"] == "system"

        # Check key grounding rules are present
        content = system_message["content"]
        assert "GROUNDING CONTRACT" in content
        assert "ONLY the provided context" in content
        assert "does not specify" in content.lower() or "not specify" in content.lower()
        assert "Do NOT use general knowledge" in content

    @pytest.mark.asyncio
    async def test_structured_output_format_requested(self, client):
        """Verify the prompt requests structured output format."""
        chunks = [{"content": "Test content", "title": "Test"}]

        await client.generate_answer(
            question="Test question?",
            chunks=chunks,
        )

        user_message = client.last_messages[1]
        assert user_message["role"] == "user"

        content = user_message["content"]
        # Check for structured output sections
        assert "**Answer:**" in content
        assert "**Supported by context:**" in content
        assert "**Not specified in context:**" in content

    @pytest.mark.asyncio
    async def test_chunks_formatted_with_citations(self, client):
        """Verify chunks are formatted with citation numbers."""
        chunks = [
            {"content": "First chunk content", "title": "Doc A"},
            {"content": "Second chunk content", "title": "Doc B"},
        ]

        await client.generate_answer(
            question="Test?",
            chunks=chunks,
        )

        user_message = client.last_messages[1]["content"]
        assert "[1] Doc A:" in user_message
        assert "[2] Doc B:" in user_message
        assert "First chunk content" in user_message
        assert "Second chunk content" in user_message

    @pytest.mark.asyncio
    async def test_locator_label_included(self, client):
        """Verify locator labels are included in context formatting."""
        chunks = [
            {
                "content": "Page content",
                "title": "Document",
                "locator_label": "p. 42",
            }
        ]

        await client.generate_answer(
            question="Test?",
            chunks=chunks,
        )

        user_message = client.last_messages[1]["content"]
        assert "[1] Document (p. 42):" in user_message


class TestGroundingBehavior:
    """Integration-style tests for grounding behavior with real prompts.

    These tests verify the prompt structure encourages proper grounding.
    Actual LLM response behavior depends on the model.
    """

    def test_grounding_contract_constant_exists(self):
        """Verify GROUNDING_CONTRACT class constant is defined."""
        assert hasattr(BaseLLMClient, "GROUNDING_CONTRACT")
        contract = BaseLLMClient.GROUNDING_CONTRACT

        # Must contain key grounding rules
        assert "ONLY" in contract
        assert "context" in contract.lower()
        assert "cite" in contract.lower() or "citation" in contract.lower()

    def test_grounding_contract_prohibits_outside_knowledge(self):
        """Verify contract explicitly prohibits using outside knowledge."""
        contract = BaseLLMClient.GROUNDING_CONTRACT
        assert "general knowledge" in contract.lower() or "outside" in contract.lower()

    def test_grounding_contract_requires_admission_of_gaps(self):
        """Verify contract requires stating when context doesn't specify."""
        contract = BaseLLMClient.GROUNDING_CONTRACT
        # Should mention what to do when context doesn't have the answer
        assert "does not specify" in contract.lower() or "not specify" in contract.lower()
