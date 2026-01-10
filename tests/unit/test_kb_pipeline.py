"""Unit tests for knowledge base extraction/verification pipeline."""

import pytest
import json

from app.services.kb_types import (
    EntityType,
    ClaimType,
    RelationType,
    VerificationStatus,
    EvidencePointer,
    ExtractedEntity,
    ExtractedClaim,
    ExtractionResult,
    ClaimVerdict,
    VerificationResult,
    PipelineResult,
    PersistenceStats,
)
from app.services.kb_prompts import (
    build_extraction_prompt,
    build_verification_prompt,
    build_synthesis_prompt,
    extract_json_from_response,
    EXTRACTION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from app.services.llm_base import LLMResponse


class TestKBTypes:
    """Tests for Pydantic model validation."""

    def test_entity_type_enum_values(self):
        """Test EntityType enum has expected values."""
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.INDICATOR.value == "indicator"
        assert EntityType.STRATEGY.value == "strategy"
        assert EntityType.EQUATION.value == "equation"

    def test_claim_type_enum_values(self):
        """Test ClaimType enum has expected values."""
        assert ClaimType.DEFINITION.value == "definition"
        assert ClaimType.RULE.value == "rule"
        assert ClaimType.WARNING.value == "warning"
        assert ClaimType.PARAMETER.value == "parameter"

    def test_relation_type_enum_values(self):
        """Test RelationType enum has expected values."""
        assert RelationType.USES.value == "uses"
        assert RelationType.REQUIRES.value == "requires"
        assert RelationType.DERIVED_FROM.value == "derived_from"
        assert RelationType.CONTRADICTS.value == "contradicts"

    def test_verification_status_enum_values(self):
        """Test VerificationStatus enum has expected values."""
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.WEAK.value == "weak"
        assert VerificationStatus.REJECTED.value == "rejected"

    def test_evidence_pointer_validation(self):
        """Test EvidencePointer model validation."""
        ev = EvidencePointer(chunk_index=0, quote="Test quote", relevance=0.9)
        assert ev.chunk_index == 0
        assert ev.quote == "Test quote"
        assert ev.relevance == 0.9

    def test_evidence_pointer_relevance_bounds(self):
        """Test EvidencePointer relevance must be 0-1."""
        with pytest.raises(ValueError):
            EvidencePointer(chunk_index=0, quote="Test", relevance=1.5)
        with pytest.raises(ValueError):
            EvidencePointer(chunk_index=0, quote="Test", relevance=-0.1)

    def test_extracted_entity_creation(self):
        """Test ExtractedEntity model creation."""
        entity = ExtractedEntity(
            type=EntityType.INDICATOR,
            name="RSI",
            aliases=["Relative Strength Index"],
            description="Momentum indicator",
            evidence=[
                EvidencePointer(chunk_index=0, quote="RSI measures...", relevance=0.9)
            ],
        )
        assert entity.type == EntityType.INDICATOR
        assert entity.name == "RSI"
        assert "Relative Strength Index" in entity.aliases

    def test_extracted_claim_requires_evidence(self):
        """Test ExtractedClaim requires at least one evidence."""
        with pytest.raises(ValueError):
            ExtractedClaim(
                claim_type=ClaimType.DEFINITION,
                text="Test claim",
                evidence=[],  # Empty evidence should fail
            )

    def test_extracted_claim_valid(self):
        """Test valid ExtractedClaim creation."""
        claim = ExtractedClaim(
            claim_type=ClaimType.RULE,
            text="RSI above 70 indicates overbought",
            entity_name="RSI",
            entity_type=EntityType.INDICATOR,
            confidence=0.8,
            evidence=[
                EvidencePointer(chunk_index=0, quote="When RSI > 70...", relevance=0.9)
            ],
        )
        assert claim.claim_type == ClaimType.RULE
        assert claim.confidence == 0.8

    def test_claim_verdict_creation(self):
        """Test ClaimVerdict model creation."""
        verdict = ClaimVerdict(
            claim_index=0,
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            reason="Evidence directly supports claim",
            corrected_text=None,
        )
        assert verdict.status == VerificationStatus.VERIFIED
        assert verdict.confidence == 0.85

    def test_pipeline_result_creation(self):
        """Test PipelineResult model creation."""
        result = PipelineResult(
            extraction=ExtractionResult(),
            verification=VerificationResult(),
            persistence=PersistenceStats(),
            synthesized_answer="Test answer",
            verified_claims_count=5,
            weak_claims_count=2,
            rejected_claims_count=1,
        )
        assert result.verified_claims_count == 5
        assert result.synthesized_answer == "Test answer"


class TestKBPrompts:
    """Tests for prompt building functions."""

    def test_extraction_prompt_includes_chunks(self):
        """Test extraction prompt includes chunk content."""
        chunks = [
            {"content": "RSI measures momentum", "title": "Technical Analysis"},
            {"content": "MACD shows trend", "title": "Indicators Guide"},
        ]
        prompt = build_extraction_prompt(chunks)

        assert "[0] Technical Analysis" in prompt
        assert "RSI measures momentum" in prompt
        assert "[1] Indicators Guide" in prompt
        assert "MACD shows trend" in prompt

    def test_extraction_prompt_with_question(self):
        """Test extraction prompt includes focus question."""
        chunks = [{"content": "Test content", "title": "Test"}]
        prompt = build_extraction_prompt(chunks, question="What is RSI?")

        assert "FOCUS QUESTION" in prompt
        assert "What is RSI?" in prompt

    def test_verification_prompt_includes_claims(self):
        """Test verification prompt includes claims to verify."""
        chunks = [{"content": "RSI above 70 indicates overbought", "title": "Guide"}]
        claims = [
            {
                "claim_type": "rule",
                "text": "RSI above 70 indicates overbought conditions",
                "evidence": [
                    {"chunk_index": 0, "quote": "RSI above 70 indicates overbought"}
                ],
            }
        ]
        prompt = build_verification_prompt(chunks, claims)

        assert "SOURCE CHUNKS" in prompt
        assert "CLAIMS TO VERIFY" in prompt
        assert "RSI above 70 indicates overbought" in prompt

    def test_synthesis_prompt_includes_verified_claims(self):
        """Test synthesis prompt includes verified claims."""
        claims = [
            {
                "claim_type": "rule",
                "text": "RSI > 70 means overbought",
                "entity_name": "RSI",
                "confidence": 0.9,
            },
            {
                "claim_type": "definition",
                "text": "RSI is a momentum indicator",
                "entity_name": "RSI",
                "confidence": 0.85,
            },
        ]
        prompt = build_synthesis_prompt("What is RSI?", claims)

        assert "VERIFIED CLAIMS" in prompt
        assert "[C1]" in prompt
        assert "[C2]" in prompt
        assert "RSI > 70 means overbought" in prompt
        assert "What is RSI?" in prompt

    def test_synthesis_prompt_with_entities(self):
        """Test synthesis prompt includes entity context."""
        claims = [{"claim_type": "definition", "text": "Test claim", "confidence": 0.9}]
        entities = [
            {
                "name": "RSI",
                "type": "indicator",
                "description": "Relative Strength Index",
            },
        ]
        prompt = build_synthesis_prompt("Test?", claims, entities)

        assert "RELEVANT ENTITIES" in prompt
        assert "RSI" in prompt
        assert "indicator" in prompt

    def test_extract_json_direct(self):
        """Test direct JSON parsing."""
        json_str = '{"entities": [], "claims": [], "relations": []}'
        result = extract_json_from_response(json_str)

        assert result["entities"] == []
        assert result["claims"] == []
        assert result["relations"] == []

    def test_extract_json_from_code_fence(self):
        """Test JSON extraction from markdown code fence."""
        response = """Here's the extraction:
```json
{"entities": [{"name": "RSI", "type": "indicator"}], "claims": [], "relations": []}
```
"""
        result = extract_json_from_response(response)

        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "RSI"

    def test_extract_json_from_plain_fence(self):
        """Test JSON extraction from plain code fence."""
        response = """```
{"entities": [], "claims": [{"text": "Test"}], "relations": []}
```"""
        result = extract_json_from_response(response)

        assert len(result["claims"]) == 1

    def test_extract_json_invalid_raises(self):
        """Test invalid JSON raises ValueError."""
        with pytest.raises(ValueError):
            extract_json_from_response("This is not JSON at all")

    def test_system_prompts_contain_key_instructions(self):
        """Test system prompts contain key instructions."""
        # Extraction prompt
        assert "OUTPUT FORMAT" in EXTRACTION_SYSTEM_PROMPT
        assert "Evidence Required" in EXTRACTION_SYSTEM_PROMPT
        assert "No Hallucination" in EXTRACTION_SYSTEM_PROMPT

        # Verification prompt
        assert "VERIFIED" in VERIFICATION_SYSTEM_PROMPT
        assert "WEAK" in VERIFICATION_SYSTEM_PROMPT
        assert "REJECTED" in VERIFICATION_SYSTEM_PROMPT

        # Synthesis prompt
        assert "GROUNDING CONTRACT" in SYNTHESIS_SYSTEM_PROMPT
        assert "verified claims" in SYNTHESIS_SYSTEM_PROMPT.lower()


class MockLLMClient:
    """Mock LLM client for testing pipeline."""

    def __init__(self):
        self.answer_model = "test-model"
        self._rerank_model = "test-rerank"
        self.provider = "mock"
        self._call_count = 0

    @property
    def effective_rerank_model(self):
        return self._rerank_model or self.answer_model

    async def generate(self, *, messages, model=None, max_tokens=2000):
        """Return mock responses based on prompt content."""
        self._call_count += 1
        user_content = messages[-1]["content"] if messages else ""
        _system_content = (  # noqa: F841
            messages[0]["content"] if len(messages) > 0 else ""
        )  # noqa: F841

        # Check for verification (before extraction since extraction is more specific)
        if "SOURCE CHUNKS" in user_content and "CLAIMS TO VERIFY" in user_content:
            return LLMResponse(
                text=json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_index": 0,
                                "status": "verified",
                                "confidence": 0.85,
                                "reason": "Evidence supports claim",
                            }
                        ]
                    }
                ),
                model=model or self.answer_model,
                provider=self.provider,
            )

        # Check for synthesis
        if "VERIFIED CLAIMS" in user_content:
            return LLMResponse(
                text="""**Answer:**
RSI (Relative Strength Index) is a momentum indicator [C1].
When RSI is above 70, it indicates overbought conditions [C1].

**Based on verified claims:**
- RSI above 70 indicates overbought

**Not addressed by verified claims:**
- None

**Confidence:** high""",
                model=model or self.answer_model,
                provider=self.provider,
            )

        # Extraction response (check last - most general)
        if "CONTEXT CHUNKS" in user_content or "Extract all" in user_content:
            return LLMResponse(
                text=json.dumps(
                    {
                        "entities": [
                            {
                                "type": "indicator",
                                "name": "RSI",
                                "aliases": ["Relative Strength Index"],
                                "description": "Momentum indicator",
                                "evidence": [
                                    {
                                        "chunk_index": 0,
                                        "quote": "RSI is...",
                                        "relevance": 0.9,
                                    }
                                ],
                            }
                        ],
                        "claims": [
                            {
                                "claim_type": "rule",
                                "text": "RSI above 70 indicates overbought",
                                "entity_name": "RSI",
                                "entity_type": "indicator",
                                "confidence": 0.8,
                                "evidence": [
                                    {
                                        "chunk_index": 0,
                                        "quote": "When RSI > 70...",
                                        "relevance": 0.9,
                                    }
                                ],
                            }
                        ],
                        "relations": [],
                    }
                ),
                model=model or self.answer_model,
                provider=self.provider,
            )

        # Default response
        return LLMResponse(
            text="{}", model=model or self.answer_model, provider=self.provider
        )


class TestKBPipeline:
    """Tests for KB Pipeline orchestration."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "content": "RSI (Relative Strength Index) is a momentum indicator. When RSI > 70, it indicates overbought conditions.",  # noqa: E501
                "title": "Technical Analysis Guide",
                "doc_id": "doc-1",
            },
            {
                "content": "MACD shows trend direction and momentum.",
                "title": "Indicator Reference",
                "doc_id": "doc-2",
            },
        ]

    @pytest.mark.asyncio
    async def test_pipeline_extraction_pass(self, mock_llm, sample_chunks):
        """Test extraction pass returns entities and claims."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)
        result = await pipeline.extract(sample_chunks)

        assert len(result.entities) >= 1
        assert result.entities[0].name == "RSI"
        assert len(result.claims) >= 1
        assert "overbought" in result.claims[0].text

    @pytest.mark.asyncio
    async def test_pipeline_verification_pass(self, mock_llm, sample_chunks):
        """Test verification pass returns verdicts."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)
        extraction = await pipeline.extract(sample_chunks)
        verification = await pipeline.verify(sample_chunks, extraction)

        assert len(verification.verdicts) >= 1
        assert verification.verdicts[0].status == VerificationStatus.VERIFIED

    @pytest.mark.asyncio
    async def test_pipeline_synthesis_pass(self, mock_llm, sample_chunks):
        """Test synthesis pass generates answer."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)
        extraction = await pipeline.extract(sample_chunks)
        verification = await pipeline.verify(sample_chunks, extraction)
        answer = await pipeline.synthesize("What is RSI?", extraction, verification)

        assert answer is not None
        assert "RSI" in answer
        assert "[C1]" in answer  # Citations

    @pytest.mark.asyncio
    async def test_pipeline_full_run(self, mock_llm, sample_chunks):
        """Test full pipeline run returns complete result."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)
        result = await pipeline.run(
            sample_chunks, question="What is RSI?", synthesize=True
        )

        assert isinstance(result, PipelineResult)
        assert len(result.extraction.entities) >= 1
        assert len(result.extraction.claims) >= 1
        assert result.verified_claims_count >= 1
        assert result.synthesized_answer is not None

    @pytest.mark.asyncio
    async def test_pipeline_run_without_synthesis(self, mock_llm, sample_chunks):
        """Test pipeline run without synthesis step."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)
        result = await pipeline.run(sample_chunks, question=None, synthesize=False)

        assert result.synthesized_answer is None
        assert len(result.extraction.entities) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_empty_chunks(self, mock_llm):
        """Test pipeline handles empty chunks gracefully."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)
        result = await pipeline.run([], question="Test?", synthesize=True)

        assert len(result.extraction.entities) == 0
        assert len(result.extraction.claims) == 0
        assert result.synthesized_answer is None

    @pytest.mark.asyncio
    async def test_pipeline_model_configuration(self, mock_llm):
        """Test pipeline uses correct models for each pass."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(
            llm=mock_llm,
            extraction_model="extract-model",
            verification_model="verify-model",
            synthesis_model="synth-model",
        )

        assert pipeline.extraction_model == "extract-model"
        assert pipeline.verification_model == "verify-model"
        assert pipeline.synthesis_model == "synth-model"

    @pytest.mark.asyncio
    async def test_pipeline_default_models(self, mock_llm):
        """Test pipeline uses default models when not specified."""
        from app.services.kb_pipeline import KBPipeline

        pipeline = KBPipeline(llm=mock_llm)

        # Extraction/verification use rerank model, synthesis uses answer model
        assert pipeline.extraction_model == mock_llm.effective_rerank_model
        assert pipeline.verification_model == mock_llm.effective_rerank_model
        assert pipeline.synthesis_model == mock_llm.answer_model


class TestConvenienceFunction:
    """Tests for run_kb_pipeline convenience function."""

    @pytest.mark.asyncio
    async def test_run_kb_pipeline_function(self):
        """Test convenience function works correctly."""
        from app.services.kb_pipeline import run_kb_pipeline

        mock_llm = MockLLMClient()
        chunks = [{"content": "RSI is a momentum indicator", "title": "Test"}]

        result = await run_kb_pipeline(
            chunks=chunks,
            question="What is RSI?",
            synthesize=True,
            llm=mock_llm,
        )

        assert isinstance(result, PipelineResult)
        assert len(result.extraction.entities) >= 0
