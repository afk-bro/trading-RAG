"""Knowledge base extraction → verification → synthesis pipeline.

Three-pass architecture:
- Pass 1 (Extract): Extract entities, claims, relations with evidence pointers
- Pass 2 (Verify): Validate each claim against source evidence
- Pass 3 (Synthesize): Generate answer using only verified claims
"""

from typing import Optional
from uuid import UUID

import structlog

from app.services.llm_base import BaseLLMClient, LLMNotConfiguredError
from app.services.llm_factory import get_llm

from app.services.kb_types import (
    ExtractionResult,
    VerificationResult,
    PipelineResult,
    ExtractedEntity,
    ExtractedClaim,
    ExtractedRelation,
    ClaimVerdict,
    EvidencePointer,
    EntityType,
    ClaimType,
    RelationType,
    VerificationStatus,
)
from app.services.kb_prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_verification_prompt,
    build_synthesis_prompt,
    extract_json_from_response,
)

logger = structlog.get_logger(__name__)

# Token limits per pass to prevent runaway JSON
MAX_TOKENS_EXTRACTION = 4000  # Extraction can be verbose with many entities
MAX_TOKENS_VERIFICATION = 2000  # Verification is more constrained
MAX_TOKENS_SYNTHESIS = 2000  # Synthesis is answer-focused


class KBPipelineError(Exception):
    """Error during knowledge base pipeline execution."""

    pass


class KBPipeline:
    """Knowledge extraction and verification pipeline.

    Orchestrates the three-pass architecture:
    1. Extract entities, claims, relations from chunks
    2. Verify claims against source evidence
    3. (Optional) Synthesize answer from verified claims
    """

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        extraction_model: str | None = None,
        verification_model: str | None = None,
        synthesis_model: str | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            llm: LLM client (defaults to singleton from factory)
            extraction_model: Model for extraction pass (defaults to rerank_model)
            verification_model: Model for verification pass (defaults to rerank_model)
            synthesis_model: Model for synthesis pass (defaults to answer_model)
        """
        self.llm = llm or get_llm()
        if not self.llm:
            raise LLMNotConfiguredError("KB Pipeline requires an LLM client")

        # Model selection: extraction/verification use cheaper model, synthesis uses answer model
        self.extraction_model = extraction_model or self.llm.effective_rerank_model
        self.verification_model = verification_model or self.llm.effective_rerank_model
        self.synthesis_model = synthesis_model or self.llm.answer_model

        logger.info(
            "KB Pipeline initialized",
            extraction_model=self.extraction_model,
            verification_model=self.verification_model,
            synthesis_model=self.synthesis_model,
        )

    async def extract(
        self,
        chunks: list[dict],
        question: str | None = None,
    ) -> ExtractionResult:
        """
        Pass 1: Extract entities, claims, and relations from chunks.

        Args:
            chunks: Context chunks with 'content' and optional 'title'/'doc_id'
            question: Optional question to focus extraction

        Returns:
            ExtractionResult with entities, claims, relations
        """
        if not chunks:
            return ExtractionResult()

        prompt = build_extraction_prompt(chunks, question)

        logger.info(
            "Starting extraction pass",
            num_chunks=len(chunks),
            has_question=bool(question),
            model=self.extraction_model,
        )

        response = await self.llm.generate(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.extraction_model,
            max_tokens=MAX_TOKENS_EXTRACTION,
        )

        # Parse JSON response
        try:
            data = extract_json_from_response(response.text)
        except ValueError as e:
            logger.error(
                "Failed to parse extraction response",
                error=str(e),
                response_preview=response.text[:500],
            )
            # Return empty result with error marker
            return ExtractionResult()  # Caller should check if empty

        # Convert to typed models
        entities = []
        for e in data.get("entities", []):
            try:
                entities.append(ExtractedEntity(
                    type=EntityType(e.get("type", "other")),
                    name=e.get("name", "Unknown"),
                    aliases=e.get("aliases", []),
                    description=e.get("description"),
                    evidence=[
                        EvidencePointer(
                            chunk_index=ev.get("chunk_index", 0),
                            quote=ev.get("quote", "")[:200],
                            relevance=ev.get("relevance", 1.0),
                        )
                        for ev in e.get("evidence", [])
                    ],
                ))
            except (ValueError, KeyError) as err:
                logger.warning("Skipping invalid entity", error=str(err), entity=e)

        claims = []
        for c in data.get("claims", []):
            try:
                evidence = [
                    EvidencePointer(
                        chunk_index=ev.get("chunk_index", 0),
                        quote=ev.get("quote", "")[:200],
                        relevance=ev.get("relevance", 1.0),
                    )
                    for ev in c.get("evidence", [])
                ]
                if not evidence:
                    logger.warning("Skipping claim without evidence", claim=c.get("text", ""))
                    continue

                entity_type = None
                if c.get("entity_type"):
                    try:
                        entity_type = EntityType(c["entity_type"])
                    except ValueError:
                        pass

                claims.append(ExtractedClaim(
                    claim_type=ClaimType(c.get("claim_type", "other")),
                    text=c.get("text", ""),
                    entity_name=c.get("entity_name"),
                    entity_type=entity_type,
                    confidence=c.get("confidence", 0.5),
                    evidence=evidence,
                ))
            except (ValueError, KeyError) as err:
                logger.warning("Skipping invalid claim", error=str(err), claim=c)

        relations = []
        for r in data.get("relations", []):
            try:
                relations.append(ExtractedRelation(
                    from_entity=r.get("from_entity", ""),
                    from_type=EntityType(r.get("from_type", "other")),
                    relation=RelationType(r.get("relation", "mentions")),
                    to_entity=r.get("to_entity", ""),
                    to_type=EntityType(r.get("to_type", "other")),
                    evidence=[
                        EvidencePointer(
                            chunk_index=ev.get("chunk_index", 0),
                            quote=ev.get("quote", "")[:200],
                            relevance=ev.get("relevance", 1.0),
                        )
                        for ev in r.get("evidence", [])
                    ],
                ))
            except (ValueError, KeyError) as err:
                logger.warning("Skipping invalid relation", error=str(err), relation=r)

        result = ExtractionResult(
            entities=entities,
            claims=claims,
            relations=relations,
        )

        logger.info(
            "Extraction complete",
            entities=len(entities),
            claims=len(claims),
            relations=len(relations),
        )

        return result

    async def verify(
        self,
        chunks: list[dict],
        extraction: ExtractionResult,
    ) -> VerificationResult:
        """
        Pass 2: Verify extracted claims against source evidence.

        Args:
            chunks: Original context chunks
            extraction: Results from extraction pass

        Returns:
            VerificationResult with verdicts for each claim
        """
        if not extraction.claims:
            return VerificationResult()

        # Convert claims to dicts for prompt building
        claims_dicts = [
            {
                "claim_type": c.claim_type.value,
                "text": c.text,
                "evidence": [
                    {"chunk_index": e.chunk_index, "quote": e.quote}
                    for e in c.evidence
                ],
            }
            for c in extraction.claims
        ]

        prompt = build_verification_prompt(chunks, claims_dicts)

        logger.info(
            "Starting verification pass",
            num_claims=len(extraction.claims),
            model=self.verification_model,
        )

        response = await self.llm.generate(
            messages=[
                {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.verification_model,
            max_tokens=MAX_TOKENS_VERIFICATION,
        )

        # Parse JSON response
        try:
            data = extract_json_from_response(response.text)
        except ValueError as e:
            logger.error("Failed to parse verification response", error=str(e))
            # Fallback: mark all as pending
            return VerificationResult(
                verdicts=[
                    ClaimVerdict(
                        claim_index=i,
                        status=VerificationStatus.PENDING,
                        confidence=0.5,
                        reason="Verification failed to parse",
                    )
                    for i in range(len(extraction.claims))
                ]
            )

        # Convert to typed models
        verdicts = []
        for v in data.get("verdicts", []):
            try:
                verdicts.append(ClaimVerdict(
                    claim_index=v.get("claim_index", 0),
                    status=VerificationStatus(v.get("status", "pending")),
                    confidence=v.get("confidence", 0.5),
                    reason=v.get("reason", ""),
                    corrected_text=v.get("corrected_text"),
                ))
            except (ValueError, KeyError) as err:
                logger.warning("Skipping invalid verdict", error=str(err), verdict=v)

        result = VerificationResult(verdicts=verdicts)

        # Count verdicts by status
        status_counts = {}
        for v in verdicts:
            status_counts[v.status.value] = status_counts.get(v.status.value, 0) + 1

        logger.info(
            "Verification complete",
            total_verdicts=len(verdicts),
            status_counts=status_counts,
        )

        return result

    async def synthesize(
        self,
        question: str,
        extraction: ExtractionResult,
        verification: VerificationResult,
    ) -> str | None:
        """
        Pass 3: Synthesize answer from verified claims.

        Args:
            question: User's question
            extraction: Extraction results
            verification: Verification results

        Returns:
            Synthesized answer text, or None if no verified claims
        """
        # Build verdict lookup
        verdict_map = {v.claim_index: v for v in verification.verdicts}

        # Filter to verified/weak claims
        verified_claims = []
        for i, claim in enumerate(extraction.claims):
            verdict = verdict_map.get(i)
            if verdict and verdict.status in (
                VerificationStatus.VERIFIED,
                VerificationStatus.WEAK,
            ):
                # Use corrected text if available
                claim_dict = {
                    "claim_type": claim.claim_type.value,
                    "text": verdict.corrected_text or claim.text,
                    "entity_name": claim.entity_name,
                    "confidence": verdict.confidence,
                }
                verified_claims.append(claim_dict)

        if not verified_claims:
            logger.info("No verified claims for synthesis")
            return None

        # Build entity list for context
        entities = [
            {
                "name": e.name,
                "type": e.type.value,
                "description": e.description,
            }
            for e in extraction.entities
        ]

        prompt = build_synthesis_prompt(question, verified_claims, entities)

        logger.info(
            "Starting synthesis pass",
            num_verified_claims=len(verified_claims),
            model=self.synthesis_model,
        )

        response = await self.llm.generate(
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.synthesis_model,
            max_tokens=MAX_TOKENS_SYNTHESIS,
        )

        logger.info("Synthesis complete", response_length=len(response.text))

        return response.text

    async def run(
        self,
        chunks: list[dict],
        question: str | None = None,
        synthesize: bool = True,
    ) -> PipelineResult:
        """
        Run the full extraction → verification → synthesis pipeline.

        Args:
            chunks: Context chunks to process
            question: Optional question (for focused extraction and synthesis)
            synthesize: Whether to run synthesis pass (default True)

        Returns:
            PipelineResult with all outputs and statistics
        """
        logger.info(
            "Starting KB pipeline",
            num_chunks=len(chunks),
            has_question=bool(question),
            will_synthesize=synthesize,
        )

        parse_errors = []
        had_extraction_error = False
        had_verification_error = False

        # Pass 1: Extract
        extraction = await self.extract(chunks, question)

        # Check for extraction failure (empty result suggests parse error)
        if chunks and not extraction.entities and not extraction.claims:
            had_extraction_error = True
            parse_errors.append("Extraction returned no entities or claims (possible JSON parse error)")

        # Pass 2: Verify
        verification = await self.verify(chunks, extraction)

        # Check for verification failure (all pending suggests parse error)
        if extraction.claims and all(
            v.status == VerificationStatus.PENDING and v.reason == "Verification failed to parse"
            for v in verification.verdicts
        ):
            had_verification_error = True
            parse_errors.append("Verification failed to parse (all claims marked pending)")

        # Count verdicts
        verified_count = 0
        weak_count = 0
        rejected_count = 0
        for v in verification.verdicts:
            if v.status == VerificationStatus.VERIFIED:
                verified_count += 1
            elif v.status == VerificationStatus.WEAK:
                weak_count += 1
            elif v.status == VerificationStatus.REJECTED:
                rejected_count += 1

        # Pass 3: Synthesize (optional)
        synthesized_answer = None
        if synthesize and question:
            synthesized_answer = await self.synthesize(question, extraction, verification)

        from app.services.kb_types import PersistenceStats

        result = PipelineResult(
            extraction=extraction,
            verification=verification,
            persistence=PersistenceStats(),  # Empty until persist() is called
            synthesized_answer=synthesized_answer,
            verified_claims_count=verified_count,
            weak_claims_count=weak_count,
            rejected_claims_count=rejected_count,
            parse_errors=parse_errors,
            had_extraction_error=had_extraction_error,
            had_verification_error=had_verification_error,
        )

        logger.info(
            "KB pipeline complete",
            entities=len(extraction.entities),
            claims=len(extraction.claims),
            relations=len(extraction.relations),
            verified=verified_count,
            weak=weak_count,
            rejected=rejected_count,
            has_answer=bool(synthesized_answer),
        )

        return result


async def run_kb_pipeline(
    chunks: list[dict],
    question: str | None = None,
    synthesize: bool = True,
    llm: BaseLLMClient | None = None,
) -> PipelineResult:
    """
    Convenience function to run the KB pipeline.

    Args:
        chunks: Context chunks to process
        question: Optional question
        synthesize: Whether to synthesize answer
        llm: Optional LLM client (uses singleton if not provided)

    Returns:
        PipelineResult with extraction, verification, and optional synthesis
    """
    pipeline = KBPipeline(llm=llm)
    return await pipeline.run(chunks, question, synthesize)
