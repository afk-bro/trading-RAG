"""System prompts for knowledge base extraction/verification pipeline.

Three-pass architecture:
- Pass 1 (Extract): Extract entities, claims, and relations with evidence pointers
- Pass 2 (Verify): Validate each claim against source evidence
- Pass 3 (Synthesize): Generate answer using only verified claims
"""

from typing import Any

# ===========================================
# Pass 1: Extraction Prompt (Flash model)
# ===========================================

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction system. Your task is to extract structured knowledge from the provided context chunks.  # noqa: E501

## OUTPUT FORMAT
You must respond with valid JSON matching this exact schema:
```json
{
  "entities": [
    {
      "type": "concept|indicator|strategy|equation|test|metric|asset|pattern|parameter|other",
      "name": "Primary name",
      "aliases": ["Alternative names"],
      "description": "Grounded description from context",
      "evidence": [{"chunk_index": 0, "quote": "verbatim quote", "relevance": 0.9}]
    }
  ],
  "claims": [
    {
      "claim_type": "definition|rule|assumption|warning|parameter|equation|observation|recommendation|other",  # noqa: E501
      "text": "Atomic truth statement",
      "entity_name": "Related entity if applicable",
      "entity_type": "concept|indicator|...",
      "confidence": 0.8,
      "evidence": [{"chunk_index": 0, "quote": "supporting quote", "relevance": 0.9}]
    }
  ],
  "relations": [
    {
      "from_entity": "Entity A name",
      "from_type": "concept|indicator|...",
      "relation": "uses|requires|derived_from|variant_of|contradicts|supports|mentions|component_of|input_to|output_of|precedes|follows",  # noqa: E501
      "to_entity": "Entity B name",
      "to_type": "concept|indicator|...",
      "evidence": [{"chunk_index": 0, "quote": "quote showing relation", "relevance": 0.9}]
    }
  ]
}
```

## CRITICAL RULES
1. **Evidence Required**: Every claim MUST have at least one evidence pointer with a verbatim quote from the chunks.  # noqa: E501
2. **Atomic Claims**: Each claim should be a single, testable statement. Split compound statements.
3. **No Hallucination**: Only extract what is explicitly stated. Do not infer or assume.
4. **Quote Accuracy**: The quote field must be a verbatim excerpt (max 200 chars) that directly supports the claim.  # noqa: E501
5. **Chunk Index**: chunk_index is 0-based, referring to the order chunks are provided.
6. **Confidence**: Initial confidence 0.5-0.9 based on evidence strength. Never 1.0 at extraction.
7. **Entity Types**: Use the most specific type. 'concept' is the catch-all.
8. **Relation Direction**: from_entity → relation → to_entity (e.g., "RSI" uses "price")

## WHAT TO EXTRACT
- **Entities**: Named concepts, indicators, strategies, formulas, metrics, assets
- **Claims**: Definitions, rules, constraints, warnings, parameters, observations
- **Relations**: Dependencies, derivations, contradictions, components

## WHAT NOT TO EXTRACT
- Vague or ambiguous statements without clear meaning
- Opinions without supporting evidence
- Claims that span multiple unconnected ideas
- Duplicate information already captured elsewhere"""


def build_extraction_prompt(chunks: list[dict], question: str | None = None) -> str:
    """Build the user prompt for extraction with context chunks.

    Args:
        chunks: List of chunks with 'content' and optional 'title'/'doc_id'
        question: Optional question to focus extraction (for query-driven mode)

    Returns:
        Formatted user prompt
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        title = chunk.get("title", chunk.get("doc_id", f"Chunk {i}"))
        context_parts.append(f"[{i}] {title}:\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    prompt = f"""## CONTEXT CHUNKS
{context}

## TASK
Extract all entities, claims, and relations from these chunks.
"""

    if question:
        prompt += f"""
## FOCUS QUESTION
Pay special attention to information relevant to: {question}
Prioritize extracting knowledge that helps answer this question.
"""

    prompt += """
## RESPONSE
Return valid JSON with entities, claims, and relations arrays. No markdown code fences."""

    return prompt


# ===========================================
# Pass 2: Verification Prompt (Flash model)
# ===========================================

VERIFICATION_SYSTEM_PROMPT = """You are a claim verification system. Your task is to verify extracted claims against source evidence.  # noqa: E501

## INPUT
You will receive:
1. Original context chunks (the source of truth)
2. Extracted claims with evidence pointers

## OUTPUT FORMAT
Return valid JSON with verdicts for each claim:
```json
{
  "verdicts": [
    {
      "claim_index": 0,
      "status": "verified|weak|rejected",
      "confidence": 0.85,
      "reason": "Brief explanation",
      "corrected_text": "Rewritten claim if needed for accuracy (optional)"
    }
  ]
}
```

## VERIFICATION CRITERIA

### VERIFIED (confidence 0.7-1.0)
- The quoted evidence DIRECTLY supports the claim
- The claim accurately represents what the source says
- No significant information is omitted or distorted

### WEAK (confidence 0.3-0.7)
- The evidence partially supports the claim
- The claim is plausible but not fully substantiated
- Minor inaccuracies or overgeneralizations

### REJECTED (confidence 0.0-0.3)
- The quoted evidence does NOT support the claim
- The claim misrepresents the source
- The claim makes assumptions not in the evidence
- The quote is fabricated or not from the chunks

## CRITICAL RULES
1. **Check the Quote**: Verify the quoted text actually appears in the referenced chunk.
2. **Check the Support**: Does the quote actually support what the claim says?
3. **No External Knowledge**: Judge ONLY based on the provided chunks, not your training data.
4. **Be Strict**: If in doubt, mark as WEAK, not VERIFIED.
5. **Correct When Possible**: If a claim is close but imprecise, provide corrected_text.

## RED FLAGS (likely REJECTED)
- Quote not found in the chunk
- Claim adds information not in the quote
- Claim contradicts the quote
- Multiple claims collapsed into one
- Speculative language presented as fact"""


def build_verification_prompt(
    chunks: list[dict],
    claims: list[dict],
) -> str:
    """Build the user prompt for verification.

    Args:
        chunks: Original context chunks
        claims: Extracted claims to verify

    Returns:
        Formatted user prompt
    """
    # Format chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        title = chunk.get("title", f"Chunk {i}")
        context_parts.append(f"[{i}] {title}:\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    # Format claims
    claims_parts = []
    for i, claim in enumerate(claims):
        evidence_str = ""
        if claim.get("evidence"):
            evidence_str = "\n    Evidence: " + "; ".join(
                f"chunk[{e.get('chunk_index', '?')}]: \"{e.get('quote', '')[:100]}...\""
                for e in claim["evidence"]
            )
        claims_parts.append(
            f"  [{i}] Type: {claim.get('claim_type', 'unknown')}\n"
            f"      Text: {claim.get('text', '')}{evidence_str}"
        )

    claims_text = "\n".join(claims_parts)

    return f"""## SOURCE CHUNKS
{context}

## CLAIMS TO VERIFY
{claims_text}

## TASK
For each claim, verify whether the quoted evidence actually supports it.
Check that quotes exist in the chunks and that they support the claim text.

## RESPONSE
Return valid JSON with verdicts array. No markdown code fences."""


# ===========================================
# Pass 3: Synthesis Prompt (Sonnet/Gemma model)
# ===========================================

SYNTHESIS_SYSTEM_PROMPT = """You are a knowledge synthesis assistant. Your task is to answer questions using ONLY verified claims from a knowledge base.  # noqa: E501

## GROUNDING CONTRACT
- Use ONLY the verified claims provided as your source of truth
- Do NOT use general knowledge, assumptions, or outside facts
- If the verified claims don't answer the question, say so clearly
- Cite claims using [C1], [C2] etc. to reference claim numbers
- Prefer accuracy over completeness - partial answers are OK

## OUTPUT FORMAT
Structure your response as:

**Answer:**
[Your answer based strictly on verified claims, with citations]

**Based on verified claims:**
- [List key claims that support your answer]

**Not addressed by verified claims:**
- [Aspects of the question the claims don't cover, if any]

**Confidence:** [high/medium/low] - based on claim coverage and confidence scores"""


def build_synthesis_prompt(
    question: str,
    verified_claims: list[dict],
    entities: list[dict] | None = None,
) -> str:
    """Build the user prompt for synthesis.

    Args:
        question: User's question
        verified_claims: Claims that passed verification
        entities: Optional relevant entities for context

    Returns:
        Formatted user prompt
    """
    # Format claims
    claims_parts = []
    for i, claim in enumerate(verified_claims):
        confidence = claim.get("confidence", 0.5)
        entity = claim.get("entity_name", "")
        entity_str = f" (about: {entity})" if entity else ""
        claims_parts.append(
            f"[C{i+1}] [{claim.get('claim_type', 'unknown')}]{entity_str}\n"
            f"     {claim.get('text', '')}\n"
            f"     Confidence: {confidence:.0%}"
        )

    claims_text = (
        "\n".join(claims_parts) if claims_parts else "(No verified claims available)"
    )

    # Format entities if provided
    entities_text = ""
    if entities:
        entity_parts = []
        for e in entities:
            desc = (
                f": {e.get('description', '')[:100]}..." if e.get("description") else ""
            )
            entity_parts.append(
                f"- {e.get('name', 'Unknown')} ({e.get('type', 'unknown')}){desc}"
            )
        entities_text = f"""
## RELEVANT ENTITIES
{chr(10).join(entity_parts)}
"""

    return f"""## VERIFIED CLAIMS
{claims_text}
{entities_text}
## QUESTION
{question}

## TASK
Answer the question using ONLY the verified claims above.
Cite specific claims [C1], [C2] etc. when making statements.
If claims don't fully answer the question, acknowledge what's missing."""


# ===========================================
# KB Answer Prompt (for mode=kb_answer)
# ===========================================

KB_ANSWER_SYSTEM_PROMPT = """You are a knowledge base answer assistant. Your task is to answer questions using ONLY verified claims from a truth store.  # noqa: E501

## OUTPUT FORMAT
You must respond with valid JSON matching this schema:
```json
{
  "answer": "Synthesized answer text with [C1], [C2] claim citations",
  "supported": ["Statement 1 [C1]", "Statement 2 [C2]"],
  "not_specified": ["Aspect the claims don't cover"]
}
```

## GROUNDING CONTRACT
1. **Truth-First**: Use ONLY the verified claims provided. These are pre-verified facts.
2. **Cite Everything**: Every factual statement needs a [C<number>] citation.
3. **No Inference**: Don't extrapolate beyond what claims explicitly state.
4. **Acknowledge Gaps**: If claims don't fully answer, list gaps in not_specified.
5. **Be Concise**: Keep the answer focused and direct.

## CITATION FORMAT
- Use [C1], [C2] etc. to reference claim numbers from the provided list
- Place citations immediately after the statement they support
- Multiple claims can support one statement: "X is true [C1, C3]"

## QUALITY RULES
- If fewer than 2-3 relevant claims, acknowledge limited knowledge
- Prefer accuracy over comprehensiveness
- Group related claims logically in the answer
- The 'supported' array should list the key facts with their citations
- The 'not_specified' array should list aspects of the question not covered"""


def build_kb_answer_prompt(
    question: str,
    claims: list[dict],
) -> str:
    """Build the user prompt for kb_answer synthesis.

    Args:
        question: User's question
        claims: Verified claims from truth store with id, text, confidence, entity info

    Returns:
        Formatted user prompt
    """
    # Format claims with reference IDs
    claims_parts = []
    for i, claim in enumerate(claims):
        confidence = claim.get("confidence", 0.5)
        entity = claim.get("entity_name", "")
        claim_type = claim.get("claim_type", "unknown")
        entity_str = f" (about: {entity})" if entity else ""
        claims_parts.append(
            f"[C{i+1}] [{claim_type}]{entity_str}\n"
            f"     {claim.get('text', '')}\n"
            f"     Confidence: {confidence:.0%}"
        )

    claims_text = (
        "\n".join(claims_parts) if claims_parts else "(No verified claims available)"
    )

    return f"""## VERIFIED CLAIMS FROM TRUTH STORE
{claims_text}

## QUESTION
{question}

## TASK
Answer the question using ONLY the verified claims above.
Cite specific claims [C1], [C2] etc. when making statements.

## RESPONSE
Return valid JSON with answer, supported, and not_specified fields. No markdown code fences."""


# ===========================================
# JSON Parsing Helpers
# ===========================================


def extract_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code fences.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as dict

    Raises:
        ValueError: If no valid JSON found
    """
    import json
    import re

    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fence
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        r"\{[\s\S]*\}",  # Raw JSON object
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            json_str = match.group(1) if "```" in pattern else match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")
