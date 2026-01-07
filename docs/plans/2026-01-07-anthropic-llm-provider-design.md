# Anthropic LLM Provider Design

**Date:** 2026-01-07
**Status:** Approved
**Author:** Claude + User collaboration

## Overview

Add direct Anthropic API support for LLM answer generation, replacing the OpenRouter-only implementation with a provider-agnostic architecture that auto-detects based on configured API keys.

## Goals

- Use Anthropic SDK directly (no OpenRouter middleman)
- Two-tier model support: Sonnet for answers, Haiku for reranking
- Backward compatible with existing OpenRouter setup
- Graceful degradation when no LLM configured

## Configuration

### New Settings

```python
# LLM Provider Selection
llm_provider: Literal["auto", "anthropic", "openrouter"] = "auto"
llm_required: bool = False  # If true, fail startup when no LLM key
llm_enabled: bool = True    # Kill switch for operations

# Provider API Keys
anthropic_api_key: Optional[str] = None
openrouter_api_key: Optional[str] = None  # Fallback

# Model Configuration
answer_model: str = "claude-sonnet-4"
rerank_model: Optional[str] = "claude-haiku-3-5"  # Set null to reuse answer_model
```

### Provider Resolution

```
if llm_enabled == false → disabled
elif llm_provider == "anthropic" → require anthropic_api_key
elif llm_provider == "openrouter" → require openrouter_api_key
elif llm_provider == "auto":
    if anthropic_api_key → use anthropic
    elif openrouter_api_key → use openrouter
    else:
        if llm_required → raise StartupError
        else → disabled
```

### Model Assignment

| Task | Model |
|------|-------|
| Answer synthesis | `answer_model` |
| Reranking/scoring | `effective_rerank_model` (rerank_model or answer_model) |

### Startup Logging

```
LLM config:
  provider_config: auto
  provider_resolved: anthropic
  answer_model: claude-sonnet-4
  rerank_model_effective: claude-haiku-3-5
  llm_enabled: true
```

## Architecture

### File Structure

```
app/services/
├── llm_base.py         # BaseLLMClient, LLMResponse, RankedChunk, LLMError
├── llm_anthropic.py    # Direct Anthropic SDK implementation
├── llm_openrouter.py   # OpenRouter implementation (extracted)
└── llm_factory.py      # Provider resolution, singleton, LLMStatus
```

### Types

```python
Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: dict | None = None  # {input_tokens, output_tokens, cache_*}
    latency_ms: float | None = None

@dataclass
class RankedChunk:
    chunk: dict
    score: float
    rank: int
    original_index: int  # For stable tiebreaker

class LLMError(Exception):
    provider: str
    model: str | None
```

### Base Client Interface

```python
class BaseLLMClient(ABC):
    answer_model: str
    effective_rerank_model: str  # property

    async def generate(self, *, messages: list[Message],
                       model: str | None = None,
                       max_tokens: int = 2000) -> LLMResponse

    async def generate_text(self, prompt: str, system: str | None = None,
                            model: str | None = None,
                            max_tokens: int = 2000) -> str

    async def generate_answer(self, question: str,
                              chunks: list[dict]) -> LLMResponse

    async def rerank(self, query: str, chunks: list[dict],
                     top_k: int = 5) -> list[RankedChunk]
```

### Factory

```python
@dataclass
class LLMStatus:
    enabled: bool
    provider_config: Literal["auto", "anthropic", "openrouter"]
    provider_resolved: Literal["anthropic", "openrouter"] | None
    answer_model: str | None
    rerank_model_effective: str | None

def get_llm_status() -> LLMStatus
def get_llm() -> BaseLLMClient | None
```

### Degraded Mode Response

When `llm_enabled=false` or no provider configured:

```json
{
  "answer": null,
  "provider": null,
  "llm_enabled": false,
  "message": "LLM not configured - returning retrieved chunks only",
  "chunks": [...]
}
```

## Anthropic Implementation Details

- Uses `AsyncAnthropic` from `anthropic` SDK
- Separates system messages (concatenates if multiple)
- Extracts text from all content blocks (handles multi-block responses)
- Captures cache token usage if present
- Wraps `APIError` in `LLMError` for graceful degradation

## Testing

### Unit Tests

- Factory/provider resolution (all combinations)
- Model fallback (rerank_model=None → answer_model)
- Rank stability (same scores preserve original order)
- Logging (once on instantiation, not per-request)
- Message formatting (system separation, multi-system concat)
- Response parsing (multi-block, empty content)
- Error wrapping

### Integration Tests

- Marked `@pytest.mark.integration` + `@pytest.mark.requires_llm`
- Auto-skip if key missing
- Small max_tokens (32-64) for speed
- Verify response structure

## Rollout Plan

1. **Config only** - Add fields to `config.py` + `.env.example`
2. **Base + factory** - Create `llm_base.py`, `llm_factory.py`
3. **Anthropic client** - Create `llm_anthropic.py` + unit tests
4. **Extract OpenRouter** - Create `llm_openrouter.py` + unit tests
5. **Wire router** - Update `query.py` to use factory
6. **Startup logging** - Log once in `main.py`

## Dependencies

```
anthropic>=0.40.0
```

## Backward Compatibility

- Accept existing env var names via Pydantic aliases
- OpenRouter path remains functional
- No breaking changes to API response shape
