# Changelog

All notable changes to the Trading RAG Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Backtest Parameter Tuning System** - Complete research workflow for strategy optimization
  - Grid/random search over strategy parameter spaces
  - IS/OOS (In-Sample/Out-of-Sample) split validation with configurable ratio
  - Composite objective functions: `sharpe`, `sharpe_dd_penalty`, `return`, `return_dd_penalty`, `calmar`
  - Gates policy enforcement (max drawdown, min trades) with audit snapshots
  - Overfit diagnostics via IS-OOS score gap analysis
- **Admin UI for Tuning**
  - `/admin/backtests/tunes` - Filterable list with validity badges
  - `/admin/backtests/leaderboard` - Global ranking by objective score
  - `/admin/backtests/compare` - N-way diff table with highlighting
  - CSV export for leaderboard, JSON export for compare
  - "Compare Selected" multi-select from leaderboard
- **Tuning API Endpoints**
  - `POST /backtests/tune` - Run parameter sweep
  - `GET /backtests/tunes` - List with filters (valid_only, objective_type, oos_enabled)
  - `GET /backtests/leaderboard` - Ranked tunes with best run metrics
  - `GET /backtests/tunes/{id}` - Tune detail with trial list
  - `POST /backtests/tunes/{id}/cancel` - Cancel running tune
- **Database Migrations** (012-016)
  - Tune run failed_reason tracking
  - IS/OOS split columns (score_is, score_oos, oos_ratio)
  - Metrics JSONB persistence (metrics_is, metrics_oos)
  - Composite objective support (objective_type, objective_params, objective_score)
  - Gates JSONB for policy snapshots

### Changed
- **BREAKING**: `OPENROUTER_API_KEY` is now optional (was required)
  - Service starts and runs without LLM configuration
  - `mode=retrieve` (semantic search) always works
  - `mode=answer` returns retrieved chunks with helpful message when LLM not configured
  - Enables graceful degradation and avoids vendor lock-in

### Added
- `LLMNotConfiguredError` exception for clear error handling when LLM is not available
- Graceful degradation for `mode=answer` queries without OpenRouter API key
- Initial release of Trading RAG Pipeline
- Document ingestion endpoint (`POST /ingest`)
- YouTube transcript ingestion (`POST /sources/youtube/ingest`)
- Semantic search with filtering (`POST /query`)
- Re-embedding for model migration (`POST /reembed`)
- Job status tracking (`GET /jobs/{job_id}`)
- Health check with dependency status (`GET /health`)
- Qdrant vector storage with 8 payload indexes
- Ollama embedding with nomic-embed-text model
- Token-aware chunking (512 tokens max)
- Metadata extraction (symbols, entities, topics)
- YouTube transcript normalization
- Timestamp-aware chunking for video content
- OpenAPI documentation (Swagger UI and ReDoc)
- Docker Compose configuration for local development
- Structured JSON logging with request context

### Technical Details
- FastAPI 0.109.2 with Pydantic v2
- Qdrant vector database (768-dim, Cosine distance)
- Supabase for document/chunk storage
- tiktoken for accurate token counting
- Request ID middleware for tracing
- CORS configuration for cross-origin requests

## [0.1.0] - 2025-01-03

### Added
- Initial project structure
- Core API endpoints
- Database schema design
- Docker configuration
- Unit tests for extractor and chunker
- Development requirements separation

---

[Unreleased]: https://github.com/username/trading-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/trading-rag/releases/tag/v0.1.0
