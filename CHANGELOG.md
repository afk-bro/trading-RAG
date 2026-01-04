# Changelog

All notable changes to the Trading RAG Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
