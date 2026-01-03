# Trading RAG Pipeline - Finance Knowledge Base

A local RAG (Retrieval-Augmented Generation) pipeline for finance and trading knowledge. The system ingests YouTube transcripts and other documents via n8n orchestration, processes them through a FastAPI service with chunking, embedding, and storage capabilities.

## Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Google    │────▶│      n8n        │────▶│   FastAPI       │
│   Sheets    │     │   Orchestrator  │     │   Service       │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                    ┌────────────────────────────────┼───────────────────────────────┐
                    │                                │                               │
                    ▼                                ▼                               ▼
            ┌─────────────┐                 ┌─────────────┐                 ┌─────────────┐
            │   Ollama    │                 │   Qdrant    │                 │  Supabase   │
            │ (Embeddings)│                 │  (Vectors)  │                 │  (Postgres) │
            └─────────────┘                 └─────────────┘                 └─────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | n8n (Docker) | Watch queue, call service, manage state |
| Backend | Python FastAPI | API endpoints, business logic |
| Primary DB | Supabase Postgres | Documents, chunks, metadata (source of truth) |
| Vector DB | Qdrant (Docker) | Embedding vectors, similarity search |
| Embeddings | Ollama (local) | nomic-embed-text (768 dimensions) |
| LLM | OpenRouter API | Answer generation (Claude Sonnet 4) |

## Features

- **YouTube Ingestion**: Parse URLs, fetch transcripts, extract metadata
- **Document Ingestion**: Support for PDF, article, note, transcript sources
- **Smart Chunking**: Token-aware (~512 tokens), timestamp preservation
- **Metadata Extraction**: Symbols, entities, topics, speakers
- **Semantic Search**: Qdrant vector search with payload filtering
- **Answer Generation**: LLM synthesis with citations
- **Model Migration**: Re-embed support for model upgrades

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Supabase project with Postgres database
- OpenRouter API key
- n8n instance (optional, for automated ingestion)

### Setup

1. Clone and navigate to the project:
   ```bash
   cd trading-RAG
   ```

2. Run the setup script:
   ```bash
   ./init.sh
   ```

3. Configure environment variables in `.env`:
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   OPENROUTER_API_KEY=your-openrouter-api-key
   ```

4. Access the services:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Qdrant Dashboard: http://localhost:6333/dashboard

## API Endpoints

### Health Check
```http
GET /health
```
Returns service status and dependency health.

### Ingest Document
```http
POST /ingest
Content-Type: application/json

{
  "workspace_id": "uuid",
  "source": {
    "url": "https://example.com/article",
    "type": "article"
  },
  "content": "Document content...",
  "metadata": {
    "title": "Article Title",
    "author": "Author Name"
  }
}
```

### Ingest YouTube
```http
POST /sources/youtube/ingest
Content-Type: application/json

{
  "workspace_id": "uuid",
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Query
```http
POST /query
Content-Type: application/json

{
  "workspace_id": "uuid",
  "question": "What is the Fed's current stance on interest rates?",
  "mode": "answer",
  "filters": {
    "source_types": ["youtube"],
    "symbols": ["SPY"],
    "topics": ["macro"]
  },
  "top_k": 5
}
```

### Re-embed
```http
POST /reembed
Content-Type: application/json

{
  "workspace_id": "uuid",
  "target_collection": "kb_new_model_v1",
  "embed_provider": "ollama",
  "embed_model": "new-model-name"
}
```

### Job Status
```http
GET /jobs/{job_id}
```

## Project Structure

```
trading-RAG/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── schemas.py           # Pydantic models
│   ├── routers/
│   │   ├── health.py
│   │   ├── ingest.py
│   │   ├── youtube.py
│   │   ├── query.py
│   │   ├── reembed.py
│   │   └── jobs.py
│   ├── services/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── extractor.py
│   │   └── llm.py
│   └── repositories/
│       ├── documents.py
│       ├── chunks.py
│       └── vectors.py
├── tests/
│   ├── unit/
│   └── integration/
├── migrations/
│   └── 001_initial_schema.sql
├── docker-compose.rag.yml
├── Dockerfile
├── requirements.txt
├── init.sh
├── feature_list.json
└── README.md
```

## Database Schema

### documents
- Core document metadata
- Unique constraint on (workspace_id, source_type, canonical_url)
- Status: active, superseded, deleted

### chunks
- Content segments with token counts
- Timestamp tracking for YouTube
- Page tracking for PDFs
- Metadata arrays: symbols, entities, topics

### chunk_vectors
- Tracks embeddings per model/collection
- Supports model migration workflows

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Supabase project URL | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Yes |
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `QDRANT_HOST` | Qdrant host (default: qdrant) | No |
| `QDRANT_PORT` | Qdrant port (default: 6333) | No |
| `OLLAMA_HOST` | Ollama host (default: ollama) | No |
| `OLLAMA_PORT` | Ollama port (default: 11434) | No |
| `EMBED_MODEL` | Embedding model (default: nomic-embed-text) | No |
| `SERVICE_PORT` | Service port (default: 8000) | No |

## Development

### Run locally
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### Run tests
```bash
pytest tests/
```

### Run with Docker
```bash
docker compose -f docker-compose.rag.yml up --build
```

## License

MIT License - See LICENSE for details.
