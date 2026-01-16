# Pine Script Read APIs Design

**Date**: 2026-01-15
**Status**: Approved
**Author**: Claude + User

## Overview

Add admin-only read APIs for inspecting indexed Pine scripts:

- `GET /sources/pine/scripts` - List indexed scripts with filtering
- `GET /sources/pine/scripts/{doc_id}` - Get script details

These endpoints provide visibility into what's indexed, enable debugging, and lay groundwork for future UI.

## Design Decisions

### Input Method
**Decision**: Query documents table, not registry files
**Rationale**: Database is source of truth for what's actually indexed

### Filtering Strategy
**Decision**: Symbol filtering only (Option 2)
**Rationale**:
- Ticker-centric workflows are constant in trading ("show me strategies for BTC, SPX")
- Leverages existing GIN index on `chunks.symbols`
- No schema changes for filtering itself
- Rich filtering (script_type, lint_errors) can come later via JSONB queries

### Identifier for Detail Endpoint
**Decision**: Document UUID
**Rationale**: Standard REST pattern, stable across renames, list provides mapping

### Metadata Storage
**Decision**: Add `pine_metadata` JSONB column to documents table
**Rationale**:
- Structured metadata exists at ingest time (registry entry + lint report)
- Storing once prevents fragile markdown parsing
- Enables future filtering without schema redesign
- Single nullable column, minimal migration

## API Specification

### List Endpoint

```
GET /sources/pine/scripts
```

**Authentication**: `X-Admin-Token` header required

**Query Parameters**:

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `workspace_id` | UUID | Yes | - | Tenant isolation |
| `symbol` | string | No | - | Filter by ticker symbol |
| `status` | enum | No | `active` | `active`, `superseded`, `deleted`, `all` |
| `q` | string | No | - | Free-text search in title/path |
| `order_by` | enum | No | `updated_at` | `updated_at`, `created_at`, `title` |
| `order_dir` | enum | No | `desc` | `asc`, `desc` |
| `limit` | int | No | 20 | 1-100 |
| `offset` | int | No | 0 | Pagination offset |

**Response** (`PineScriptListResponse`):

```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "canonical_url": "pine://local/strategies/breakout.pine",
      "rel_path": "strategies/breakout.pine",
      "title": "Breakout Strategy",
      "script_type": "strategy",
      "pine_version": "5",
      "symbols": ["SPX", "AAPL"],
      "lint_summary": {"errors": 0, "warnings": 1, "info": 0},
      "lint_available": true,
      "sha256": "abc123...",
      "chunk_count": 3,
      "created_at": "2026-01-15T12:00:00Z",
      "updated_at": "2026-01-15T12:00:00Z",
      "status": "active"
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0,
  "has_more": true,
  "next_offset": 20
}
```

### Detail Endpoint

```
GET /sources/pine/scripts/{doc_id}
```

**Authentication**: `X-Admin-Token` header required

**Path Parameters**:

| Param | Type | Description |
|-------|------|-------------|
| `doc_id` | UUID | Document ID |

**Query Parameters**:

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `workspace_id` | UUID | Yes | - | Tenant isolation (must match doc) |
| `include_chunks` | bool | No | false | Include chunk content |
| `chunk_limit` | int | No | 50 | Max chunks to return |
| `chunk_offset` | int | No | 0 | Chunk pagination offset |
| `include_lint_findings` | bool | No | false | Include lint details |

**Response** (`PineScriptDetailResponse`):

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "canonical_url": "pine://local/strategies/breakout.pine",
  "rel_path": "strategies/breakout.pine",
  "title": "Breakout Strategy",
  "script_type": "strategy",
  "pine_version": "5",
  "symbols": ["SPX", "AAPL"],
  "lint_summary": {"errors": 0, "warnings": 1, "info": 0},
  "lint_available": true,
  "lint_findings": [
    {
      "code": "W002",
      "severity": "warning",
      "message": "lookahead=barmerge.lookahead_on detected",
      "line": 45,
      "column": 10
    }
  ],
  "sha256": "abc123...",
  "created_at": "2026-01-15T12:00:00Z",
  "updated_at": "2026-01-15T12:00:00Z",
  "status": "active",
  "inputs": [
    {
      "name": "lookback_days",
      "type": "int",
      "default": "52",
      "tooltip": "Days to look back"
    }
  ],
  "imports": [
    {"path": "lib/utils.pine", "alias": "utils"}
  ],
  "features": {
    "uses_request_security": true,
    "uses_lookahead_on": false
  },
  "chunk_total": 3,
  "chunks": [
    {
      "id": "chunk-uuid",
      "index": 0,
      "content": "# Breakout Strategy...",
      "token_count": 512,
      "symbols": ["SPX"]
    }
  ],
  "chunk_has_more": false,
  "chunk_next_offset": null
}
```

**Error Responses**:

| Code | Condition |
|------|-----------|
| 400 | Invalid UUID, limit out of range |
| 401 | Missing admin token |
| 403 | Invalid token or workspace mismatch |
| 404 | Document not found |

## Data Model

### Schema Migration

```sql
-- migrations/017_add_pine_metadata.sql
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS pine_metadata JSONB;

COMMENT ON COLUMN documents.pine_metadata IS
  'Structured metadata for Pine scripts (script_type, inputs, features, lint)';
```

### pine_metadata Shape

```json
{
  "schema_version": "pine_meta_v1",
  "script_type": "strategy",
  "pine_version": "5",
  "rel_path": "strategies/breakout.pine",
  "inputs": [
    {"name": "lookback_days", "type": "int", "default": "52", "tooltip": "..."}
  ],
  "imports": [
    {"path": "lib/utils.pine", "alias": "utils"}
  ],
  "features": {
    "uses_request_security": true,
    "uses_lookahead_on": false
  },
  "lint_summary": {"errors": 0, "warnings": 1, "info": 0},
  "lint_available": true,
  "lint_findings": [
    {"code": "W002", "severity": "warning", "message": "...", "line": 45}
  ]
}
```

**Constraints**:
- `lint_findings` capped at 200 entries to prevent row bloat
- `schema_version` enables future migrations

## Response Schema Types

```python
from typing import Literal
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

ScriptType = Literal["indicator", "strategy", "library"]
PineVersion = Literal["4", "5", "6"]
DocStatus = Literal["active", "superseded", "deleted"]

class LintSummary(BaseModel):
    errors: int = 0
    warnings: int = 0
    info: int = 0

class LintFinding(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    line: int | None = None
    column: int | None = None

class PineInput(BaseModel):
    name: str
    type: str
    default: str | None = None
    tooltip: str | None = None

class PineImport(BaseModel):
    path: str
    alias: str | None = None

class ChunkItem(BaseModel):
    id: UUID
    index: int
    content: str
    token_count: int
    symbols: list[str]

class PineScriptListItem(BaseModel):
    id: UUID
    canonical_url: str
    rel_path: str
    title: str  # Guaranteed fallback to basename(rel_path)
    script_type: ScriptType | None
    pine_version: PineVersion | None
    symbols: list[str]
    lint_summary: LintSummary
    lint_available: bool
    sha256: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    status: DocStatus

class PineScriptListResponse(BaseModel):
    items: list[PineScriptListItem]
    total: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None

class PineScriptDetailResponse(BaseModel):
    id: UUID
    canonical_url: str
    rel_path: str
    title: str
    script_type: ScriptType | None
    pine_version: PineVersion | None
    symbols: list[str]
    lint_summary: LintSummary
    lint_available: bool
    lint_findings: list[LintFinding] | None
    sha256: str
    created_at: datetime
    updated_at: datetime
    status: DocStatus
    inputs: list[PineInput] | None
    imports: list[PineImport] | None
    features: dict[str, bool] | None
    chunk_total: int
    chunks: list[ChunkItem] | None
    chunk_has_more: bool | None
    chunk_next_offset: int | None
```

## Implementation Plan

### Files to Create/Modify

| File | Change |
|------|--------|
| `migrations/017_add_pine_metadata.sql` | NEW - Add JSONB column |
| `app/services/pine/ingest.py` | Populate `pine_metadata` on document insert |
| `app/routers/pine.py` | Add list/detail endpoints |
| `app/schemas.py` | Add response models |
| `tests/unit/pine/test_pine_read_api.py` | NEW - Unit tests |

### Query Patterns

**List Query** (aggregates at document level):

```sql
WITH script_symbols AS (
    SELECT
        c.doc_id,
        array_agg(DISTINCT s) FILTER (WHERE s IS NOT NULL) as symbols
    FROM chunks c
    CROSS JOIN LATERAL unnest(c.symbols) AS s
    WHERE c.workspace_id = $1
    GROUP BY c.doc_id
)
SELECT
    d.id, d.canonical_url, d.title, d.status,
    d.pine_metadata, d.content_hash,
    d.created_at, d.updated_at,
    (SELECT COUNT(*) FROM chunks WHERE doc_id = d.id) as chunk_count,
    COALESCE(ss.symbols, '{}') as symbols
FROM documents d
LEFT JOIN script_symbols ss ON ss.doc_id = d.id
WHERE d.workspace_id = $1
  AND d.source_type = 'pine_script'
  AND ($2::text IS NULL OR d.status = $2)
  AND ($3::text IS NULL OR $3 = ANY(ss.symbols))
  AND ($4::text IS NULL OR d.title ILIKE '%' || $4 || '%'
       OR d.canonical_url ILIKE '%' || $4 || '%')
ORDER BY
    CASE WHEN $5 = 'updated_at' AND $6 = 'desc' THEN d.updated_at END DESC,
    CASE WHEN $5 = 'updated_at' AND $6 = 'asc' THEN d.updated_at END ASC,
    CASE WHEN $5 = 'created_at' AND $6 = 'desc' THEN d.created_at END DESC,
    CASE WHEN $5 = 'created_at' AND $6 = 'asc' THEN d.created_at END ASC,
    CASE WHEN $5 = 'title' AND $6 = 'desc' THEN d.title END DESC,
    CASE WHEN $5 = 'title' AND $6 = 'asc' THEN d.title END ASC
LIMIT $7 OFFSET $8
```

**Detail Query**:

```sql
-- Document
SELECT
    d.id, d.canonical_url, d.title, d.status,
    d.pine_metadata, d.content_hash,
    d.created_at, d.updated_at,
    d.workspace_id
FROM documents d
WHERE d.id = $1 AND d.source_type = 'pine_script'

-- Symbols (aggregated)
SELECT array_agg(DISTINCT s) FILTER (WHERE s IS NOT NULL)
FROM chunks c
CROSS JOIN LATERAL unnest(c.symbols) AS s
WHERE c.doc_id = $1

-- Chunks (paginated)
SELECT id, chunk_index, content, token_count, symbols
FROM chunks
WHERE doc_id = $1
ORDER BY chunk_index
LIMIT $2 OFFSET $3

-- Chunk total
SELECT COUNT(*) FROM chunks WHERE doc_id = $1
```

### Backfill Strategy

**Not required for initial ship**. New ingests populate `pine_metadata`. Existing docs return `lint_available=false` and `None` for structured fields.

**Optional admin command** (post-ship):

```bash
# Re-ingest Pine scripts to backfill metadata
curl -X POST http://localhost:8000/sources/pine/ingest \
  -H "X-Admin-Token: $TOKEN" \
  -d '{
    "workspace_id": "...",
    "registry_path": "/data/pine/pine_registry.json",
    "update_existing": true
  }'
```

## Testing Strategy

### Unit Tests

- Request validation (Pydantic)
- Query parameter parsing
- Response serialization
- Workspace isolation (403 on mismatch)
- Pagination math (has_more, next_offset)

### Integration Tests

- List with symbol filter
- List with free-text search
- Detail with/without chunks
- Detail with lint findings
- Empty results handling
- Large pagination

## Future Enhancements (Out of Scope)

- GIN index on `pine_metadata` for rich filtering
- `script_type` and `lint_errors` filters
- Bulk export endpoint
- Admin UI for Pine scripts
