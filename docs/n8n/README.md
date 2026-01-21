# n8n Workflow Integration

This document describes the expected n8n workflow patterns for integrating with the trading-RAG API.

## Overview

The trading-RAG service provides HTTP APIs for document ingestion, search, and answer generation. n8n workflows orchestrate these APIs for automated pipelines (e.g., YouTube channel monitoring, scheduled ingestion).

## Required Environment Variables

```bash
# Trading RAG Service
TRADING_RAG_URL=http://localhost:8000
TRADING_RAG_WORKSPACE_ID=<uuid>

# YouTube Data API (for playlist expansion)
YOUTUBE_API_KEY=<your-api-key>

# Google Sheets (for queue management)
GOOGLE_SHEET_ID=<sheet-id>
GOOGLE_SHEETS_CREDENTIAL_ID=<n8n-credential-id>
```

## Workflow Patterns

### 1. Lock Pattern (Concurrent Processing Prevention)

Prevent multiple workflow executions from processing the same item simultaneously.

**Implementation:**
- Use Google Sheet row as queue with `status` column
- Before processing: Check if `status` is `pending` or `error`
- During processing: Update `status` to `processing`
- After success: Update `status` to `completed`
- On failure: Update `status` to `error`

**n8n Nodes:**
1. Google Sheets Trigger (on row added)
2. IF node: Check `status == 'pending'`
3. Google Sheets Update: Set `status = 'processing'`
4. HTTP Request: Call trading-RAG API
5. Google Sheets Update: Set final status

### 2. Lease Expiration (Stuck Item Recovery)

Allow reprocessing of items stuck in `processing` state.

**Implementation:**
- Add `processing_started_at` timestamp column
- Scheduled workflow checks for stale items: `status == 'processing' AND processing_started_at < NOW() - 15 minutes`
- Reset stale items to `pending` for reprocessing

**n8n Nodes:**
1. Schedule Trigger (every 15 minutes)
2. Google Sheets: Query stale processing rows
3. Loop: Reset each to `status = 'pending'`

### 3. State Transitions

Track document processing through states:

```
pending → processing → completed
                    ↘ error → pending (retry)
```

**Google Sheet Columns:**
| Column | Type | Description |
|--------|------|-------------|
| url | string | YouTube URL or document source |
| status | string | pending/processing/completed/error |
| error_reason | string | Error message if failed |
| retryable | boolean | Whether error can be retried |
| processing_started_at | datetime | When processing began |
| completed_at | datetime | When processing finished |
| doc_id | string | UUID from trading-RAG |
| chunks_created | number | Chunk count |

### 4. Playlist Fan-Out

Expand YouTube playlists into individual video rows.

**Implementation:**
1. POST `/sources/youtube/ingest` with playlist URL
2. Response contains `is_playlist=true` and `video_urls` array
3. For each video URL, create a new row in the queue sheet
4. Each video processes independently

**n8n Nodes:**
1. HTTP Request: POST to `/sources/youtube/ingest`
2. IF: Check `is_playlist == true`
3. Split In Batches: Iterate `video_urls`
4. Google Sheets: Create row for each video

### 5. Idempotent Fan-Out

Prevent duplicate video rows when playlist is re-processed.

**Implementation:**
- Include `idempotency_key` in each row (e.g., `playlist_id:video_id`)
- Before creating row, check if key exists
- Skip if already present

**n8n Nodes:**
1. Google Sheets: Query existing rows by video_id
2. IF: Row doesn't exist
3. Google Sheets: Create new row

### 6. Retry Logic

Handle retryable errors with exponential backoff.

**Implementation:**
- Check `retryable` field in API response
- If retryable: increment `retry_count`, set `status = 'pending'`
- If not retryable: set `status = 'error'` (terminal)
- Max retries: 3 (configurable)
- Backoff: 1 min, 5 min, 15 min

**n8n Nodes:**
1. IF: `retryable == true AND retry_count < 3`
2. Google Sheets: Increment retry_count, set status='pending'
3. Wait node: Exponential delay
4. ELSE: Set terminal error state

## Google Sheet Schema

Required columns for YouTube ingestion queue:

```
| Column | Type | Required | Description |
|--------|------|----------|-------------|
| url | string | Yes | YouTube video/playlist URL |
| workspace_id | string | Yes | UUID for multi-tenancy |
| status | string | Yes | pending/processing/completed/error |
| error_reason | string | No | Error message |
| retryable | boolean | No | Can retry on error |
| retry_count | number | No | Current retry attempt |
| processing_started_at | datetime | No | Lock timestamp |
| completed_at | datetime | No | Completion timestamp |
| doc_id | string | No | Returned document UUID |
| chunks_created | number | No | Chunk count |
| video_id | string | No | Extracted video ID |
| playlist_id | string | No | Parent playlist (for fan-out) |
| idempotency_key | string | No | Deduplication key |
```

## API Endpoints

### YouTube Ingestion

```http
POST /sources/youtube/ingest
Content-Type: application/json

{
  "workspace_id": "uuid",
  "url": "https://youtube.com/watch?v=...",
  "idempotency_key": "optional-key",
  "max_playlist_videos": 200
}
```

**Response (success):**
```json
{
  "doc_id": "uuid",
  "video_id": "abc123",
  "status": "indexed",
  "chunks_created": 15,
  "retryable": false
}
```

**Response (playlist):**
```json
{
  "status": "playlist_expanded",
  "is_playlist": true,
  "playlist_id": "PLxxx",
  "video_urls": ["https://...", "https://..."]
}
```

**Response (error):**
```json
{
  "status": "error",
  "error_reason": "no_transcript",
  "retryable": false
}
```

## Workflow Templates

See `/n8n/templates/` in the main automation-infra repository for importable workflow JSON files.

## Monitoring

- Check `/health` endpoint for service status
- Circuit breaker status shows in `circuit_breakers` field
- Use ops alerts for failure notifications
