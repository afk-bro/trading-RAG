## 1) Ingestion inputs (how data enters)

### Input types

* File uploads: PDF, TXT, MD, DOCX (optional), HTML (optional), JSON (optional)
* Links:

  * YouTube video
  * YouTube playlist
  * YouTube channel (optional)
  * Generic URL (web page)
* Raw text:

  * “Paste text” ingestion (manual)
* API ingestion:

  * “Bring your own content” via JSON (title + body + metadata)
* Batch ingestion:

  * multi-file upload
  * list of URLs
  * CSV manifest (source_uri + metadata) (optional)

### Input UX + validation

* Drag & drop, file picker
* Client-side type/size validation
* Server-side validation + virus/malware scan hook (optional)
* Duplicate detection warnings (“already ingested”)
* Preflight “estimated cost/time” (pages, tokens, chunks)

### Endpoint behaviors

* Workspace-scoped ingestion endpoints
* Async ingestion kickoff returning `job_id` + `document_id(s)`
* Idempotency keys (client-supplied) to prevent double submits
* Per-input metadata fields:

  * title override
  * tags
  * source label
  * visibility flags (private/public within workspace)
  * custom parsing settings override (one-off run)

---

## 2) Document creation + storage

### Storage backends

* Supabase Storage (recommended)
* Local filesystem (dev)
* S3-compatible (future)

### Document model features

* Document identity:

  * `document_id`, `workspace_id`
  * source_type (`upload|youtube|url|paste|api`)
  * source_uri (canonical)
  * checksum/hash (sha256)
* Document metadata:

  * title, author, published_at
  * mime_type, file_size, page_count
  * language detection
  * tags (array)
  * custom metadata (jsonb)
* Provenance:

  * created_by / owner_id
  * ingested_at timestamp
  * ingestion method + pipeline version
* Document lifecycle:

  * statuses: `queued, fetching, extracting, chunking, embedding, indexing, ready, failed, disabled`
  * error fields: `error_code`, `error_message`, `error_stack` (optional)
  * warnings array (low text density, missing captions, etc.)
* Versioning:

  * document re-ingest creates `revision` or new doc with `supersedes_document_id`
  * “reindex” with updated chunking or embed model
* Dedup & collisions:

  * unique constraints (workspace+source_uri) and/or (workspace+checksum)
  * dedup policy: skip, merge, new revision
* Retention:

  * delete doc → cascade chunks/vectors
  * soft delete with restore (optional)
  * archive/unarchive

### File handling

* Original file stored (binary)
* Extracted plain text stored (optional, for debugging)
* Extracted structured data (pages, sections) stored (optional)
* Access control for stored files (signed URLs)

---

## 3) PDF/Text extraction layer

### PDF extraction capabilities (v1 “simple”)

* Extract text by page (PyMuPDF)
* Extract text blocks with coordinates (optional)
* Detect:

  * scanned/no-text PDF
  * multi-column layout risk (heuristic)
  * repeated headers/footers (heuristic)
* Page range controls:

  * max pages
  * skip first/last pages options
* Basic cleanup:

  * normalize whitespace
  * remove hyphenation at line breaks (optional)
  * de-duplicate repeated lines

### Other extractors

* TXT/MD: read as-is + normalize newlines
* HTML:

  * boilerplate removal (readability-like) (optional)
  * convert to text with preserved headings (optional)
* DOCX:

  * extract text + headings (optional)

### Observability + quality scoring

* Extract report:

  * chars/page, total chars
  * percent blank pages
  * warnings list
* Quality score (0–1) to decide if OCR is needed

### Extensibility hooks

* Backend selection:

  * `pymupdf`, `pdfplumber`, future `mineru`, future `studyg-service`
* Workspace-configurable extraction settings (`workspaces.config.pdf`)
* Override extraction config per ingestion job

---

## 4) Chunking + metadata

### Chunking strategies

* Fixed-size token chunking
* Recursive split (paragraph → sentence → token)
* Section-aware chunking (headings) (optional)
* Page-aware chunking (keep page boundaries)
* Table-aware chunking (optional)
* Code-aware chunking (optional)

### Parameters (workspace defaults + per-job override)

* chunk size (tokens/characters)
* overlap
* separators priority
* min chunk size (drop tiny chunks)
* max chunk size (force split)
* “join pages with marker” behavior

### Metadata features

* Attach chunk metadata:

  * chunk index
  * page_start/page_end
  * section heading path (optional)
  * timestamps for YouTube (start/end seconds)
  * speaker labels (optional)
  * source_type/source_uri
  * language
  * tags
* Metadata normalization:

  * whitelist keys stored in vector payload (keep filters lean)
  * store full metadata in DB jsonb

### Data quality controls

* Remove duplicate chunks
* Remove low-signal chunks (too short, too repetitive, mostly symbols)
* PII flagging (optional)
* profanity filtering (optional, usually unnecessary)

---

## 5) Embedding + vector upsert

### Embedding provider features

* Provider selection:

  * `ollama` local embeddings
  * `openai` (future)
  * `cohere` / `voyage` (future)
* Model selection per workspace + override per job
* Batch embedding with configurable batch sizes
* Rate limiting / concurrency control
* Retry policies:

  * exponential backoff
  * dead-letter queue state for failed chunks

### Vector schema + consistency

* Store embedding dimension + model id on each vector record
* Validate that:

  * collection exists and dimension matches
  * distance metric is compatible
* Vector payload design:

  * `workspace_id`, `document_id`, `chunk_id`
  * filterable fields: source_type, tags, created_at, page_start, youtube_ts
* Multi-collection routing:

  * per-workspace collection
  * or per-doc type collections (optional)

### Qdrant upsert features

* Create collection if missing (on workspace create or first ingest)
* Upsert points id strategy:

  * use chunk_id UUID as point id
* Partial reindex:

  * delete vectors for a document then re-upsert
  * “re-embed only changed chunks”
* Consistency checks:

  * verify count in Qdrant matches chunk_vectors rows
  * reconciliation job

---

## 6) Search/Retrieval API (verification backbone)

### Query modes

* Semantic vector search (default)
* Hybrid search (BM25 + vector) (future)
* Metadata-only filter search (documents/chunks)
* “Find in document” fulltext (optional via Postgres FTS)

### Retrieval features

* top_k controls
* score threshold
* reranking:

  * lightweight heuristic rerank (source bias, freshness)
  * future cross-encoder reranker
* Chunk grouping:

  * group results by document
  * merge adjacent chunks
* Snippet generation:

  * highlight query terms
  * show page/timestamp context

### Filters + facets

* filter by:

  * document_id
  * source_type
  * tags
  * created_at range
  * page range
  * youtube timestamp range
* Facets (counts by tags/source_type) (optional)

### Debug and developer tools

* “Explain search” mode:

  * embed model used
  * collection used
  * filters applied
  * score breakdown (as available)
* Raw Qdrant response toggle (dev only)

---

## 7) Q&A API (simple RAG)

### Core RAG features

* Query → retrieve chunks → compose context → generate answer
* Workspace routing header drives:

  * embed model (for query embedding)
  * collection
  * distance
* Prompt templates:

  * default system prompt
  * workspace custom prompt override
* Answer formatting:

  * markdown output
  * bullet summaries
  * “direct quote” blocks (short)
* Citations:

  * chunk_id, document title, page range/timestamp
  * link to source (YouTube URL w/ t=, file viewer anchor)
* Guardrails:

  * “I don’t know” if evidence insufficient
  * require citations for factual claims (soft rule)

### Conversation support (optional but valuable)

* Threaded chat sessions
* Store:

  * user messages
  * assistant responses
  * retrieved chunk ids
  * model metadata
* “Regenerate” with same retrieval set vs re-retrieve

### Evaluation hooks

* Feedback buttons:

  * 👍/👎
  * “citation wrong”
  * “missing info”
* Save “golden questions” per workspace (for regression tests)

---

## 8) Background jobs + statuses (so UI isn’t blocking)

### Job system features

* Job types:

  * ingest_upload
  * ingest_youtube
  * extract_pdf
  * chunk
  * embed
  * upsert_qdrant
  * reindex_document
  * delete_document
* Job states:

  * queued, running, succeeded, failed, cancelled
* Progress reporting:

  * percent complete
  * stage label (extracting/chunking/embedding)
  * processed counts (pages, chunks embedded)
* Retry handling:

  * automatic retries for transient failures
  * max retries + backoff
* Dead-letter queue:

  * failed jobs preserved with error details

### Status endpoints / events

* Polling endpoints for job/document status
* SSE stream for job updates (nice in UI)
* Job logs:

  * structured logs per job
  * downloadable debug info (dev only)

### Worker architecture

* v1: in-process background tasks
* v2: separate worker container + queue (Redis)
* Concurrency limits per workspace

---

## 9) Minimal UI to prove it works (control panel)

### Workspace UX

* Workspace selector (slug)
* Workspace settings:

  * routing header fields
  * config JSON editor with schema validation
  * toggle ingestion_enabled / is_active
* Show collection stats:

  * docs count, chunks count, vectors count, last ingest time

### Ingestion UX

* Upload widget:

  * file list
  * per-file status
  * error display + retry
* YouTube form:

  * paste link
  * auto-detect playlist vs video
  * show fetched title/thumbnail (optional)

### Documents UX

* Documents table:

  * title, source_type, status, created_at, page_count, tags
  * actions: view, reindex, delete, download original
* Document detail page:

  * extraction warnings
  * chunk count
  * view extracted text preview
  * view chunks with metadata

### Verification UX

* Search tab:

  * query input
  * filters
  * results list with scores + citations
* Ask tab (chat):

  * answer + citations
  * expandable “context used”
  * feedback buttons
* Debug view (dev):

  * show prompt + retrieved chunk ids

---

## 10) Guardrails (small but important)

### Security & access

* Auth:

  * simple API key header (dev)
  * Supabase Auth / JWT (prod)
* Workspace-level access control:

  * owner_id checks
  * service-role-only ingestion (if you want it closed)
* Storage access:

  * signed URLs
  * restrict raw file access

### Safety & correctness

* File size limits, mime allowlist
* Rate limits per IP/workspace/user
* Idempotency keys for ingest endpoints
* Input sanitization for URLs
* Timeout limits for extraction/embedding
* Quotas:

  * max docs per workspace
  * max pages per doc
  * max chunks per doc

### Data integrity

* Transaction boundaries:

  * doc row created → chunks created → vectors created
  * mark doc failed if stage fails
* Cleanup on failure:

  * delete partial chunks/vectors OR keep for debugging with status flags
* Reconciliation tools:

  * “repair workspace”: compare DB vectors vs Qdrant points

### Observability

* Structured logging
* Request IDs
* Metrics:

  * ingest throughput
  * embedding latency
  * search latency
* Error reporting (Sentry or similar)

---

If you want, I can convert this into a **single “milestones” build plan** (vertical slices) that maps directly to your tables and endpoints, so you can knock it out in the cleanest order without feature-bloat.
