# Pine Script System

Parsing, linting, and ingestion system for Pine Script files.

## Registry (`app/services/pine/`)

**Purpose**: Catalog Pine Script files with metadata extraction and static analysis for downstream RAG ingestion.

**Architecture**:
```
.pine files → Filesystem Adapter → Parser → Linter → Registry Builder → JSON artifacts
```

**Components**:
- `models.py` - Data models: `PineRegistry`, `PineScriptEntry`, `PineLintReport`, `LintFinding`
- `parser.py` - Regex-based parser extracts version, declaration, inputs, imports, features
- `linter.py` - Static analysis rules
- `registry.py` - Build orchestration + CLI entry point
- `adapters/filesystem.py` - File scanning

**CLI Usage**:
```bash
python -m app.services.pine --build ./scripts           # Build from directory
python -m app.services.pine --build ./scripts -o ./data # Custom output dir
python -m app.services.pine --build ./scripts -q        # Quiet mode
```

**Output Artifacts**:
- `pine_registry.json` - Script metadata with lint summaries
- `pine_lint_report.json` - Full lint findings per script

**Lint Rules**:
| Code | Severity | Description |
|------|----------|-------------|
| E001 | Error | Missing `//@version` directive |
| E002 | Error | Invalid version number |
| E003 | Error | Missing declaration (`indicator`/`strategy`/`library`) |
| W002 | Warning | `lookahead=barmerge.lookahead_on` (future data leakage) |
| W003 | Warning | Deprecated `security()` instead of `request.security()` |
| I001 | Info | Script has exports but is not a library |
| I002 | Info | Script exceeds recommended line count (500) |

## Ingest API

**Endpoint**: `POST /sources/pine/ingest` (admin-only)

**Request**:
```json
{
  "workspace_id": "uuid",
  "registry_path": "/data/pine/pine_registry.json",
  "lint_path": null,
  "source_root": null,
  "include_source": true,
  "max_source_lines": 100,
  "skip_lint_errors": false,
  "update_existing": false,
  "dry_run": false
}
```

**Response**:
```json
{
  "status": "success|partial|failed|dry_run",
  "scripts_processed": 50,
  "scripts_indexed": 45,
  "scripts_already_indexed": 3,
  "scripts_skipped": 2,
  "scripts_failed": 0,
  "chunks_added": 96,
  "errors": [],
  "ingest_run_id": "pine-ingest-abc12345"
}
```

**Security**: All paths validated against `DATA_DIR` allowlist. Path traversal returns 403.

## Read APIs

**List**: `GET /sources/pine/scripts?workspace_id=<uuid>&symbol=BTC&q=breakout&limit=20`

Query params: `workspace_id` (required), `symbol`, `status`, `q`, `order_by`, `order_dir`, `limit`, `offset`

**Detail**: `GET /sources/pine/scripts/{doc_id}?workspace_id=<uuid>&include_chunks=true`

Query params: `workspace_id` (required), `include_chunks`, `chunk_limit`, `chunk_offset`, `include_lint_findings`

## DB-Backed Discovery State

Pine script discovery state is persisted in two tables:

**`strategy_scripts`** — One row per discovered script (unique on `workspace_id + source_type + rel_path`):
- `sha256`, `status` (`discovered → spec_generated → published → archived`), `spec_json`, `lint_json`
- `doc_id`, `ingest_status`, `last_ingested_sha` — ingest tracking (populated after KB ingest)
- `repo_id`, `scan_commit`, `source_url` — populated for GitHub-sourced scripts
- `deleted_at` — soft delete; `NULL` = active

**`pine_repos`** — GitHub repositories registered for auto-discovery:
- `repo_slug` (owner/repo), `branch`, `clone_path`, `scan_globs` (`**/*.pine` by default)
- `last_scan_at`, `last_scan_ok`, `scripts_count` — scan state
- `next_scan_at`, `failure_count` — polling schedule with exponential backoff on failures

The pg_cron function `pine_archive_stale_scripts(older_than_days, batch_limit, dry_run)` archives scripts not seen in N days (default 7). Runs at 3:35 AM UTC daily when pg_cron is available; otherwise trigger via the admin endpoint.

## Auto-Strategy Discovery

Automatic parameter spec generation from Pine Script inputs (`app/services/pine/spec_generator.py`).

```
Pine Script → Parser → PineInput[] → SpecGenerator → StrategySpec
```

**Sweepable Detection**:
- Bool inputs: Always sweepable
- Int/Float with bounds: Sweepable if `min_value` and `max_value` defined
- Options array: Sweepable if length > 1
- Source/color/session: Not sweepable

**Priority Scoring** (higher = more likely to affect strategy):
- Base by type: int/float = 10, bool = 5
- Keywords boost: `length`, `period`, `threshold` = +10
- Keywords penalty: `color`, `style`, `display` = -10
- Bounds present: +15

**Usage**:
```python
from app.services.pine.spec_generator import generate_strategy_spec
spec = generate_strategy_spec(pine_entry)
sweepable = spec.sweepable_params
sweep_config = spec.sweep_config
```
