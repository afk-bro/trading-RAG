# Pine Script System

Parsing, linting, and ingestion system for Pine Script files.

## Pine Script Registry

Located in `app/services/pine/`.

**Purpose**: Catalog Pine Script files with metadata extraction and static analysis for downstream RAG ingestion.

### Architecture

```
.pine files → Filesystem Adapter → Parser → Linter → Registry Builder → JSON artifacts
```

### Components

| File | Description |
|------|-------------|
| `models.py` | Data models: `PineRegistry`, `PineScriptEntry`, `PineLintReport`, `LintFinding` |
| `parser.py` | Regex-based parser extracts version, declaration, inputs, imports, features |
| `linter.py` | Static analysis rules (E001-E003 errors, W002-W003 warnings, I001-I002 info) |
| `registry.py` | Build orchestration + CLI entry point |
| `adapters/filesystem.py` | File scanning returning `SourceFile` structured output |

### CLI Usage

```bash
python -m app.services.pine --build ./scripts           # Build from directory
python -m app.services.pine --build ./scripts -o ./data # Custom output dir
python -m app.services.pine --build ./scripts -q        # Quiet mode
```

### Output Artifacts

- `pine_registry.json` - Script metadata (version, type, title, inputs, imports, features) with lint summaries
- `pine_lint_report.json` - Full lint findings per script

### Design Choices

- **Best-effort**: Parse errors recorded as E999 synthetic errors, build continues
- **Deterministic**: Sorted keys, consistent JSON formatting for diff stability
- **Fingerprinted**: SHA256 from raw content for change detection
- **GitHub-ready**: `root_kind` field distinguishes filesystem vs future GitHub adapter

### Lint Rules

| Code | Severity | Description |
|------|----------|-------------|
| E001 | Error | Missing `//@version` directive |
| E002 | Error | Invalid version number |
| E003 | Error | Missing declaration (`indicator`/`strategy`/`library`) |
| W002 | Warning | `lookahead=barmerge.lookahead_on` (future data leakage risk) |
| W003 | Warning | Deprecated `security()` instead of `request.security()` |
| I001 | Info | Script has exports but is not a library |
| I002 | Info | Script exceeds recommended line count (500) |

---

## Pine Script Ingest API

Admin-only endpoint for ingesting Pine Script registries into the RAG system.

**Endpoint**: `POST /sources/pine/ingest`
**Authentication**: Requires `X-Admin-Token` header.

### Request

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

### Parameters

| Parameter | Description |
|-----------|-------------|
| `registry_path` | Server path to `pine_registry.json` (must be within `DATA_DIR`) |
| `lint_path` | Optional lint report path (auto-derived if null) |
| `source_root` | Directory with `.pine` files (defaults to registry parent) |
| `skip_lint_errors` | Skip scripts with lint errors |
| `update_existing` | Upsert if sha256 changed (false = skip changed scripts) |
| `dry_run` | Validate without database writes |

### Response

```json
{
  "status": "success",
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

**Status Values**: `success`, `partial`, `failed`, `dry_run`

### Security

- All paths validated against `DATA_DIR` allowlist
- Path traversal attempts return 403
- Non-.json extensions rejected with 400

---

## Pine Script Read APIs

Admin-only endpoints for querying indexed Pine scripts.

### List Endpoint

```
GET /sources/pine/scripts?workspace_id=<uuid>&symbol=BTC&q=breakout&limit=20
```

**Query Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `workspace_id` | Target workspace UUID (required) | - |
| `symbol` | Filter by ticker symbol (GIN index) | - |
| `status` | `active`, `superseded`, `deleted`, `all` | `active` |
| `q` | Free-text search on title | - |
| `order_by` | `updated_at`, `created_at`, `title` | `updated_at` |
| `order_dir` | `desc`, `asc` | `desc` |
| `limit` | Results per page (1-100) | 20 |
| `offset` | Pagination offset | 0 |

### Detail Endpoint

```
GET /sources/pine/scripts/<uuid>?workspace_id=<uuid>&include_chunks=true&include_lint_findings=true
```

**Additional Parameters**:
- `include_chunks` - Include chunk content (default false)
- `chunk_limit` - Chunks per page (1-200, default 50)
- `include_lint_findings` - Include lint findings array (default false, capped at 200)

---

## Auto-Strategy Discovery

Automatic parameter spec generation from Pine Script inputs for backtesting automation.

**Source**: `app/services/pine/spec_generator.py`

### Pipeline

```
Pine Script → Parser → PineInput[] → SpecGenerator → StrategySpec
```

### Key Components

- `ParamSpec` - Parameter specification with bounds, step, options, sweepable flag, priority
- `StrategySpec` - Complete strategy spec with params list and auto-generated sweep config
- `pine_input_to_param_spec()` - Converts parsed Pine inputs to ParamSpec
- `generate_strategy_spec()` - Generates full StrategySpec from PineScriptEntry

### Sweepable Detection

| Input Type | Sweepable When |
|------------|----------------|
| Bool | Always (true/false) |
| Int/Float | `min_value` and `max_value` defined |
| Options | `options` length > 1 |
| Source/color/session | Generally not sweepable |

### Priority Scoring

Higher priority = more likely to affect strategy.

| Factor | Adjustment |
|--------|------------|
| Base (int/float) | +10 |
| Base (bool) | +5 |
| Keywords: `length`, `period`, `threshold`, `atr`, `rsi` | +10 |
| Keywords: `color`, `style`, `display`, `show` | -10 |
| Bounds present | +15 |

### Usage

```python
from app.services.pine.spec_generator import generate_strategy_spec

spec = generate_strategy_spec(pine_entry)
sweepable = spec.sweepable_params  # Only params suitable for optimization
sweep_config = spec.sweep_config   # Auto-generated grid for tuning
```
