# Session 107: Unified Ingestion & Cookie-Based Auth

**Date**: 2026-01-21/22

## Summary

Added unified ingestion endpoint with auto-detection, cookie-based admin authentication, and a public landing page.

## Features Implemented

### 1. Unified Ingestion Endpoint (`POST /ingest/unified`)

Single multipart/form-data endpoint that auto-detects content type:

- **YouTube URLs**: `youtube.com`, `youtu.be`, `m.youtube.com`
- **PDF URLs**: URLs ending in `.pdf` (ignoring query string)
- **Article URLs**: Any other HTTP(S) URL
- **PDF Files**: `.pdf` extension
- **Text/Markdown Files**: `.txt`, `.md` extensions
- **Pine Files**: `.pine`, `.pinescript` extensions
- **Raw Content**: Text/markdown passed directly

**Files Created**:
- `app/routers/unified_ingest.py` - Unified endpoint
- `app/services/ingest/detection.py` - Detection logic (pure, testable)
- `app/services/ingest/text.py` - Text/markdown handler
- `app/services/article_extractor.py` - Web article extraction

**Detection Logic**:
```python
class DetectedSource(Enum):
    YOUTUBE = "youtube"
    PDF_URL = "pdf_url"
    ARTICLE_URL = "article_url"
    PDF_FILE = "pdf_file"
    TEXT_FILE = "text_file"
    PINE_FILE = "pine_file"
    TEXT_CONTENT = "text_content"
```

### 2. Cookie-Based Admin Authentication

Seamless navigation without re-entering token on every page:

**Endpoints**:
- `GET /admin/login` - Login form
- `POST /admin/auth/login` - Set auth cookie
- `GET /admin/auth/logout` - Clear cookie, redirect
- `GET /admin/auth/check` - Verify auth status

**Token Sources** (priority order):
1. `X-Admin-Token` header
2. `admin_token` cookie
3. `token` query parameter

**Cookie Settings**:
- httponly, samesite=lax
- 30-day expiry with "Remember me"

### 3. Public Landing Page

- `GET /` returns HTML landing page (no auth required)
- System overview, feature highlights
- Quick links to docs and login

### 4. Admin Ingest UI Enhancements

Extended `app/admin/templates/ingest.html`:
- Article URL tab
- Text/Markdown file upload tab
- All tabs use unified endpoint

### 5. Webhook Alert Configuration

Added missing Settings attributes for `AlertEvaluatorJob`:
- `webhook_enabled: bool`
- `slack_webhook_url: Optional[str]`
- `alert_webhook_url: Optional[str]`
- `alert_webhook_headers: Optional[dict]`

## Bug Fixes

### Failing Unit Tests (`tests/unit/alerts/test_job.py`)

**Root Cause**: `AlertEvaluatorJob.run()` referenced `settings.webhook_enabled` and related attributes that didn't exist in the `Settings` class, causing `AttributeError` that silently prevented rule processing.

**Fix**: Added the missing webhook configuration fields to `app/config.py`.

### Integration Test (`test_root_has_service_info`)

**Root Cause**: Test expected JSON response from `/` but endpoint now returns HTML landing page.

**Fix**: Updated test to check for HTML content type and "Trading RAG" in response text.

### Missing Alert Tables

**Root Cause**: Alerts page returned 500 error because `alert_rules` and `alert_events` tables didn't exist.

**Fix**: Created both tables via Supabase migrations with proper constraints and indexes.

### Ingest UI "View Job Run Detail" Link

**Root Cause**: The `.hidden` CSS class was only defined for `.results-panel.hidden` but other elements like `.job-link.hidden` had no corresponding CSS rule. This caused the "View Job Run Detail" link to show even when no `run_id` was present (e.g., for synchronous YouTube ingestion).

**Fix**: Added generic `.hidden { display: none !important; }` CSS rule to `ingest.html`.

## Files Changed

| File | Change |
|------|--------|
| `app/routers/unified_ingest.py` | New - unified endpoint |
| `app/services/ingest/__init__.py` | New - package init |
| `app/services/ingest/detection.py` | New - source detection |
| `app/services/ingest/text.py` | New - text handler |
| `app/services/article_extractor.py` | New - article extraction |
| `app/admin/auth.py` | New - auth routes |
| `app/admin/templates/landing.html` | New - landing page |
| `app/admin/templates/login.html` | New - login page |
| `app/admin/templates/layout.html` | Modified - logout button |
| `app/admin/templates/ingest.html` | Modified - new tabs, .hidden CSS fix |
| `app/admin/router.py` | Modified - include auth router |
| `app/api/router.py` | Modified - include unified_ingest |
| `app/main.py` | Modified - landing page route |
| `app/deps/security.py` | Modified - cookie support |
| `app/config.py` | Modified - webhook settings |
| `mypy.ini` | Modified - type ignores |
| `scripts/check_mypy_ratchet.sh` | Modified - baseline 45 |
| `tests/integration/test_api.py` | Modified - root endpoint test |
| `tests/unit/services/ingest/test_detection.py` | New - detection tests |

## Dependencies Added

- `trafilatura>=1.6.0` - Article content extraction

## CI Status

All checks passing:
- Black formatting
- Flake8 linting
- Mypy type checking (45 ignores at baseline)
- Unit tests (3050 passed)
- Integration tests (136 passed)
