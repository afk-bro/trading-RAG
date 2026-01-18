# Backtest Automation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement job-based backtesting automation with CCXT data fetching, parameter sweeps, and walk-forward optimization.

**Architecture:** Postgres-backed job queue with pg_cron scheduling. Workers claim jobs via `FOR UPDATE SKIP LOCKED`. Market data fetched via CCXT (KuCoin/Binance) and stored in `ohlcv_candles` table. TuneJob runs parameter sweeps via existing `BacktestEngine`. WFOJob orchestrates folds of child TuneJobs.

**Tech Stack:** FastAPI, asyncpg, CCXT, backtesting.py, Pydantic, pytest

**Design Doc:** `docs/plans/2025-01-18-backtest-automation-design.md` (commit 15a80a7)

---

## PR0: Scaffolding & Conventions

**Goal:** Add foundational types, config knobs, and package structure without behavior changes.

### Task 0.1: Add Job System Types

**Files:**
- Create: `app/jobs/types.py`
- Test: `tests/unit/jobs/test_types.py`

**Step 1: Write the failing test**

```python
# tests/unit/jobs/test_types.py
"""Tests for job system types."""
import pytest
from app.jobs.types import JobType, JobStatus


class TestJobType:
    def test_job_types_exist(self):
        assert JobType.DATA_SYNC == "data_sync"
        assert JobType.DATA_FETCH == "data_fetch"
        assert JobType.TUNE == "tune"
        assert JobType.WFO == "wfo"

    def test_job_type_values(self):
        assert set(JobType) == {
            JobType.DATA_SYNC,
            JobType.DATA_FETCH,
            JobType.TUNE,
            JobType.WFO,
        }


class TestJobStatus:
    def test_job_statuses_exist(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.SUCCEEDED == "succeeded"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELED == "canceled"

    def test_terminal_statuses(self):
        assert JobStatus.SUCCEEDED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELED.is_terminal
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.RUNNING.is_terminal
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/jobs/test_types.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.jobs.types'"

**Step 3: Create tests directory structure**

```bash
mkdir -p tests/unit/jobs
touch tests/unit/jobs/__init__.py
```

**Step 4: Write minimal implementation**

```python
# app/jobs/types.py
"""Job system type definitions."""
from enum import Enum


class JobType(str, Enum):
    """Job types for the automation system."""

    DATA_SYNC = "data_sync"
    DATA_FETCH = "data_fetch"
    TUNE = "tune"
    WFO = "wfo"


class JobStatus(str, Enum):
    """Job lifecycle statuses."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        """Check if this status is terminal (job won't change)."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/unit/jobs/test_types.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/jobs/types.py tests/unit/jobs/
git commit -m "feat(jobs): add job type and status enums

- JobType: data_sync, data_fetch, tune, wfo
- JobStatus: pending, running, succeeded, failed, canceled
- is_terminal property for status checks

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 0.2: Add Job Models

**Files:**
- Create: `app/jobs/models.py`
- Test: `tests/unit/jobs/test_models.py`

**Step 1: Write the failing test**

```python
# tests/unit/jobs/test_models.py
"""Tests for job models."""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from app.jobs.models import Job, JobEvent
from app.jobs.types import JobType, JobStatus


class TestJob:
    def test_create_job(self):
        job = Job(
            id=uuid4(),
            type=JobType.DATA_FETCH,
            status=JobStatus.PENDING,
            payload={"symbol": "BTC-USDT"},
        )
        assert job.type == JobType.DATA_FETCH
        assert job.status == JobStatus.PENDING
        assert job.attempt == 0
        assert job.max_attempts == 3

    def test_job_with_workspace(self):
        ws_id = uuid4()
        job = Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.PENDING,
            payload={},
            workspace_id=ws_id,
        )
        assert job.workspace_id == ws_id

    def test_job_with_parent(self):
        parent_id = uuid4()
        job = Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.PENDING,
            payload={},
            parent_job_id=parent_id,
        )
        assert job.parent_job_id == parent_id


class TestJobEvent:
    def test_create_event(self):
        job_id = uuid4()
        event = JobEvent(
            job_id=job_id,
            level="info",
            message="Job started",
        )
        assert event.job_id == job_id
        assert event.level == "info"
        assert event.meta is None

    def test_event_with_meta(self):
        event = JobEvent(
            job_id=uuid4(),
            level="error",
            message="Fetch failed",
            meta={"error_code": "RATE_LIMIT"},
        )
        assert event.meta == {"error_code": "RATE_LIMIT"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/jobs/test_models.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# app/jobs/models.py
"""Job system data models."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from app.jobs.types import JobType, JobStatus


@dataclass
class Job:
    """A job in the queue."""

    id: UUID
    type: JobType
    status: JobStatus
    payload: dict[str, Any]

    # Retry handling
    attempt: int = 0
    max_attempts: int = 3
    run_after: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Lock info
    locked_at: Optional[datetime] = None
    locked_by: Optional[str] = None

    # Relationships
    parent_job_id: Optional[UUID] = None
    workspace_id: Optional[UUID] = None
    dedupe_key: Optional[str] = None

    # Lifecycle timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result (populated on completion)
    result: Optional[dict[str, Any]] = None
    priority: int = 100


@dataclass
class JobEvent:
    """An event logged during job execution."""

    job_id: UUID
    level: str  # "info", "warn", "error"
    message: str
    meta: Optional[dict[str, Any]] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Optional[int] = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/jobs/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/jobs/models.py tests/unit/jobs/test_models.py
git commit -m "feat(jobs): add Job and JobEvent dataclasses

- Job with retry, locking, relationships, lifecycle tracking
- JobEvent for structured job logging

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 0.3: Add Config Knobs

**Files:**
- Modify: `app/config.py`
- Test: `tests/unit/test_config.py` (add to existing)

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_config.py or create if doesn't exist
def test_data_config_defaults():
    """Test data layer config defaults."""
    from app.config import get_settings
    settings = get_settings()

    # Clear cache to get fresh settings
    get_settings.cache_clear()

    # These should have defaults
    assert hasattr(settings, "data_dir")
    assert hasattr(settings, "ccxt_rate_limit_ms")
    assert hasattr(settings, "core_timeframes")


def test_ccxt_rate_limit_default():
    from app.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.ccxt_rate_limit_ms == 100


def test_core_timeframes_default():
    from app.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.core_timeframes == ["1m", "5m", "15m", "1h", "1d"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_ccxt_rate_limit_default -v`
Expected: FAIL with AttributeError

**Step 3: Add config fields to app/config.py**

Add these fields to the Settings class (around line 275, before the `@property` methods):

```python
    # Job System Configuration
    ccxt_rate_limit_ms: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Minimum milliseconds between CCXT API calls",
    )
    core_timeframes: list[str] = Field(
        default=["1m", "5m", "15m", "1h", "1d"],
        description="Default timeframes for core symbol sync",
    )
    job_poll_interval_s: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Job queue poll interval in seconds",
    )
    job_stale_timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Minutes before a running job is considered stale",
    )
    artifacts_dir: str = Field(
        default="/data/artifacts",
        description="Directory for storing job artifacts (tunes, WFO results)",
    )
    artifacts_retention_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Days to retain unpinned artifacts",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::test_ccxt_rate_limit_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py tests/unit/test_config.py
git commit -m "feat(config): add job system configuration knobs

- ccxt_rate_limit_ms: API rate limiting
- core_timeframes: default timeframes for sync
- job_poll_interval_s, job_stale_timeout_minutes
- artifacts_dir, artifacts_retention_days

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 0.4: Add Job Registry Skeleton

**Files:**
- Create: `app/jobs/registry.py`
- Test: `tests/unit/jobs/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/unit/jobs/test_registry.py
"""Tests for job registry."""
import pytest
from app.jobs.registry import JobRegistry
from app.jobs.types import JobType


class TestJobRegistry:
    def test_register_handler(self):
        registry = JobRegistry()

        async def dummy_handler(job, ctx):
            return {"ok": True}

        registry.register(JobType.DATA_FETCH, dummy_handler)
        assert registry.get_handler(JobType.DATA_FETCH) == dummy_handler

    def test_get_unregistered_handler_raises(self):
        registry = JobRegistry()
        with pytest.raises(KeyError):
            registry.get_handler(JobType.DATA_FETCH)

    def test_decorator_registration(self):
        registry = JobRegistry()

        @registry.handler(JobType.TUNE)
        async def tune_handler(job, ctx):
            pass

        assert registry.get_handler(JobType.TUNE) == tune_handler
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/jobs/test_registry.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# app/jobs/registry.py
"""Job handler registry."""
from typing import Any, Callable, Coroutine

from app.jobs.types import JobType
from app.jobs.models import Job

# Handler signature: async def handler(job: Job, ctx: dict) -> dict
JobHandler = Callable[[Job, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class JobRegistry:
    """Registry mapping job types to their handlers."""

    def __init__(self):
        self._handlers: dict[JobType, JobHandler] = {}

    def register(self, job_type: JobType, handler: JobHandler) -> None:
        """Register a handler for a job type."""
        self._handlers[job_type] = handler

    def get_handler(self, job_type: JobType) -> JobHandler:
        """Get the handler for a job type. Raises KeyError if not found."""
        if job_type not in self._handlers:
            raise KeyError(f"No handler registered for job type: {job_type}")
        return self._handlers[job_type]

    def handler(self, job_type: JobType) -> Callable[[JobHandler], JobHandler]:
        """Decorator to register a handler."""

        def decorator(fn: JobHandler) -> JobHandler:
            self.register(job_type, fn)
            return fn

        return decorator


# Global registry instance
default_registry = JobRegistry()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/jobs/test_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/jobs/registry.py tests/unit/jobs/test_registry.py
git commit -m "feat(jobs): add job handler registry

- JobRegistry class for type → handler mapping
- Decorator syntax for registration
- Global default_registry instance

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 0.5: Update Job Package Exports

**Files:**
- Modify: `app/jobs/__init__.py`

**Step 1: Write the failing test**

```python
# tests/unit/jobs/test_init.py
"""Test job package exports."""


def test_job_package_exports():
    from app.jobs import (
        JobType,
        JobStatus,
        Job,
        JobEvent,
        JobRegistry,
        default_registry,
    )

    assert JobType.TUNE == "tune"
    assert JobStatus.PENDING == "pending"
    assert Job is not None
    assert JobEvent is not None
    assert JobRegistry is not None
    assert default_registry is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/jobs/test_init.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# app/jobs/__init__.py
"""Job system package."""
from app.jobs.types import JobType, JobStatus
from app.jobs.models import Job, JobEvent
from app.jobs.registry import JobRegistry, default_registry

__all__ = [
    "JobType",
    "JobStatus",
    "Job",
    "JobEvent",
    "JobRegistry",
    "default_registry",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/jobs/test_init.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/jobs/__init__.py tests/unit/jobs/test_init.py
git commit -m "feat(jobs): export job types from package

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 0.6: Run Full Test Suite & Verify PR0 Complete

**Step 1: Run linting**

```bash
black app/jobs/ tests/unit/jobs/
flake8 app/jobs/ tests/unit/jobs/ --max-line-length=100
mypy app/jobs/ --ignore-missing-imports
```

**Step 2: Run full test suite**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: All tests pass

**Step 3: Final PR0 commit (if any fixes needed)**

```bash
git add -A
git commit -m "chore: PR0 lint fixes and cleanup

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## PR1: Data Layer Tables + Admin Core Symbols

**Goal:** Create OHLCV storage schema and admin endpoints for managing core symbols.

### Task 1.1: Create OHLCV Candles Migration

**Files:**
- Create: `migrations/062_ohlcv_candles.sql`

**Step 1: Write the migration**

```sql
-- migrations/062_ohlcv_candles.sql
-- OHLCV candle storage for market data

CREATE TABLE IF NOT EXISTS ohlcv_candles (
    exchange_id    TEXT NOT NULL,
    symbol         TEXT NOT NULL,           -- canonical: 'BTC-USDT'
    timeframe      TEXT NOT NULL CHECK (timeframe IN ('1m','5m','15m','1h','1d')),
    ts             TIMESTAMPTZ NOT NULL,    -- candle close, aligned to TF boundary, UTC
    open           DOUBLE PRECISION NOT NULL,
    high           DOUBLE PRECISION NOT NULL CHECK (high >= GREATEST(open, close, low)),
    low            DOUBLE PRECISION NOT NULL CHECK (low <= LEAST(open, close, high)),
    close          DOUBLE PRECISION NOT NULL,
    volume         DOUBLE PRECISION NOT NULL CHECK (volume >= 0),
    PRIMARY KEY (exchange_id, symbol, timeframe, ts)
);

-- Efficient range queries
CREATE INDEX IF NOT EXISTS idx_ohlcv_range
    ON ohlcv_candles (exchange_id, symbol, timeframe, ts DESC);

-- Comment for documentation
COMMENT ON TABLE ohlcv_candles IS 'OHLCV market data storage for backtesting automation';
COMMENT ON COLUMN ohlcv_candles.ts IS 'Candle close timestamp, UTC, aligned to timeframe boundary';
```

**Step 2: Apply migration via Supabase MCP**

Use the `mcp__supabase__apply_migration` tool with name `ohlcv_candles`.

**Step 3: Verify table exists**

```sql
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'ohlcv_candles' ORDER BY ordinal_position;
```

**Step 4: Commit**

```bash
git add migrations/062_ohlcv_candles.sql
git commit -m "feat(db): add ohlcv_candles table for market data

- Primary key: (exchange_id, symbol, timeframe, ts)
- CHECK constraints for OHLCV validity
- Descending index for efficient range queries

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.2: Create Core Symbols Migration

**Files:**
- Create: `migrations/063_core_symbols.sql`

**Step 1: Write the migration**

```sql
-- migrations/063_core_symbols.sql
-- Core symbols and symbol request tracking

CREATE TABLE IF NOT EXISTS core_symbols (
    exchange_id      TEXT NOT NULL,
    canonical_symbol TEXT NOT NULL,
    raw_symbol       TEXT NOT NULL,
    timeframes       TEXT[] DEFAULT ARRAY['1m','5m','15m','1h','1d'],
    is_enabled       BOOLEAN DEFAULT true,
    added_at         TIMESTAMPTZ DEFAULT now(),
    added_by         TEXT,
    UNIQUE (exchange_id, canonical_symbol)
);

CREATE INDEX IF NOT EXISTS idx_core_symbols_enabled
    ON core_symbols (exchange_id) WHERE is_enabled = true;

-- Write-only log for future auto-promote feature
CREATE TABLE IF NOT EXISTS symbol_requests (
    id               BIGSERIAL PRIMARY KEY,
    exchange_id      TEXT NOT NULL,
    canonical_symbol TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    requested_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_symbol_requests_symbol
    ON symbol_requests (exchange_id, canonical_symbol, requested_at DESC);

COMMENT ON TABLE core_symbols IS 'Universe of symbols to keep warm via scheduled sync';
COMMENT ON TABLE symbol_requests IS 'Log of ad-hoc symbol requests for future auto-promote';
```

**Step 2: Apply migration via Supabase MCP**

**Step 3: Commit**

```bash
git add migrations/063_core_symbols.sql
git commit -m "feat(db): add core_symbols and symbol_requests tables

- core_symbols: managed symbol universe
- symbol_requests: write-only log for auto-promote

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.3: Create Data Revisions Migration

**Files:**
- Create: `migrations/064_data_revisions.sql`

**Step 1: Write the migration**

```sql
-- migrations/064_data_revisions.sql
-- Data revision tracking for drift detection

CREATE TABLE IF NOT EXISTS data_revisions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange_id  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    start_ts     TIMESTAMPTZ NOT NULL,
    end_ts       TIMESTAMPTZ NOT NULL,
    row_count    INT NOT NULL,
    checksum     TEXT NOT NULL,            -- deterministic sample hash
    computed_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (exchange_id, symbol, timeframe, start_ts, end_ts)
);

CREATE INDEX IF NOT EXISTS idx_data_revisions_lookup
    ON data_revisions (exchange_id, symbol, timeframe, computed_at DESC);

COMMENT ON TABLE data_revisions IS 'Checksums for detecting data drift in OHLCV ranges';
COMMENT ON COLUMN data_revisions.checksum IS 'SHA256 truncated to 16 chars of sampled candles';
```

**Step 2: Apply migration via Supabase MCP**

**Step 3: Commit**

```bash
git add migrations/064_data_revisions.sql
git commit -m "feat(db): add data_revisions table for drift detection

- Tracks checksums for OHLCV ranges
- Unique constraint on (exchange, symbol, tf, start, end)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.4: Create OHLCV Repository

**Files:**
- Create: `app/repositories/ohlcv.py`
- Test: `tests/unit/repositories/test_ohlcv.py`

**Step 1: Write the failing test**

```python
# tests/unit/repositories/test_ohlcv.py
"""Tests for OHLCV repository."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.repositories.ohlcv import OHLCVRepository, Candle


class TestCandle:
    def test_candle_creation(self):
        candle = Candle(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=100.5,
        )
        assert candle.symbol == "BTC-USDT"
        assert candle.close == 42200.0

    def test_candle_ohlc_validation(self):
        # high >= all others
        with pytest.raises(ValueError):
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime.now(timezone.utc),
                open=42000.0,
                high=41000.0,  # Invalid: high < open
                low=41800.0,
                close=42200.0,
                volume=100.0,
            )


class TestOHLCVRepository:
    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    def test_repository_creation(self, mock_pool):
        repo = OHLCVRepository(mock_pool)
        assert repo._pool == mock_pool
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/repositories/test_ohlcv.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# app/repositories/ohlcv.py
"""Repository for OHLCV candle data."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Candle:
    """Single OHLCV candle."""

    exchange_id: str
    symbol: str
    timeframe: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        """Validate OHLCV constraints."""
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(
                f"high ({self.high}) must be >= open, close, and low"
            )
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(
                f"low ({self.low}) must be <= open, close, and high"
            )
        if self.volume < 0:
            raise ValueError(f"volume ({self.volume}) must be >= 0")


class OHLCVRepository:
    """Repository for OHLCV candle operations."""

    def __init__(self, pool):
        self._pool = pool

    async def upsert_candles(self, candles: list[Candle]) -> int:
        """Upsert candles into ohlcv_candles table.

        Returns number of rows affected.
        """
        if not candles:
            return 0

        query = """
            INSERT INTO ohlcv_candles
                (exchange_id, symbol, timeframe, ts, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (exchange_id, symbol, timeframe, ts)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        async with self._pool.acquire() as conn:
            result = await conn.executemany(
                query,
                [
                    (
                        c.exchange_id,
                        c.symbol,
                        c.timeframe,
                        c.ts,
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                    )
                    for c in candles
                ],
            )
        return len(candles)

    async def get_range(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[Candle]:
        """Get candles in a time range [start_ts, end_ts)."""
        query = """
            SELECT exchange_id, symbol, timeframe, ts, open, high, low, close, volume
            FROM ohlcv_candles
            WHERE exchange_id = $1
              AND symbol = $2
              AND timeframe = $3
              AND ts >= $4
              AND ts < $5
            ORDER BY ts ASC
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, exchange_id, symbol, timeframe, start_ts, end_ts)

        return [
            Candle(
                exchange_id=row["exchange_id"],
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                ts=row["ts"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for row in rows
        ]

    async def get_available_range(
        self, exchange_id: str, symbol: str, timeframe: str
    ) -> Optional[tuple[datetime, datetime]]:
        """Get the min and max timestamps for a symbol/timeframe."""
        query = """
            SELECT MIN(ts) as min_ts, MAX(ts) as max_ts
            FROM ohlcv_candles
            WHERE exchange_id = $1
              AND symbol = $2
              AND timeframe = $3
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, symbol, timeframe)

        if row and row["min_ts"] is not None:
            return (row["min_ts"], row["max_ts"])
        return None

    async def count_in_range(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> int:
        """Count candles in a range."""
        query = """
            SELECT COUNT(*) as cnt
            FROM ohlcv_candles
            WHERE exchange_id = $1
              AND symbol = $2
              AND timeframe = $3
              AND ts >= $4
              AND ts < $5
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, symbol, timeframe, start_ts, end_ts)

        return row["cnt"] if row else 0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/repositories/test_ohlcv.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/repositories/ohlcv.py tests/unit/repositories/test_ohlcv.py
git commit -m "feat(repo): add OHLCVRepository for candle storage

- Candle dataclass with OHLCV validation
- upsert_candles, get_range, get_available_range
- count_in_range for gap detection

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.5: Create Core Symbols Repository

**Files:**
- Create: `app/repositories/core_symbols.py`
- Test: `tests/unit/repositories/test_core_symbols.py`

**Step 1: Write the failing test**

```python
# tests/unit/repositories/test_core_symbols.py
"""Tests for core symbols repository."""
import pytest
from dataclasses import asdict
from unittest.mock import MagicMock

from app.repositories.core_symbols import CoreSymbolsRepository, CoreSymbol


class TestCoreSymbol:
    def test_core_symbol_creation(self):
        cs = CoreSymbol(
            exchange_id="kucoin",
            canonical_symbol="BTC-USDT",
            raw_symbol="BTC-USDT",
        )
        assert cs.is_enabled is True
        assert cs.timeframes == ["1m", "5m", "15m", "1h", "1d"]


class TestCoreSymbolsRepository:
    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    def test_repository_creation(self, mock_pool):
        repo = CoreSymbolsRepository(mock_pool)
        assert repo._pool == mock_pool
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/repositories/test_core_symbols.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# app/repositories/core_symbols.py
"""Repository for core symbols management."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CoreSymbol:
    """A symbol in the core universe."""

    exchange_id: str
    canonical_symbol: str
    raw_symbol: str
    timeframes: list[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"]
    )
    is_enabled: bool = True
    added_at: Optional[datetime] = None
    added_by: Optional[str] = None


class CoreSymbolsRepository:
    """Repository for core symbols operations."""

    def __init__(self, pool):
        self._pool = pool

    async def add_symbol(self, symbol: CoreSymbol) -> bool:
        """Add a symbol to the core universe. Returns True if inserted."""
        query = """
            INSERT INTO core_symbols
                (exchange_id, canonical_symbol, raw_symbol, timeframes, is_enabled, added_by)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (exchange_id, canonical_symbol) DO NOTHING
            RETURNING canonical_symbol
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                symbol.exchange_id,
                symbol.canonical_symbol,
                symbol.raw_symbol,
                symbol.timeframes,
                symbol.is_enabled,
                symbol.added_by,
            )
        return row is not None

    async def remove_symbol(self, exchange_id: str, canonical_symbol: str) -> bool:
        """Remove a symbol from core universe. Returns True if deleted."""
        query = """
            DELETE FROM core_symbols
            WHERE exchange_id = $1 AND canonical_symbol = $2
            RETURNING canonical_symbol
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, canonical_symbol)
        return row is not None

    async def set_enabled(
        self, exchange_id: str, canonical_symbol: str, enabled: bool
    ) -> bool:
        """Enable or disable a symbol. Returns True if updated."""
        query = """
            UPDATE core_symbols
            SET is_enabled = $3
            WHERE exchange_id = $1 AND canonical_symbol = $2
            RETURNING canonical_symbol
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, canonical_symbol, enabled)
        return row is not None

    async def list_symbols(
        self, exchange_id: Optional[str] = None, enabled_only: bool = True
    ) -> list[CoreSymbol]:
        """List core symbols, optionally filtered by exchange."""
        conditions = []
        params = []

        if exchange_id:
            params.append(exchange_id)
            conditions.append(f"exchange_id = ${len(params)}")

        if enabled_only:
            conditions.append("is_enabled = true")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT exchange_id, canonical_symbol, raw_symbol, timeframes,
                   is_enabled, added_at, added_by
            FROM core_symbols
            {where_clause}
            ORDER BY exchange_id, canonical_symbol
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            CoreSymbol(
                exchange_id=row["exchange_id"],
                canonical_symbol=row["canonical_symbol"],
                raw_symbol=row["raw_symbol"],
                timeframes=row["timeframes"],
                is_enabled=row["is_enabled"],
                added_at=row["added_at"],
                added_by=row["added_by"],
            )
            for row in rows
        ]

    async def get_symbol(
        self, exchange_id: str, canonical_symbol: str
    ) -> Optional[CoreSymbol]:
        """Get a single core symbol."""
        query = """
            SELECT exchange_id, canonical_symbol, raw_symbol, timeframes,
                   is_enabled, added_at, added_by
            FROM core_symbols
            WHERE exchange_id = $1 AND canonical_symbol = $2
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, canonical_symbol)

        if not row:
            return None

        return CoreSymbol(
            exchange_id=row["exchange_id"],
            canonical_symbol=row["canonical_symbol"],
            raw_symbol=row["raw_symbol"],
            timeframes=row["timeframes"],
            is_enabled=row["is_enabled"],
            added_at=row["added_at"],
            added_by=row["added_by"],
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/repositories/test_core_symbols.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/repositories/core_symbols.py tests/unit/repositories/test_core_symbols.py
git commit -m "feat(repo): add CoreSymbolsRepository

- CoreSymbol dataclass with timeframes
- add_symbol, remove_symbol, set_enabled
- list_symbols with exchange/enabled filters

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.6: Create Admin Data Router

**Files:**
- Create: `app/admin/data.py`
- Test: `tests/unit/admin/test_data.py`

**Step 1: Write the failing test**

```python
# tests/unit/admin/test_data.py
"""Tests for admin data endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


class TestCoreSymbolsEndpoints:
    @pytest.fixture
    def mock_repo(self):
        repo = AsyncMock()
        repo.list_symbols.return_value = []
        repo.add_symbol.return_value = True
        return repo

    def test_list_core_symbols_schema(self, mock_repo):
        """Test that list endpoint returns expected schema."""
        from app.admin.data import CoreSymbolResponse

        response = CoreSymbolResponse(
            exchange_id="kucoin",
            canonical_symbol="BTC-USDT",
            raw_symbol="BTC-USDT",
            timeframes=["1h", "1d"],
            is_enabled=True,
        )
        assert response.exchange_id == "kucoin"

    def test_add_core_symbol_request(self):
        """Test add symbol request model."""
        from app.admin.data import AddCoreSymbolsRequest

        req = AddCoreSymbolsRequest(
            exchange_id="kucoin",
            symbols=["BTC-USDT", "ETH-USDT"],
        )
        assert len(req.symbols) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/admin/test_data.py -v`
Expected: FAIL with import error

**Step 3: Create tests directory**

```bash
mkdir -p tests/unit/admin
touch tests/unit/admin/__init__.py
```

**Step 4: Write minimal implementation**

```python
# app/admin/data.py
"""Admin endpoints for data management."""
from typing import Literal, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.repositories.core_symbols import CoreSymbol, CoreSymbolsRepository

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/data", tags=["admin-data"])

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_core_symbols_repo() -> CoreSymbolsRepository:
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return CoreSymbolsRepository(_db_pool)


# ===========================================
# Request/Response Models
# ===========================================


class CoreSymbolResponse(BaseModel):
    """Response model for a core symbol."""

    exchange_id: str
    canonical_symbol: str
    raw_symbol: str
    timeframes: list[str]
    is_enabled: bool
    added_at: Optional[str] = None
    added_by: Optional[str] = None


class AddCoreSymbolsRequest(BaseModel):
    """Request to add symbols to core universe."""

    exchange_id: str = Field(..., description="Exchange ID (e.g., 'kucoin')")
    symbols: list[str] = Field(
        ..., min_length=1, max_length=50, description="Canonical symbols to add"
    )
    timeframes: Optional[list[str]] = Field(
        default=None, description="Timeframes to sync (defaults to all)"
    )
    added_by: Optional[str] = Field(default=None, description="Who added these symbols")


class ModifyCoreSymbolsRequest(BaseModel):
    """Request to modify core symbols."""

    action: Literal["remove", "enable", "disable"]
    exchange_id: str
    symbols: list[str] = Field(..., min_length=1, max_length=50)


class CoreSymbolsActionResponse(BaseModel):
    """Response from core symbols actions."""

    action: str
    exchange_id: str
    affected: int
    symbols: list[str]


# ===========================================
# Endpoints
# ===========================================


@router.get(
    "/core-symbols",
    response_model=list[CoreSymbolResponse],
    dependencies=[Depends(require_admin_token)],
)
async def list_core_symbols(
    exchange_id: Optional[str] = Query(None, description="Filter by exchange"),
    include_disabled: bool = Query(False, description="Include disabled symbols"),
):
    """List all core symbols."""
    repo = _get_core_symbols_repo()
    symbols = await repo.list_symbols(
        exchange_id=exchange_id, enabled_only=not include_disabled
    )

    return [
        CoreSymbolResponse(
            exchange_id=s.exchange_id,
            canonical_symbol=s.canonical_symbol,
            raw_symbol=s.raw_symbol,
            timeframes=s.timeframes,
            is_enabled=s.is_enabled,
            added_at=s.added_at.isoformat() if s.added_at else None,
            added_by=s.added_by,
        )
        for s in symbols
    ]


@router.post(
    "/core-symbols",
    response_model=CoreSymbolsActionResponse,
    dependencies=[Depends(require_admin_token)],
)
async def add_core_symbols(request: AddCoreSymbolsRequest):
    """Add symbols to the core universe."""
    repo = _get_core_symbols_repo()

    added = []
    for sym in request.symbols:
        cs = CoreSymbol(
            exchange_id=request.exchange_id,
            canonical_symbol=sym,
            raw_symbol=sym,  # Default to same as canonical
            timeframes=request.timeframes or ["1m", "5m", "15m", "1h", "1d"],
            is_enabled=True,
            added_by=request.added_by,
        )
        if await repo.add_symbol(cs):
            added.append(sym)

    logger.info(
        "core_symbols_added",
        exchange_id=request.exchange_id,
        added_count=len(added),
        requested_count=len(request.symbols),
    )

    return CoreSymbolsActionResponse(
        action="add",
        exchange_id=request.exchange_id,
        affected=len(added),
        symbols=added,
    )


@router.patch(
    "/core-symbols",
    response_model=CoreSymbolsActionResponse,
    dependencies=[Depends(require_admin_token)],
)
async def modify_core_symbols(request: ModifyCoreSymbolsRequest):
    """Modify (remove/enable/disable) core symbols."""
    repo = _get_core_symbols_repo()

    affected = []
    for sym in request.symbols:
        if request.action == "remove":
            if await repo.remove_symbol(request.exchange_id, sym):
                affected.append(sym)
        elif request.action == "enable":
            if await repo.set_enabled(request.exchange_id, sym, True):
                affected.append(sym)
        elif request.action == "disable":
            if await repo.set_enabled(request.exchange_id, sym, False):
                affected.append(sym)

    logger.info(
        "core_symbols_modified",
        action=request.action,
        exchange_id=request.exchange_id,
        affected_count=len(affected),
    )

    return CoreSymbolsActionResponse(
        action=request.action,
        exchange_id=request.exchange_id,
        affected=len(affected),
        symbols=affected,
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/unit/admin/test_data.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/admin/data.py tests/unit/admin/
git commit -m "feat(admin): add data management endpoints

- GET /admin/data/core-symbols - list symbols
- POST /admin/data/core-symbols - add symbols
- PATCH /admin/data/core-symbols - remove/enable/disable

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.7: Wire Admin Data Router to App

**Files:**
- Modify: `app/main.py`

**Step 1: Add import and include router**

Add to `app/main.py` imports:
```python
from app.admin.data import router as admin_data_router
from app.admin.data import set_db_pool as set_admin_data_pool
```

In the `lifespan` function, add after other `set_db_pool` calls:
```python
set_admin_data_pool(pool)
```

In router includes section:
```python
app.include_router(admin_data_router)
```

**Step 2: Verify app starts**

```bash
uvicorn app.main:app --port 8000 &
sleep 2
curl http://localhost:8000/health
kill %1
```

**Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat(app): wire admin data router

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## PR2: CCXT Provider + DataFetchJob

**Goal:** Implement market data fetching via CCXT and the DataFetchJob handler.

### Task 2.1: Create Market Data Provider Interface

**Files:**
- Create: `app/services/market_data/__init__.py`
- Create: `app/services/market_data/base.py`
- Test: `tests/unit/services/market_data/test_base.py`

**Step 1: Write the failing test**

```python
# tests/unit/services/market_data/test_base.py
"""Tests for market data provider interface."""
import pytest
from datetime import datetime, timezone

from app.services.market_data.base import (
    MarketDataProvider,
    MarketDataCandle,
    normalize_timeframe,
)


class TestMarketDataCandle:
    def test_candle_creation(self):
        candle = MarketDataCandle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )
        assert candle.close == 103.0


class TestNormalizeTimeframe:
    def test_standard_timeframes(self):
        assert normalize_timeframe("1m") == "1m"
        assert normalize_timeframe("5m") == "5m"
        assert normalize_timeframe("1h") == "1h"
        assert normalize_timeframe("1d") == "1d"

    def test_alternate_formats(self):
        assert normalize_timeframe("1min") == "1m"
        assert normalize_timeframe("1hour") == "1h"
        assert normalize_timeframe("1day") == "1d"
```

**Step 2: Create test directory**

```bash
mkdir -p tests/unit/services/market_data
touch tests/unit/services/market_data/__init__.py
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/unit/services/market_data/test_base.py -v`
Expected: FAIL with import error

**Step 4: Write minimal implementation**

```python
# app/services/market_data/__init__.py
"""Market data services."""
from app.services.market_data.base import (
    MarketDataProvider,
    MarketDataCandle,
    normalize_timeframe,
)

__all__ = ["MarketDataProvider", "MarketDataCandle", "normalize_timeframe"]
```

```python
# app/services/market_data/base.py
"""Base interface for market data providers."""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class MarketDataCandle:
    """Single candle from market data provider."""

    ts: datetime  # Candle close time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


# Timeframe normalization map
TIMEFRAME_MAP = {
    "1m": "1m",
    "1min": "1m",
    "5m": "5m",
    "5min": "5m",
    "15m": "15m",
    "15min": "15m",
    "1h": "1h",
    "1hour": "1h",
    "60m": "1h",
    "1d": "1d",
    "1day": "1d",
    "24h": "1d",
}


def normalize_timeframe(tf: str) -> str:
    """Normalize timeframe to canonical format."""
    return TIMEFRAME_MAP.get(tf.lower(), tf)


class MarketDataProvider(Protocol):
    """Protocol for market data providers."""

    @property
    def exchange_id(self) -> str:
        """Get the exchange identifier."""
        ...

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[MarketDataCandle]:
        """Fetch OHLCV data for a symbol and time range.

        Args:
            symbol: Canonical symbol (e.g., 'BTC-USDT')
            timeframe: Canonical timeframe ('1m', '5m', '15m', '1h', '1d')
            start_ts: Range start (inclusive), UTC
            end_ts: Range end (exclusive), UTC

        Returns:
            List of candles, sorted by timestamp ascending
        """
        ...

    def normalize_symbol(self, raw: str) -> str:
        """Convert exchange-specific symbol to canonical format."""
        ...

    def exchange_symbol(self, canonical: str) -> str:
        """Convert canonical symbol to exchange-specific format."""
        ...
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/unit/services/market_data/test_base.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/services/market_data/ tests/unit/services/market_data/
git commit -m "feat(market_data): add provider interface

- MarketDataProvider protocol
- MarketDataCandle dataclass
- normalize_timeframe helper

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: Create CCXT Provider Implementation

**Files:**
- Create: `app/services/market_data/ccxt_provider.py`
- Test: `tests/unit/services/market_data/test_ccxt_provider.py`

**Step 1: Write the failing test**

```python
# tests/unit/services/market_data/test_ccxt_provider.py
"""Tests for CCXT market data provider."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.market_data.ccxt_provider import CcxtMarketDataProvider


class TestCcxtProvider:
    def test_exchange_id(self):
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.exchange_id == "kucoin"

    def test_normalize_symbol(self):
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.normalize_symbol("BTC/USDT") == "BTC-USDT"
        assert provider.normalize_symbol("ETH/USDT") == "ETH-USDT"

    def test_exchange_symbol(self):
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.exchange_symbol("BTC-USDT") == "BTC/USDT"
        assert provider.exchange_symbol("ETH-USDT") == "ETH/USDT"

    def test_canonical_timeframe(self):
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.canonical_timeframe("1m") == "1m"
        assert provider.canonical_timeframe("1h") == "1h"

    def test_ccxt_timeframe(self):
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.ccxt_timeframe("1m") == "1m"
        assert provider.ccxt_timeframe("1h") == "1h"
        assert provider.ccxt_timeframe("1d") == "1d"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/services/market_data/test_ccxt_provider.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# app/services/market_data/ccxt_provider.py
"""CCXT-based market data provider."""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt.async_support as ccxt
import structlog

from app.config import get_settings
from app.services.market_data.base import MarketDataCandle, normalize_timeframe

logger = structlog.get_logger(__name__)

# Timeframe to milliseconds
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


class CcxtMarketDataProvider:
    """Market data provider using CCXT library."""

    def __init__(
        self,
        exchange_id: str,
        rate_limit_ms: Optional[int] = None,
    ):
        """Initialize provider for an exchange.

        Args:
            exchange_id: CCXT exchange ID (e.g., 'kucoin', 'binance')
            rate_limit_ms: Override rate limit in milliseconds
        """
        self._exchange_id = exchange_id
        settings = get_settings()
        self._rate_limit_ms = rate_limit_ms or settings.ccxt_rate_limit_ms
        self._exchange: Optional[ccxt.Exchange] = None
        self._last_request_time: float = 0

    @property
    def exchange_id(self) -> str:
        """Get the exchange identifier."""
        return self._exchange_id

    def _get_exchange(self) -> ccxt.Exchange:
        """Get or create the CCXT exchange instance."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "rateLimit": self._rate_limit_ms,
                }
            )
        return self._exchange

    async def close(self):
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    def normalize_symbol(self, raw: str) -> str:
        """Convert exchange symbol (BTC/USDT) to canonical (BTC-USDT)."""
        return raw.replace("/", "-")

    def exchange_symbol(self, canonical: str) -> str:
        """Convert canonical symbol (BTC-USDT) to exchange (BTC/USDT)."""
        return canonical.replace("-", "/")

    def canonical_timeframe(self, tf: str) -> str:
        """Normalize timeframe to canonical format."""
        return normalize_timeframe(tf)

    def ccxt_timeframe(self, canonical_tf: str) -> str:
        """Convert canonical timeframe to CCXT format."""
        # CCXT uses same format as our canonical for most exchanges
        return canonical_tf

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        import time

        now = time.time() * 1000
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_ms:
            await asyncio.sleep((self._rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[MarketDataCandle]:
        """Fetch OHLCV data for a symbol and time range.

        Handles pagination automatically for large ranges.
        """
        exchange = self._get_exchange()
        ccxt_symbol = self.exchange_symbol(symbol)
        ccxt_tf = self.ccxt_timeframe(timeframe)

        # Convert to milliseconds
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        tf_ms = TIMEFRAME_MS.get(timeframe, 60000)
        limit = 1000  # Max candles per request (varies by exchange)

        all_candles: list[MarketDataCandle] = []
        since = start_ms

        log = logger.bind(
            exchange=self._exchange_id,
            symbol=symbol,
            timeframe=timeframe,
        )

        while since < end_ms:
            await self._rate_limit()

            try:
                ohlcv = await exchange.fetch_ohlcv(
                    ccxt_symbol, ccxt_tf, since=since, limit=limit
                )
            except Exception as e:
                log.error("ccxt_fetch_failed", error=str(e), since=since)
                raise

            if not ohlcv:
                break

            for candle in ohlcv:
                ts_ms, o, h, l, c, v = candle
                # ts is candle open time in CCXT, we want close time
                close_ts_ms = ts_ms + tf_ms

                # Stop if past end
                if close_ts_ms > end_ms:
                    break

                all_candles.append(
                    MarketDataCandle(
                        ts=datetime.fromtimestamp(close_ts_ms / 1000, tz=timezone.utc),
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v) if v else 0.0,
                    )
                )

            # Move cursor to next batch
            last_ts = ohlcv[-1][0]
            if last_ts <= since:
                # No progress, stop
                break
            since = last_ts + tf_ms

            log.debug(
                "ccxt_batch_fetched",
                batch_size=len(ohlcv),
                total_candles=len(all_candles),
            )

        log.info(
            "ccxt_fetch_complete",
            total_candles=len(all_candles),
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
        )

        return all_candles
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/services/market_data/test_ccxt_provider.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/market_data/ccxt_provider.py tests/unit/services/market_data/test_ccxt_provider.py
git commit -m "feat(market_data): add CCXT provider implementation

- Symbol and timeframe normalization
- Rate limiting between requests
- Pagination for large ranges
- Candle close time alignment

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.3: Create Data Revision Service

**Files:**
- Create: `app/services/market_data/revision.py`
- Test: `tests/unit/services/market_data/test_revision.py`

**Step 1: Write the failing test**

```python
# tests/unit/services/market_data/test_revision.py
"""Tests for data revision computation."""
import pytest
from datetime import datetime, timezone

from app.services.market_data.revision import compute_checksum
from app.repositories.ohlcv import Candle


class TestComputeChecksum:
    def test_empty_candles(self):
        checksum = compute_checksum([])
        assert checksum == "empty"

    def test_single_candle(self):
        candle = Candle(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=100.0,
        )
        checksum = compute_checksum([candle])
        assert len(checksum) == 16
        assert checksum.isalnum()

    def test_deterministic(self):
        candles = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                open=42000.0 + i,
                high=42500.0 + i,
                low=41800.0 + i,
                close=42200.0 + i,
                volume=100.0 + i,
            )
            for i in range(5)
        ]
        checksum1 = compute_checksum(candles)
        checksum2 = compute_checksum(candles)
        assert checksum1 == checksum2

    def test_different_data_different_checksum(self):
        candles1 = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open=42000.0,
                high=42500.0,
                low=41800.0,
                close=42200.0,
                volume=100.0,
            )
        ]
        candles2 = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open=42000.0,
                high=42500.0,
                low=41800.0,
                close=42201.0,  # Different close
                volume=100.0,
            )
        ]
        assert compute_checksum(candles1) != compute_checksum(candles2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/services/market_data/test_revision.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# app/services/market_data/revision.py
"""Data revision computation for drift detection."""
import hashlib
from datetime import datetime
from typing import Sequence

from app.repositories.ohlcv import Candle


def compute_checksum(candles: Sequence[Candle]) -> str:
    """Compute a deterministic checksum from candle data.

    Algorithm:
    - Sample first 10 + last 10 + every 1000th candle
    - Format: ISO timestamp + OHLCV with 10 decimal places
    - SHA256 truncated to 16 chars
    """
    if not candles:
        return "empty"

    # Sampling logic
    n = len(candles)
    sample_indices = set()

    # First 10
    for i in range(min(10, n)):
        sample_indices.add(i)

    # Last 10
    for i in range(max(0, n - 10), n):
        sample_indices.add(i)

    # Every 1000th
    for i in range(0, n, 1000):
        sample_indices.add(i)

    # Sort for determinism
    sample_indices = sorted(sample_indices)

    # Build content string
    content_parts = []
    for idx in sample_indices:
        c = candles[idx]
        # Format with 10 decimal places for precision
        part = (
            f"{c.ts.isoformat()}|"
            f"{c.open:.10f}|{c.high:.10f}|{c.low:.10f}|{c.close:.10f}|{c.volume:.10f}"
        )
        content_parts.append(part)

    content = "\n".join(content_parts)

    # Hash and truncate
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return h[:16]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/services/market_data/test_revision.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/market_data/revision.py tests/unit/services/market_data/test_revision.py
git commit -m "feat(market_data): add revision checksum computation

- Samples first/last 10 + every 1000th candle
- SHA256 truncated to 16 chars
- Deterministic for drift detection

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

**[CONTINUED IN PR3-PR8...]**

*The plan continues with the remaining PRs following the same TDD pattern. Each PR builds on the previous, maintaining the incremental development approach with frequent commits.*

---

## Summary: PR Dependency Graph

```
PR0 (Scaffolding) ──┬──► PR1 (Data Layer)
                    │         │
                    │         ▼
                    └──► PR2 (CCXT + DataFetch) ──► PR3 (Job Queue)
                                                        │
                                                        ▼
                                                   PR4 (DataSync)
                                                        │
                                                        ▼
                                                   PR5 (TuneJob)
                                                        │
                                                        ▼
                                                   PR6 (Artifacts)
                                                        │
                                                        ▼
                                                   PR7 (WFO)
                                                        │
                                                        ▼
                                                   PR8 (Ops Polish)
```

## Test Commands Quick Reference

```bash
# Run specific PR tests
pytest tests/unit/jobs/ -v                    # PR0
pytest tests/unit/repositories/test_ohlcv.py  # PR1
pytest tests/unit/services/market_data/ -v    # PR2

# Run all tests
pytest tests/unit/ -v --tb=short

# Lint
black app/ tests/ && flake8 app/ tests/ --max-line-length=100

# Type check
mypy app/ --ignore-missing-imports
```
