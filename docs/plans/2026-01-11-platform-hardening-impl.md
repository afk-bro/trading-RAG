# Phase 1 Platform Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add run plan idempotency, event retention with rollups, and Supabase auth wiring.

**Architecture:** Three independent features implemented sequentially. Idempotency uses dual-key approach (client key + request hash). Retention uses tiered deletion with daily rollup aggregation. Auth integrates Supabase JWT validation with workspace membership checks.

**Tech Stack:** Python/FastAPI, asyncpg, Supabase Auth, PostgreSQL

---

## Feature 1: Run Plan Idempotency

### Task 1.1: Add idempotency columns migration

**Files:**
- Create: `migrations/039_run_plans_idempotency.sql`

**Step 1: Write the migration**

```sql
-- migrations/039_run_plans_idempotency.sql
-- Add idempotency support to run_plans table

ALTER TABLE run_plans
    ADD COLUMN IF NOT EXISTS idempotency_key TEXT,
    ADD COLUMN IF NOT EXISTS request_hash TEXT;

-- Unique constraint on idempotency_key (when provided)
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_plans_idempotency_key
    ON run_plans(idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Index for request_hash lookups
CREATE INDEX IF NOT EXISTS idx_run_plans_request_hash
    ON run_plans(request_hash)
    WHERE request_hash IS NOT NULL;
```

**Step 2: Apply migration via Supabase MCP**

Run: `mcp__supabase__apply_migration` with name `run_plans_idempotency` and the SQL above.

**Step 3: Verify columns exist**

Run: `mcp__supabase__execute_sql` with:
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'run_plans' AND column_name IN ('idempotency_key', 'request_hash');
```
Expected: 2 rows returned

**Step 4: Commit**

```bash
git add migrations/039_run_plans_idempotency.sql
git commit -m "feat(db): add idempotency columns to run_plans"
```

---

### Task 1.2: Add request hash utility

**Files:**
- Create: `app/services/testing/idempotency.py`
- Test: `tests/unit/test_idempotency.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_idempotency.py
"""Tests for run plan idempotency utilities."""

import pytest
from uuid import UUID

from app.services.testing.idempotency import compute_request_hash


class TestComputeRequestHash:
    """Tests for request hash computation."""

    def test_deterministic_for_same_input(self):
        """Same input always produces same hash."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")
        plan = {"inputs": {"symbol": "BTC"}, "resolved": {}}

        hash1 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )
        hash2 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256 truncated to 32 chars

    def test_different_for_different_input(self):
        """Different inputs produce different hashes."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")
        plan = {"inputs": {"symbol": "BTC"}, "resolved": {}}

        hash1 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )
        hash2 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="calmar",  # Different objective
            plan=plan,
        )

        assert hash1 != hash2

    def test_key_order_does_not_affect_hash(self):
        """Dict key order doesn't change hash (canonical JSON)."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")

        plan1 = {"a": 1, "b": 2}
        plan2 = {"b": 2, "a": 1}

        hash1 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan1,
        )
        hash2 = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan2,
        )

        assert hash1 == hash2

    def test_includes_strategy_entity_id_when_provided(self):
        """Strategy entity ID affects hash when provided."""
        workspace_id = UUID("12345678-1234-5678-1234-567812345678")
        strategy_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        plan = {"inputs": {}}

        hash_without = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            plan=plan,
        )
        hash_with = compute_request_hash(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
            objective_name="sharpe",
            plan=plan,
        )

        assert hash_without != hash_with
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_idempotency.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.testing.idempotency'`

**Step 3: Write minimal implementation**

```python
# app/services/testing/idempotency.py
"""Idempotency utilities for run plans."""

import hashlib
import json
from typing import Any, Optional
from uuid import UUID


def compute_request_hash(
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID],
    objective_name: str,
    plan: dict[str, Any],
) -> str:
    """
    Compute canonical hash of run plan request for duplicate detection.

    Args:
        workspace_id: Workspace ID
        strategy_entity_id: Optional strategy entity ID
        objective_name: Objective function name
        plan: Full plan dict

    Returns:
        32-character hex hash (SHA256 truncated)
    """
    canonical = {
        "workspace_id": str(workspace_id),
        "strategy_entity_id": str(strategy_entity_id) if strategy_entity_id else None,
        "objective_name": objective_name,
        "plan": plan,
    }
    # Sort keys recursively for determinism
    json_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:32]
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_idempotency.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add app/services/testing/idempotency.py tests/unit/test_idempotency.py
git commit -m "feat(testing): add request hash utility for idempotency"
```

---

### Task 1.3: Update repository with idempotency support

**Files:**
- Modify: `app/repositories/run_plans.py`
- Test: `tests/unit/test_run_plans_repo.py`

**Step 1: Write the failing tests**

Create new test file or add to existing:

```python
# tests/unit/test_run_plans_repo.py
"""Tests for run_plans repository idempotency."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.repositories.run_plans import RunPlansRepository


class TestRunPlansIdempotency:
    """Tests for idempotency key handling."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_get_by_idempotency_key_found(self, mock_pool):
        """Returns plan when idempotency key exists."""
        pool, conn = mock_pool
        plan_id = uuid4()
        conn.fetchrow.return_value = {
            "id": plan_id,
            "status": "pending",
            "idempotency_key": "test-key-123",
        }

        repo = RunPlansRepository(pool)
        result = await repo.get_by_idempotency_key("test-key-123")

        assert result is not None
        assert result["id"] == plan_id
        conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_idempotency_key_not_found(self, mock_pool):
        """Returns None when idempotency key doesn't exist."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        repo = RunPlansRepository(pool)
        result = await repo.get_by_idempotency_key("nonexistent-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_request_hash_found(self, mock_pool):
        """Returns plan when request hash exists."""
        pool, conn = mock_pool
        plan_id = uuid4()
        conn.fetchrow.return_value = {
            "id": plan_id,
            "status": "pending",
            "request_hash": "abc123",
        }

        repo = RunPlansRepository(pool)
        result = await repo.get_by_request_hash("abc123")

        assert result is not None
        assert result["id"] == plan_id

    @pytest.mark.asyncio
    async def test_create_run_plan_with_idempotency(self, mock_pool):
        """Creates plan with idempotency key and request hash."""
        pool, conn = mock_pool
        plan_id = uuid4()
        workspace_id = uuid4()
        conn.fetchval.return_value = plan_id

        repo = RunPlansRepository(pool)
        result = await repo.create_run_plan(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            n_variants=5,
            plan={"test": True},
            status="pending",
            idempotency_key="client-key-456",
            request_hash="hash789",
        )

        assert result == plan_id
        # Verify query includes new columns
        call_args = conn.fetchval.call_args
        assert "idempotency_key" in call_args[0][0]
        assert "request_hash" in call_args[0][0]
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_run_plans_repo.py -v`

Expected: FAIL with `AttributeError: 'RunPlansRepository' object has no attribute 'get_by_idempotency_key'`

**Step 3: Update repository implementation**

Add to `app/repositories/run_plans.py`:

```python
# Add these methods to RunPlansRepository class

    async def get_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[dict[str, Any]]:
        """Get a run plan by idempotency key."""
        query = """
            SELECT id, status, idempotency_key, request_hash,
                   workspace_id, strategy_entity_id, objective_name,
                   created_at
            FROM run_plans
            WHERE idempotency_key = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, idempotency_key)
            if not row:
                return None
            return dict(row)

    async def get_by_request_hash(
        self, request_hash: str
    ) -> Optional[dict[str, Any]]:
        """Get a run plan by request hash."""
        query = """
            SELECT id, status, idempotency_key, request_hash,
                   workspace_id, strategy_entity_id, objective_name,
                   created_at
            FROM run_plans
            WHERE request_hash = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, request_hash)
            if not row:
                return None
            return dict(row)
```

Also update `create_run_plan` method signature and query:

```python
    async def create_run_plan(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        objective_name: str,
        n_variants: int,
        plan: dict[str, Any],
        status: str = "pending",
        idempotency_key: Optional[str] = None,
        request_hash: Optional[str] = None,
    ) -> UUID:
        """
        Create a new run plan record.

        Args:
            workspace_id: Workspace ID
            strategy_entity_id: Optional strategy entity ID
            objective_name: Objective function name
            n_variants: Number of variants in plan
            plan: Full plan JSON (inputs, resolved, provenance)
            status: Initial status (default: pending)
            idempotency_key: Optional client-provided idempotency key
            request_hash: Optional server-computed request hash

        Returns:
            The new run plan ID
        """
        query = """
            INSERT INTO run_plans (
                workspace_id, strategy_entity_id, objective_name,
                n_variants, plan, status, idempotency_key, request_hash
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            plan_id = await conn.fetchval(
                query,
                workspace_id,
                strategy_entity_id,
                objective_name,
                n_variants,
                json.dumps(plan),
                status,
                idempotency_key,
                request_hash,
            )

        logger.info(
            "Created run plan",
            plan_id=str(plan_id),
            workspace_id=str(workspace_id),
            n_variants=n_variants,
            has_idempotency_key=idempotency_key is not None,
        )

        return plan_id
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_run_plans_repo.py -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add app/repositories/run_plans.py tests/unit/test_run_plans_repo.py
git commit -m "feat(repo): add idempotency key support to run_plans"
```

---

### Task 1.4: Add idempotency handling to router

**Files:**
- Modify: `app/routers/testing.py` (or wherever run plan creation endpoint lives)
- Test: `tests/unit/routers/test_testing_idempotency.py`

**Step 1: Write the failing tests**

```python
# tests/unit/routers/test_testing_idempotency.py
"""Tests for run plan creation idempotency."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import HTTPException


class TestRunPlanIdempotency:
    """Tests for idempotency handling in run plan creation."""

    @pytest.mark.asyncio
    async def test_returns_existing_on_idempotency_key_match(self):
        """Returns existing plan when idempotency key matches."""
        from app.routers.testing import create_run_plan_with_idempotency

        existing_plan = {
            "id": uuid4(),
            "status": "pending",
            "idempotency_key": "test-key",
        }

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = existing_plan

        result = await create_run_plan_with_idempotency(
            workspace_id=uuid4(),
            strategy_entity_id=None,
            objective_name="sharpe",
            plan={"test": True},
            idempotency_key="test-key",
            repo=mock_repo,
        )

        assert result["id"] == existing_plan["id"]
        assert result["status"] == "existing"
        mock_repo.create_run_plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_409_on_idempotency_key_match_non_pending(self):
        """Returns 409 when idempotency key matches non-pending plan."""
        from app.routers.testing import create_run_plan_with_idempotency

        existing_plan = {
            "id": uuid4(),
            "status": "running",  # Not pending
            "idempotency_key": "test-key",
        }

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = existing_plan

        with pytest.raises(HTTPException) as exc_info:
            await create_run_plan_with_idempotency(
                workspace_id=uuid4(),
                strategy_entity_id=None,
                objective_name="sharpe",
                plan={"test": True},
                idempotency_key="test-key",
                repo=mock_repo,
            )

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_409_on_request_hash_match(self):
        """Returns 409 when request hash matches (duplicate request)."""
        from app.routers.testing import create_run_plan_with_idempotency

        existing_plan = {
            "id": uuid4(),
            "status": "pending",
            "request_hash": "abc123",
        }

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = None
        mock_repo.get_by_request_hash.return_value = existing_plan

        with pytest.raises(HTTPException) as exc_info:
            await create_run_plan_with_idempotency(
                workspace_id=uuid4(),
                strategy_entity_id=None,
                objective_name="sharpe",
                plan={"test": True},
                idempotency_key=None,
                repo=mock_repo,
            )

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_creates_new_plan_when_no_match(self):
        """Creates new plan when no idempotency or hash match."""
        from app.routers.testing import create_run_plan_with_idempotency

        new_plan_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = None
        mock_repo.get_by_request_hash.return_value = None
        mock_repo.create_run_plan.return_value = new_plan_id

        result = await create_run_plan_with_idempotency(
            workspace_id=uuid4(),
            strategy_entity_id=None,
            objective_name="sharpe",
            plan={"test": True},
            idempotency_key="new-key",
            repo=mock_repo,
        )

        assert result["id"] == new_plan_id
        assert result["status"] == "created"
        mock_repo.create_run_plan.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/routers/test_testing_idempotency.py -v`

Expected: FAIL with import error

**Step 3: Write implementation**

Add to `app/routers/testing.py` (or create service function):

```python
# app/services/testing/idempotency.py (add to existing file)

from fastapi import HTTPException
from typing import Any, Optional
from uuid import UUID

from app.repositories.run_plans import RunPlansRepository


async def create_run_plan_with_idempotency(
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID],
    objective_name: str,
    plan: dict[str, Any],
    idempotency_key: Optional[str],
    repo: RunPlansRepository,
    n_variants: int = 0,
) -> dict[str, Any]:
    """
    Create run plan with idempotency handling.

    Args:
        workspace_id: Workspace ID
        strategy_entity_id: Optional strategy entity ID
        objective_name: Objective function name
        plan: Full plan dict
        idempotency_key: Optional client-provided key
        repo: Run plans repository
        n_variants: Number of variants

    Returns:
        Dict with id and status ("created" or "existing")

    Raises:
        HTTPException 409: If duplicate detected
    """
    # Check for existing by idempotency key
    if idempotency_key:
        existing = await repo.get_by_idempotency_key(idempotency_key)
        if existing:
            if existing["status"] != "pending":
                raise HTTPException(
                    status_code=409,
                    detail=f"Plan {existing['id']} already {existing['status']}",
                )
            return {"id": existing["id"], "status": "existing"}

    # Compute request hash
    request_hash = compute_request_hash(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        objective_name=objective_name,
        plan=plan,
    )

    # Check for existing by request hash
    existing_by_hash = await repo.get_by_request_hash(request_hash)
    if existing_by_hash:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate request (plan {existing_by_hash['id']})",
        )

    # Create new plan
    plan_id = await repo.create_run_plan(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        objective_name=objective_name,
        n_variants=n_variants,
        plan=plan,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )

    return {"id": plan_id, "status": "created"}
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/routers/test_testing_idempotency.py -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add app/services/testing/idempotency.py tests/unit/routers/test_testing_idempotency.py
git commit -m "feat(testing): add idempotency handling for run plan creation"
```

---

## Feature 2: Event Retention

### Task 2.1: Add severity and pinned columns migration

**Files:**
- Create: `migrations/040_trade_events_retention.sql`

**Step 1: Write the migration**

```sql
-- migrations/040_trade_events_retention.sql
-- Add retention support columns to trade_events

-- Add severity column with default
ALTER TABLE trade_events
    ADD COLUMN IF NOT EXISTS severity TEXT NOT NULL DEFAULT 'info';

-- Add check constraint for severity values
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'trade_events_severity_check'
    ) THEN
        ALTER TABLE trade_events
            ADD CONSTRAINT trade_events_severity_check
            CHECK (severity IN ('debug', 'info', 'warn', 'error'));
    END IF;
END $$;

-- Add pinned column
ALTER TABLE trade_events
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;

-- Index for retention queries
CREATE INDEX IF NOT EXISTS idx_trade_events_retention
    ON trade_events(created_at, severity, pinned);
```

**Step 2: Apply migration**

Run: `mcp__supabase__apply_migration` with name `trade_events_retention`

**Step 3: Verify columns exist**

Run: `mcp__supabase__execute_sql`:
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'trade_events' AND column_name IN ('severity', 'pinned');
```
Expected: 2 rows

**Step 4: Commit**

```bash
git add migrations/040_trade_events_retention.sql
git commit -m "feat(db): add severity and pinned columns to trade_events"
```

---

### Task 2.2: Create rollup table migration

**Files:**
- Create: `migrations/041_trade_event_rollups.sql`

**Step 1: Write the migration**

```sql
-- migrations/041_trade_event_rollups.sql
-- Daily rollup table for historical analytics

CREATE TABLE IF NOT EXISTS trade_event_rollups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID REFERENCES kb_entities(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    rollup_date DATE NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    sample_correlation_ids TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_rollup_key UNIQUE (workspace_id, strategy_entity_id, event_type, rollup_date)
);

CREATE INDEX IF NOT EXISTS idx_rollups_workspace_date
    ON trade_event_rollups(workspace_id, rollup_date DESC);

CREATE INDEX IF NOT EXISTS idx_rollups_strategy
    ON trade_event_rollups(strategy_entity_id, rollup_date DESC)
    WHERE strategy_entity_id IS NOT NULL;
```

**Step 2: Apply migration**

Run: `mcp__supabase__apply_migration` with name `trade_event_rollups`

**Step 3: Verify table exists**

Run: `mcp__supabase__list_tables` with schemas `["public"]`

Expected: `trade_event_rollups` in list

**Step 4: Commit**

```bash
git add migrations/041_trade_event_rollups.sql
git commit -m "feat(db): add trade_event_rollups table"
```

---

### Task 2.3: Create rollup repository

**Files:**
- Create: `app/repositories/event_rollups.py`
- Test: `tests/unit/test_event_rollups_repo.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_event_rollups_repo.py
"""Tests for event rollups repository."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.repositories.event_rollups import EventRollupsRepository


class TestEventRollupsRepository:
    """Tests for rollup operations."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_run_daily_rollup(self, mock_pool):
        """Aggregates events into rollups."""
        pool, conn = mock_pool
        conn.execute.return_value = "INSERT 0 5"

        repo = EventRollupsRepository(pool)
        count = await repo.run_daily_rollup(date(2026, 1, 10))

        assert count == 5
        conn.execute.assert_called_once()
        # Verify query includes ON CONFLICT
        query = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in query

    @pytest.mark.asyncio
    async def test_get_rollups_for_workspace(self, mock_pool):
        """Returns rollups for workspace in date range."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        conn.fetch.return_value = [
            {
                "event_type": "ORDER_FILLED",
                "rollup_date": date(2026, 1, 10),
                "event_count": 50,
                "error_count": 2,
            }
        ]

        repo = EventRollupsRepository(pool)
        rollups = await repo.get_rollups(
            workspace_id=workspace_id,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )

        assert len(rollups) == 1
        assert rollups[0]["event_type"] == "ORDER_FILLED"
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_event_rollups_repo.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# app/repositories/event_rollups.py
"""Repository for trade event rollups."""

from datetime import date
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class EventRollupsRepository:
    """Repository for event rollup operations."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def run_daily_rollup(self, target_date: date) -> int:
        """
        Aggregate events from target_date into rollups.

        Idempotent via ON CONFLICT - safe to run multiple times.

        Args:
            target_date: Date to aggregate

        Returns:
            Number of rollup rows upserted
        """
        query = """
            INSERT INTO trade_event_rollups (
                workspace_id, strategy_entity_id, event_type, rollup_date,
                event_count, error_count, sample_correlation_ids
            )
            SELECT
                workspace_id,
                strategy_entity_id,
                event_type,
                $1::date as rollup_date,
                COUNT(*) as event_count,
                COUNT(*) FILTER (WHERE severity = 'error') as error_count,
                (ARRAY_AGG(DISTINCT correlation_id)
                    FILTER (WHERE correlation_id IS NOT NULL))[1:5]
            FROM trade_events
            WHERE created_at >= $1::date
              AND created_at < ($1::date + INTERVAL '1 day')
            GROUP BY workspace_id, strategy_entity_id, event_type
            ON CONFLICT (workspace_id, strategy_entity_id, event_type, rollup_date)
            DO UPDATE SET
                event_count = EXCLUDED.event_count,
                error_count = EXCLUDED.error_count,
                sample_correlation_ids = EXCLUDED.sample_correlation_ids
        """

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, target_date)

        # Parse "INSERT 0 N" or "UPDATE N"
        count = int(result.split()[-1])
        logger.info("Daily rollup complete", date=str(target_date), rows=count)
        return count

    async def get_rollups(
        self,
        workspace_id: UUID,
        start_date: date,
        end_date: date,
        strategy_entity_id: Optional[UUID] = None,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get rollups for a workspace in date range.

        Args:
            workspace_id: Workspace ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            strategy_entity_id: Optional filter by strategy
            event_type: Optional filter by event type

        Returns:
            List of rollup records
        """
        conditions = ["workspace_id = $1", "rollup_date >= $2", "rollup_date <= $3"]
        params: list[Any] = [workspace_id, start_date, end_date]
        param_idx = 4

        if strategy_entity_id:
            conditions.append(f"strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        where = " AND ".join(conditions)
        query = f"""
            SELECT
                id, workspace_id, strategy_entity_id, event_type,
                rollup_date, event_count, error_count, sample_correlation_ids,
                created_at
            FROM trade_event_rollups
            WHERE {where}
            ORDER BY rollup_date DESC, event_type
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_event_rollups_repo.py -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add app/repositories/event_rollups.py tests/unit/test_event_rollups_repo.py
git commit -m "feat(repo): add event rollups repository"
```

---

### Task 2.4: Create retention cleanup service

**Files:**
- Create: `app/services/retention.py`
- Test: `tests/unit/test_retention.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_retention.py
"""Tests for event retention service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.services.retention import RetentionService


class TestRetentionService:
    """Tests for retention cleanup."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_cleanup_respects_severity_tiers(self, mock_pool):
        """Deletes info/debug at 30 days, warn/error at 90 days."""
        pool, conn = mock_pool
        conn.execute.side_effect = ["DELETE 100", "DELETE 50"]

        service = RetentionService(pool)
        result = await service.run_cleanup()

        assert result["info_debug_deleted"] == 100
        assert result["warn_error_deleted"] == 50
        assert conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_preserves_pinned(self, mock_pool):
        """Pinned events are never deleted."""
        pool, conn = mock_pool
        conn.execute.side_effect = ["DELETE 0", "DELETE 0"]

        service = RetentionService(pool)
        await service.run_cleanup()

        # Verify both queries include pinned = FALSE
        for call in conn.execute.call_args_list:
            query = call[0][0]
            assert "pinned = FALSE" in query
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_retention.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# app/services/retention.py
"""Event retention and cleanup service."""

from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)


class RetentionService:
    """Service for managing event retention."""

    # Retention periods
    INFO_DEBUG_DAYS = 30
    WARN_ERROR_DAYS = 90

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def run_cleanup(self) -> dict[str, int]:
        """
        Delete expired events based on severity tier.

        Retention policy:
        - INFO/DEBUG: 30 days
        - WARN/ERROR: 90 days
        - Pinned: Never deleted

        Returns:
            Dict with counts of deleted events per tier
        """
        now = datetime.utcnow()

        async with self.pool.acquire() as conn:
            # Delete INFO/DEBUG older than 30 days (not pinned)
            info_cutoff = now - timedelta(days=self.INFO_DEBUG_DAYS)
            info_result = await conn.execute(
                """
                DELETE FROM trade_events
                WHERE created_at < $1
                  AND severity IN ('debug', 'info')
                  AND pinned = FALSE
                """,
                info_cutoff,
            )

            # Delete WARN/ERROR older than 90 days (not pinned)
            error_cutoff = now - timedelta(days=self.WARN_ERROR_DAYS)
            error_result = await conn.execute(
                """
                DELETE FROM trade_events
                WHERE created_at < $1
                  AND severity IN ('warn', 'error')
                  AND pinned = FALSE
                """,
                error_cutoff,
            )

        info_deleted = int(info_result.split()[-1])
        error_deleted = int(error_result.split()[-1])

        logger.info(
            "Retention cleanup complete",
            info_debug_deleted=info_deleted,
            warn_error_deleted=error_deleted,
        )

        return {
            "info_debug_deleted": info_deleted,
            "warn_error_deleted": error_deleted,
        }
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_retention.py -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add app/services/retention.py tests/unit/test_retention.py
git commit -m "feat(services): add event retention cleanup service"
```

---

### Task 2.5: Add admin endpoints for retention jobs

**Files:**
- Modify: `app/admin/router.py`
- Test: `tests/unit/admin/test_retention_endpoints.py`

**Step 1: Write the failing test**

```python
# tests/unit/admin/test_retention_endpoints.py
"""Tests for retention admin endpoints."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


class TestRetentionEndpoints:
    """Tests for /admin/jobs/* endpoints."""

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_rollup_requires_admin_token(self, client):
        """Rollup endpoint requires admin auth."""
        response = client.post("/admin/jobs/rollup-events")
        assert response.status_code == 401

    def test_cleanup_requires_admin_token(self, client):
        """Cleanup endpoint requires admin auth."""
        response = client.post("/admin/jobs/cleanup-events")
        assert response.status_code == 401

    @patch("app.admin.router.EventRollupsRepository")
    def test_rollup_runs_for_yesterday(self, mock_repo_class, client):
        """Rollup defaults to yesterday."""
        mock_repo = AsyncMock()
        mock_repo.run_daily_rollup.return_value = 10
        mock_repo_class.return_value = mock_repo

        response = client.post(
            "/admin/jobs/rollup-events",
            headers={"X-Admin-Token": "test-token"},
        )

        assert response.status_code == 200
        assert response.json()["rows_affected"] == 10

    @patch("app.admin.router.RetentionService")
    def test_cleanup_returns_counts(self, mock_service_class, client):
        """Cleanup returns deletion counts."""
        mock_service = AsyncMock()
        mock_service.run_cleanup.return_value = {
            "info_debug_deleted": 100,
            "warn_error_deleted": 50,
        }
        mock_service_class.return_value = mock_service

        response = client.post(
            "/admin/jobs/cleanup-events",
            headers={"X-Admin-Token": "test-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["info_debug_deleted"] == 100
        assert data["warn_error_deleted"] == 50
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/admin/test_retention_endpoints.py -v`

Expected: FAIL (404 or import error)

**Step 3: Add endpoints to admin router**

Add to `app/admin/router.py`:

```python
from datetime import date, timedelta
from app.repositories.event_rollups import EventRollupsRepository
from app.services.retention import RetentionService


@router.post("/jobs/rollup-events")
async def run_rollup_job(
    target_date: Optional[date] = None,
    pool=Depends(get_pool),
    _=Depends(require_admin_token),
):
    """
    Run daily event rollup job.

    Defaults to yesterday if no date provided.
    Idempotent - safe to run multiple times.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    repo = EventRollupsRepository(pool)
    count = await repo.run_daily_rollup(target_date)

    return {
        "status": "ok",
        "target_date": str(target_date),
        "rows_affected": count,
    }


@router.post("/jobs/cleanup-events")
async def run_cleanup_job(
    pool=Depends(get_pool),
    _=Depends(require_admin_token),
):
    """
    Run event retention cleanup job.

    Deletes expired events based on severity tier:
    - INFO/DEBUG: 30 days
    - WARN/ERROR: 90 days
    - Pinned events: Never deleted
    """
    service = RetentionService(pool)
    result = await service.run_cleanup()

    return {
        "status": "ok",
        **result,
    }
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/admin/test_retention_endpoints.py -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add app/admin/router.py tests/unit/admin/test_retention_endpoints.py
git commit -m "feat(admin): add retention job endpoints"
```

---

## Feature 3: Supabase Auth Wiring

### Task 3.1: Create workspace_members table migration

**Files:**
- Create: `migrations/042_workspace_members.sql`

**Step 1: Write the migration**

```sql
-- migrations/042_workspace_members.sql
-- Workspace membership for user authorization

CREATE TABLE IF NOT EXISTS workspace_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT workspace_members_role_check
        CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    CONSTRAINT uq_workspace_member
        UNIQUE (workspace_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_members_user
    ON workspace_members(user_id);

CREATE INDEX IF NOT EXISTS idx_workspace_members_workspace
    ON workspace_members(workspace_id);

-- Seed: Make existing workspace owners members
INSERT INTO workspace_members (workspace_id, user_id, role)
SELECT id, owner_id, 'owner'
FROM workspaces
WHERE owner_id IS NOT NULL
ON CONFLICT DO NOTHING;
```

**Step 2: Apply migration**

Run: `mcp__supabase__apply_migration` with name `workspace_members`

**Step 3: Verify table exists**

Run: `mcp__supabase__list_tables`

Expected: `workspace_members` in list

**Step 4: Commit**

```bash
git add migrations/042_workspace_members.sql
git commit -m "feat(db): add workspace_members table"
```

---

### Task 3.2: Add RequestContext and auth dependencies

**Files:**
- Modify: `app/deps/security.py`
- Test: `tests/unit/test_security_auth.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_security_auth.py
"""Tests for auth dependencies."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import HTTPException


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_default_values(self):
        from app.deps.security import RequestContext

        ctx = RequestContext()
        assert ctx.user_id is None
        assert ctx.workspace_id is None
        assert ctx.role is None
        assert ctx.is_admin is False

    def test_admin_context(self):
        from app.deps.security import RequestContext

        ctx = RequestContext(is_admin=True, workspace_id=uuid4())
        assert ctx.is_admin is True
        assert ctx.user_id is None


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_admin_token_bypass(self):
        """Admin token returns admin context."""
        from app.deps.security import get_current_user

        with patch("app.deps.security.verify_admin_token", return_value=True):
            ctx = await get_current_user(
                authorization=None,
                x_admin_token="valid-token",
            )

        assert ctx.is_admin is True
        assert ctx.user_id is None

    @pytest.mark.asyncio
    async def test_missing_auth_raises_401(self):
        """Missing authorization raises 401."""
        from app.deps.security import get_current_user

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                authorization=None,
                x_admin_token=None,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_bearer_format_raises_401(self):
        """Non-Bearer auth raises 401."""
        from app.deps.security import get_current_user

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                authorization="Basic abc123",
                x_admin_token=None,
            )

        assert exc_info.value.status_code == 401


class TestRequireWorkspaceAccess:
    """Tests for require_workspace_access dependency."""

    @pytest.mark.asyncio
    async def test_admin_bypass_with_workspace(self):
        """Admin can access any workspace."""
        from app.deps.security import RequestContext, require_workspace_access

        ctx = RequestContext(is_admin=True)
        workspace_id = uuid4()

        mock_pool = MagicMock()

        result = await require_workspace_access(
            ctx=ctx,
            workspace_id=workspace_id,
            min_role="viewer",
            pool=mock_pool,
        )

        assert result.is_admin is True
        assert result.workspace_id == workspace_id

    @pytest.mark.asyncio
    async def test_non_member_raises_403(self):
        """Non-member raises 403."""
        from app.deps.security import RequestContext, require_workspace_access

        ctx = RequestContext(user_id=uuid4())
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None  # No membership
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await require_workspace_access(
                ctx=ctx,
                workspace_id=workspace_id,
                min_role="viewer",
                pool=mock_pool,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_insufficient_role_raises_403(self):
        """Insufficient role raises 403."""
        from app.deps.security import RequestContext, require_workspace_access

        ctx = RequestContext(user_id=uuid4())
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"role": "viewer"}  # Has viewer
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await require_workspace_access(
                ctx=ctx,
                workspace_id=workspace_id,
                min_role="admin",  # Requires admin
                pool=mock_pool,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_sufficient_role_returns_context(self):
        """Member with sufficient role returns context."""
        from app.deps.security import RequestContext, require_workspace_access

        user_id = uuid4()
        ctx = RequestContext(user_id=user_id)
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"role": "admin"}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await require_workspace_access(
            ctx=ctx,
            workspace_id=workspace_id,
            min_role="member",
            pool=mock_pool,
        )

        assert result.user_id == user_id
        assert result.workspace_id == workspace_id
        assert result.role == "admin"
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_security_auth.py -v`

Expected: FAIL with `ImportError` or `AttributeError`

**Step 3: Update security module**

Add to `app/deps/security.py`:

```python
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Query

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RequestContext:
    """Auth context resolved from request."""

    user_id: Optional[UUID] = None
    workspace_id: Optional[UUID] = None
    role: Optional[str] = None
    is_admin: bool = False


ROLE_RANK = {"viewer": 1, "member": 2, "admin": 3, "owner": 4}


async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> RequestContext:
    """
    Resolve user identity from JWT or admin token.

    Does NOT resolve workspace - that's separate.
    """
    # (1) Admin token bypass
    if x_admin_token:
        if verify_admin_token(x_admin_token):
            return RequestContext(is_admin=True)
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # (2) Require Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = authorization.split(" ", 1)[1]

    # (3) Validate via Supabase Auth API
    # TODO: Initialize supabase client from settings
    try:
        from app.deps.supabase import get_supabase_client

        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return RequestContext(user_id=UUID(user_response.user.id))
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Auth validation failed", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


async def require_workspace_access(
    ctx: RequestContext,
    workspace_id: UUID,
    min_role: str = "viewer",
    pool=None,
) -> RequestContext:
    """
    Verify user has access to workspace with minimum role.

    Admin bypass still requires explicit workspace_id.
    """
    # Admin bypass - still needs workspace_id for scoping
    if ctx.is_admin:
        return RequestContext(is_admin=True, workspace_id=workspace_id)

    if not ctx.user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    # Look up membership
    query = """
        SELECT role FROM workspace_members
        WHERE workspace_id = $1 AND user_id = $2
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, workspace_id, ctx.user_id)

    if not row:
        raise HTTPException(status_code=403, detail="Not a member of this workspace")

    role = row["role"]
    if ROLE_RANK.get(role, 0) < ROLE_RANK.get(min_role, 0):
        raise HTTPException(
            status_code=403, detail=f"Requires {min_role} role, you have {role}"
        )

    return RequestContext(
        user_id=ctx.user_id,
        workspace_id=workspace_id,
        role=role,
    )
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_security_auth.py -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add app/deps/security.py tests/unit/test_security_auth.py
git commit -m "feat(auth): add RequestContext and auth dependencies"
```

---

### Task 3.3: Add Supabase client dependency

**Files:**
- Create: `app/deps/supabase.py`
- Test: `tests/unit/test_supabase_client.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_supabase_client.py
"""Tests for Supabase client dependency."""

import pytest
from unittest.mock import patch


class TestSupabaseClient:
    """Tests for Supabase client initialization."""

    def test_client_uses_settings(self):
        """Client initializes with settings."""
        with patch("app.deps.supabase.get_settings") as mock_settings:
            mock_settings.return_value.supabase_url = "https://test.supabase.co"
            mock_settings.return_value.supabase_service_role_key = "test-key"

            from app.deps.supabase import get_supabase_client

            # Should not raise
            client = get_supabase_client()
            assert client is not None
```

**Step 2: Run test to verify it fails**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_supabase_client.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# app/deps/supabase.py
"""Supabase client dependency."""

from functools import lru_cache

from supabase import create_client, Client

from app.config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get cached Supabase client.

    Uses service role key for server-side operations.
    """
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )
```

**Step 4: Run test to verify it passes**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/test_supabase_client.py -v`

Expected: Pass

**Step 5: Commit**

```bash
git add app/deps/supabase.py tests/unit/test_supabase_client.py
git commit -m "feat(deps): add Supabase client dependency"
```

---

### Task 3.4: Run full test suite

**Step 1: Run all unit tests**

Run: `SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key pytest tests/unit/ -v --tb=short`

Expected: All tests pass

**Step 2: Run linting**

Run: `black --check app/ tests/ && flake8 app/ tests/ --max-line-length=100`

Expected: No errors

**Step 3: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix: address linting issues"
```

---

### Task 3.5: Final commit and summary

**Step 1: Create summary commit**

```bash
git log --oneline feature/platform-hardening ^master | head -20
```

**Step 2: Report completion**

Report:
- Total commits in branch
- Features implemented
- Tests added
- Ready for PR

---

## Execution Checklist

| Task | Description | Est. Steps |
|------|-------------|------------|
| 1.1 | Idempotency migration | 4 |
| 1.2 | Request hash utility | 5 |
| 1.3 | Repository idempotency | 5 |
| 1.4 | Router idempotency | 5 |
| 2.1 | Severity/pinned migration | 4 |
| 2.2 | Rollup table migration | 4 |
| 2.3 | Rollup repository | 5 |
| 2.4 | Retention service | 5 |
| 2.5 | Admin endpoints | 5 |
| 3.1 | Members table migration | 4 |
| 3.2 | Auth dependencies | 5 |
| 3.3 | Supabase client | 5 |
| 3.4 | Full test suite | 3 |
| 3.5 | Summary | 2 |

**Total: ~61 steps across 14 tasks**
