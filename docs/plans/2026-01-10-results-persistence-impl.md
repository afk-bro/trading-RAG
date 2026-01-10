# Results Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist run plans and variant results to database tables instead of in-memory only.

**Architecture:** Add `run_plans` table as grouping container, extend `backtest_runs` with variant columns. Write path persists on completion, dual-writes lifecycle events as breadcrumbs.

**Tech Stack:** PostgreSQL (via asyncpg), Pydantic models, FastAPI endpoints

---

## Task 1: Create run_plans migration

**Files:**
- Create: `migrations/022_run_plans.sql`

**Step 1: Write the migration SQL**

```sql
-- Migration: 022_run_plans
-- Create run_plans table for orchestration grouping

CREATE TABLE IF NOT EXISTS run_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    objective_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    n_variants INT NOT NULL DEFAULT 0,
    n_completed INT NOT NULL DEFAULT 0,
    n_failed INT NOT NULL DEFAULT 0,
    n_skipped INT NOT NULL DEFAULT 0,
    best_backtest_run_id UUID NULL,
    best_objective_score DOUBLE PRECISION NULL,
    error_summary TEXT NULL,
    plan JSONB NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_run_plans_workspace
    ON run_plans(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_plans_status
    ON run_plans(status, created_at DESC);

COMMENT ON TABLE run_plans IS 'Grouping container for test/tune orchestration runs';
COMMENT ON COLUMN run_plans.plan IS 'Immutable JSON: inputs, resolved variants, provenance';
```

**Step 2: Apply migration via Supabase MCP**

Run: `mcp__supabase__apply_migration` with name `run_plans` and the SQL above

**Step 3: Commit**

```bash
git add migrations/022_run_plans.sql
git commit -m "feat(db): add run_plans table for orchestration grouping"
```

---

## Task 2: Extend backtest_runs with variant columns

**Files:**
- Create: `migrations/023_backtest_runs_variants.sql`

**Step 1: Write the migration SQL**

```sql
-- Migration: 023_backtest_runs_variants
-- Add variant tracking columns to backtest_runs

-- New columns for run plan linkage
ALTER TABLE backtest_runs
ADD COLUMN IF NOT EXISTS run_plan_id UUID NULL REFERENCES run_plans(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS variant_index INT NULL,
ADD COLUMN IF NOT EXISTS variant_fingerprint TEXT NULL,
ADD COLUMN IF NOT EXISTS run_kind TEXT NOT NULL DEFAULT 'backtest'
    CHECK (run_kind IN ('backtest', 'tune_variant', 'test_variant')),
ADD COLUMN IF NOT EXISTS objective_score DOUBLE PRECISION NULL,
ADD COLUMN IF NOT EXISTS skip_reason TEXT NULL;

-- Artifact metadata escape hatches
ALTER TABLE backtest_runs
ADD COLUMN IF NOT EXISTS has_equity_curve BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS has_trades BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS equity_points INT NULL,
ADD COLUMN IF NOT EXISTS trade_count INT NULL,
ADD COLUMN IF NOT EXISTS artifacts_ref JSONB NULL;

-- Update status check to include 'skipped'
-- First drop existing constraint, then add new one
ALTER TABLE backtest_runs DROP CONSTRAINT IF EXISTS backtest_runs_status_check;
ALTER TABLE backtest_runs ADD CONSTRAINT backtest_runs_status_check
    CHECK (status IN ('running', 'completed', 'failed', 'skipped'));

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_backtest_runs_plan_variant
    ON backtest_runs(run_plan_id, variant_index);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_plan_score
    ON backtest_runs(run_plan_id, objective_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_kind_created
    ON backtest_runs(run_kind, created_at DESC);

COMMENT ON COLUMN backtest_runs.run_plan_id IS 'Links to parent run_plan (NULL for standalone backtests)';
COMMENT ON COLUMN backtest_runs.variant_index IS '0..N-1 ordering within a run plan';
COMMENT ON COLUMN backtest_runs.variant_fingerprint IS 'hash(canonical(params)) for verification';
COMMENT ON COLUMN backtest_runs.run_kind IS 'Distinguishes standalone vs plan variants';
COMMENT ON COLUMN backtest_runs.objective_score IS 'Extracted for fast ORDER BY queries';
COMMENT ON COLUMN backtest_runs.artifacts_ref IS 'Future S3 refs: {"equity_curve": "s3://..."}';
```

**Step 2: Apply migration via Supabase MCP**

Run: `mcp__supabase__apply_migration` with name `backtest_runs_variants` and the SQL above

**Step 3: Commit**

```bash
git add migrations/023_backtest_runs_variants.sql
git commit -m "feat(db): extend backtest_runs with variant tracking columns"
```

---

## Task 3: Add run_failed and run_cancelled event types

**Files:**
- Modify: `app/schemas.py` (TradeEventType enum)
- Create: `migrations/024_run_event_types.sql`

**Step 1: Find and read the TradeEventType enum**

Run: `grep -n "class TradeEventType" app/schemas.py`

**Step 2: Add new event types to enum**

Add to `TradeEventType` enum:
```python
RUN_FAILED = "run_failed"
RUN_CANCELLED = "run_cancelled"
```

**Step 3: Write migration to update CHECK constraint**

```sql
-- Migration: 024_run_event_types
-- Add run_failed and run_cancelled to trade_events event_type check

ALTER TABLE trade_events DROP CONSTRAINT IF EXISTS trade_events_event_type_check;
ALTER TABLE trade_events ADD CONSTRAINT trade_events_event_type_check
    CHECK (event_type IN (
        'intent_emitted', 'policy_evaluated', 'intent_approved', 'intent_rejected',
        'order_filled', 'position_opened', 'position_scaled', 'position_closed',
        'run_started', 'run_completed', 'run_failed', 'run_cancelled'
    ));
```

**Step 4: Apply migration**

**Step 5: Commit**

```bash
git add app/schemas.py migrations/024_run_event_types.sql
git commit -m "feat: add run_failed and run_cancelled event types"
```

---

## Task 4: Create RunPlansRepository

**Files:**
- Create: `app/repositories/run_plans.py`
- Test: `tests/unit/repositories/test_run_plans.py`

**Step 1: Write the failing test**

```python
"""Unit tests for RunPlansRepository."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.run_plans import RunPlansRepository


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def repo(mock_pool):
    """Create repository with mock pool."""
    return RunPlansRepository(mock_pool)


class TestCreateRunPlan:
    """Tests for create_run_plan method."""

    @pytest.mark.asyncio
    async def test_create_run_plan_returns_id(self, repo, mock_pool):
        """create_run_plan returns the new plan ID."""
        plan_id = uuid4()
        workspace_id = uuid4()

        # Setup mock
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=plan_id)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await repo.create_run_plan(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe_dd_penalty",
            n_variants=10,
            plan={"inputs": {}, "resolved": {}, "provenance": {}},
        )

        assert result == plan_id
        mock_conn.fetchval.assert_called_once()


class TestUpdateRunPlanStatus:
    """Tests for update_run_plan_status method."""

    @pytest.mark.asyncio
    async def test_update_to_running(self, repo, mock_pool):
        """update_run_plan_status updates status to running."""
        plan_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_run_plan_status(plan_id, "running")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "running" in str(call_args)


class TestCompleteRunPlan:
    """Tests for complete_run_plan method."""

    @pytest.mark.asyncio
    async def test_complete_run_plan_sets_aggregates(self, repo, mock_pool):
        """complete_run_plan updates all aggregate fields."""
        plan_id = uuid4()
        best_run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.complete_run_plan(
            plan_id=plan_id,
            status="completed",
            n_completed=8,
            n_failed=1,
            n_skipped=1,
            best_backtest_run_id=best_run_id,
            best_objective_score=1.42,
        )

        mock_conn.execute.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/repositories/test_run_plans.py -v`
Expected: FAIL with "No module named 'app.repositories.run_plans'"

**Step 3: Write the repository implementation**

```python
"""Repository for run_plans table operations."""

import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class RunPlansRepository:
    """Repository for run_plans table operations."""

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self.pool = pool

    async def create_run_plan(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        objective_name: str,
        n_variants: int,
        plan: dict[str, Any],
        status: str = "pending",
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

        Returns:
            The new run plan ID
        """
        query = """
            INSERT INTO run_plans (
                workspace_id, strategy_entity_id, objective_name,
                n_variants, plan, status
            )
            VALUES ($1, $2, $3, $4, $5, $6)
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
            )

        logger.info(
            "Created run plan",
            plan_id=str(plan_id),
            workspace_id=str(workspace_id),
            n_variants=n_variants,
        )

        return plan_id

    async def update_run_plan_status(
        self,
        plan_id: UUID,
        status: str,
    ) -> None:
        """Update run plan status."""
        query = "UPDATE run_plans SET status = $2 WHERE id = $1"

        async with self.pool.acquire() as conn:
            await conn.execute(query, plan_id, status)

    async def complete_run_plan(
        self,
        plan_id: UUID,
        status: str,
        n_completed: int,
        n_failed: int,
        n_skipped: int,
        best_backtest_run_id: Optional[UUID] = None,
        best_objective_score: Optional[float] = None,
        error_summary: Optional[str] = None,
    ) -> None:
        """
        Complete a run plan with final aggregates.

        Args:
            plan_id: Run plan ID
            status: Final status (completed, failed, cancelled)
            n_completed: Count of successful variants
            n_failed: Count of failed variants
            n_skipped: Count of skipped variants
            best_backtest_run_id: ID of best variant's backtest_runs row
            best_objective_score: Best variant's objective score
            error_summary: Error message if failed
        """
        query = """
            UPDATE run_plans
            SET status = $2,
                completed_at = NOW(),
                n_completed = $3,
                n_failed = $4,
                n_skipped = $5,
                best_backtest_run_id = $6,
                best_objective_score = $7,
                error_summary = $8
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                plan_id,
                status,
                n_completed,
                n_failed,
                n_skipped,
                best_backtest_run_id,
                best_objective_score,
                error_summary,
            )

        logger.info(
            "Completed run plan",
            plan_id=str(plan_id),
            status=status,
            n_completed=n_completed,
            n_failed=n_failed,
            n_skipped=n_skipped,
        )

    async def get_run_plan(self, plan_id: UUID) -> Optional[dict[str, Any]]:
        """Get a run plan by ID."""
        query = """
            SELECT rp.*,
                   e.name as strategy_name
            FROM run_plans rp
            LEFT JOIN kb_entities e ON rp.strategy_entity_id = e.id
            WHERE rp.id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, plan_id)
            if not row:
                return None

            result = dict(row)

            # Parse JSONB
            if result.get("plan") and isinstance(result["plan"], str):
                result["plan"] = json.loads(result["plan"])

            return result

    async def list_run_plans(
        self,
        workspace_id: UUID,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List run plans with optional filtering.

        Returns:
            Tuple of (plans list, total count)
        """
        conditions = ["rp.workspace_id = $1"]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if status:
            conditions.append(f"rp.status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM run_plans rp WHERE {where_clause}"

        # Don't include full plan JSONB in list query
        list_query = f"""
            SELECT rp.id, rp.workspace_id, rp.strategy_entity_id,
                   rp.status, rp.objective_name, rp.created_at, rp.completed_at,
                   rp.n_variants, rp.n_completed, rp.n_failed, rp.n_skipped,
                   rp.best_backtest_run_id, rp.best_objective_score, rp.error_summary,
                   e.name as strategy_name
            FROM run_plans rp
            LEFT JOIN kb_entities e ON rp.strategy_entity_id = e.id
            WHERE {where_clause}
            ORDER BY rp.created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        plans = [dict(row) for row in rows]
        return plans, total

    async def list_runs_for_plan(
        self,
        plan_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List backtest_runs for a run plan (without large blobs).

        Returns:
            Tuple of (runs list, total count)
        """
        count_query = """
            SELECT COUNT(*) FROM backtest_runs WHERE run_plan_id = $1
        """

        list_query = """
            SELECT id, workspace_id, strategy_entity_id, run_plan_id,
                   variant_index, variant_fingerprint, run_kind,
                   status, objective_score, params, summary,
                   has_equity_curve, has_trades, equity_points, trade_count,
                   skip_reason, error, started_at, completed_at, created_at
            FROM backtest_runs
            WHERE run_plan_id = $1
            ORDER BY variant_index ASC
            LIMIT $2 OFFSET $3
        """

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, plan_id)
            rows = await conn.fetch(list_query, plan_id, limit, offset)

        runs = []
        for row in rows:
            run = dict(row)
            # Parse JSONB fields
            for field in ["params", "summary"]:
                if run.get(field) and isinstance(run[field], str):
                    run[field] = json.loads(run[field])
            runs.append(run)

        return runs, total
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/repositories/test_run_plans.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/repositories/run_plans.py tests/unit/repositories/test_run_plans.py
git commit -m "feat: add RunPlansRepository for run plan persistence"
```

---

## Task 5: Extend BacktestRepository with variant methods

**Files:**
- Modify: `app/repositories/backtests.py`
- Test: `tests/unit/repositories/test_backtests_variants.py`

**Step 1: Write the failing test**

```python
"""Unit tests for BacktestRepository variant methods."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.backtests import BacktestRepository


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def repo(mock_pool):
    """Create repository with mock pool."""
    return BacktestRepository(mock_pool)


class TestCreateVariantRun:
    """Tests for create_variant_run method."""

    @pytest.mark.asyncio
    async def test_create_variant_run_returns_id(self, repo, mock_pool):
        """create_variant_run returns the new run ID."""
        run_id = uuid4()
        run_plan_id = uuid4()
        workspace_id = uuid4()
        strategy_entity_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=run_id)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await repo.create_variant_run(
            run_plan_id=run_plan_id,
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            variant_index=0,
            variant_fingerprint="abc123def456",
            params={"lookback_days": 200},
            dataset_meta={"filename": "BTC_1h.csv"},
        )

        assert result == run_id
        mock_conn.fetchval.assert_called_once()


class TestUpdateVariantCompleted:
    """Tests for update_variant_completed method."""

    @pytest.mark.asyncio
    async def test_update_variant_completed_sets_fields(self, repo, mock_pool):
        """update_variant_completed updates all result fields."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_variant_completed(
            run_id=run_id,
            summary={"sharpe": 1.42, "return_pct": 12.5},
            equity_curve=[{"t": "2024-01-01", "equity": 10000}],
            trades=[],
            objective_score=1.42,
            has_equity_curve=True,
            has_trades=False,
            equity_points=100,
            trade_count=0,
        )

        mock_conn.execute.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/repositories/test_backtests_variants.py -v`
Expected: FAIL with "BacktestRepository has no attribute 'create_variant_run'"

**Step 3: Add methods to BacktestRepository**

Add to `app/repositories/backtests.py`:

```python
    async def create_variant_run(
        self,
        run_plan_id: UUID,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        variant_index: int,
        variant_fingerprint: str,
        params: dict[str, Any],
        dataset_meta: dict[str, Any],
        run_kind: str = "test_variant",
    ) -> UUID:
        """
        Create a new backtest run for a plan variant.

        Returns:
            The new run ID
        """
        query = """
            INSERT INTO backtest_runs (
                run_plan_id, workspace_id, strategy_entity_id,
                variant_index, variant_fingerprint, run_kind,
                params, dataset_meta, status, started_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'running', NOW())
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            run_id = await conn.fetchval(
                query,
                run_plan_id,
                workspace_id,
                strategy_entity_id,
                variant_index,
                variant_fingerprint,
                run_kind,
                json.dumps(params),
                json.dumps(dataset_meta),
            )

        logger.info(
            "Created variant run",
            run_id=str(run_id),
            run_plan_id=str(run_plan_id),
            variant_index=variant_index,
        )

        return run_id

    async def update_variant_completed(
        self,
        run_id: UUID,
        summary: dict[str, Any],
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        objective_score: Optional[float],
        has_equity_curve: bool,
        has_trades: bool,
        equity_points: Optional[int],
        trade_count: Optional[int],
        warnings: Optional[list[str]] = None,
    ) -> None:
        """Update variant run with completed results."""
        query = """
            UPDATE backtest_runs
            SET status = 'completed',
                summary = $2,
                equity_curve = $3,
                trades = $4,
                objective_score = $5,
                has_equity_curve = $6,
                has_trades = $7,
                equity_points = $8,
                trade_count = $9,
                warnings = $10,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                json.dumps(summary),
                json.dumps(equity_curve),
                json.dumps(trades),
                objective_score,
                has_equity_curve,
                has_trades,
                equity_points,
                trade_count,
                json.dumps(warnings or []),
            )

        logger.info("Updated variant run as completed", run_id=str(run_id))

    async def update_variant_skipped(
        self,
        run_id: UUID,
        skip_reason: str,
    ) -> None:
        """Update variant run as skipped."""
        query = """
            UPDATE backtest_runs
            SET status = 'skipped',
                skip_reason = $2,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, run_id, skip_reason)

        logger.info(
            "Updated variant run as skipped",
            run_id=str(run_id),
            skip_reason=skip_reason,
        )

    async def update_variant_failed(
        self,
        run_id: UUID,
        error: str,
    ) -> None:
        """Update variant run as failed."""
        query = """
            UPDATE backtest_runs
            SET status = 'failed',
                error = $2,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, run_id, error)

        logger.info(
            "Updated variant run as failed",
            run_id=str(run_id),
            error=error,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/repositories/test_backtests_variants.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/repositories/backtests.py tests/unit/repositories/test_backtests_variants.py
git commit -m "feat: add variant run methods to BacktestRepository"
```

---

## Task 6: Create PlanBuilder helper for plan JSONB

**Files:**
- Create: `app/services/testing/plan_builder.py`
- Test: `tests/unit/services/testing/test_plan_builder.py`

**Step 1: Write the failing test**

```python
"""Unit tests for PlanBuilder."""

import pytest

from app.services.testing.plan_builder import PlanBuilder


class TestPlanBuilder:
    """Tests for PlanBuilder."""

    def test_build_plan_has_three_sections(self):
        """build returns dict with inputs, resolved, provenance."""
        builder = PlanBuilder(
            base_spec={"strategy_name": "breakout"},
            objective="sharpe_dd_penalty",
            constraints={"max_variants": 25},
            dataset_ref="BTC_1h.csv",
            generator_name="grid_search_v1",
            generator_version="1.0.0",
        )

        builder.add_variant(0, {"lookback_days": 200}, "baseline")
        builder.add_variant(1, {"lookback_days": 252}, "grid")

        plan = builder.build()

        assert "inputs" in plan
        assert "resolved" in plan
        assert "provenance" in plan
        assert plan["resolved"]["n_variants"] == 2
        assert len(plan["resolved"]["variants"]) == 2

    def test_build_includes_fingerprints(self):
        """build includes fingerprints in provenance."""
        builder = PlanBuilder(
            base_spec={"strategy_name": "breakout"},
            objective="sharpe_dd_penalty",
            constraints={},
            dataset_ref="BTC_1h.csv",
            generator_name="grid_search_v1",
            generator_version="1.0.0",
        )

        plan = builder.build()

        assert "fingerprints" in plan["provenance"]
        assert "plan" in plan["provenance"]["fingerprints"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/services/testing/test_plan_builder.py -v`
Expected: FAIL with "No module named 'app.services.testing.plan_builder'"

**Step 3: Write the implementation**

```python
"""PlanBuilder: Constructs immutable plan JSONB for run_plans table."""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class PlanBuilder:
    """
    Builder for run_plans.plan JSONB structure.

    Constructs the three-layer plan format:
    - inputs: what was requested
    - resolved: what variants were generated
    - provenance: how to interpret it later
    """

    def __init__(
        self,
        base_spec: dict[str, Any],
        objective: str,
        constraints: dict[str, Any],
        dataset_ref: str,
        generator_name: str,
        generator_version: str,
        generator_config: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize plan builder.

        Args:
            base_spec: Base strategy specification
            objective: Objective function name
            constraints: Generator constraints
            dataset_ref: Dataset reference string
            generator_name: Name of generator (e.g., "grid_search_v1")
            generator_version: Version of generator
            generator_config: Optional generator configuration
            seed: Optional random seed
        """
        self._base_spec = base_spec
        self._objective = objective
        self._constraints = constraints
        self._dataset_ref = dataset_ref
        self._generator_name = generator_name
        self._generator_version = generator_version
        self._generator_config = generator_config or {}
        self._seed = seed
        self._variants: list[dict[str, Any]] = []

    def add_variant(
        self,
        variant_index: int,
        params: dict[str, Any],
        param_source: str,
    ) -> "PlanBuilder":
        """
        Add a resolved variant to the plan.

        Args:
            variant_index: 0-based index
            params: Fully materialized params for this variant
            param_source: Source type ("baseline", "grid", "ablation", etc.)

        Returns:
            self for chaining
        """
        self._variants.append({
            "variant_index": variant_index,
            "params": params,
            "param_source": param_source,
        })
        return self

    def build(self) -> dict[str, Any]:
        """
        Build the final plan JSONB structure.

        Returns:
            Complete plan dict with inputs, resolved, provenance
        """
        created_at = datetime.now(timezone.utc).isoformat()

        inputs = {
            "base_spec": self._base_spec,
            "objective": {
                "name": self._objective,
                "direction": "maximize",
            },
            "constraints": self._constraints,
            "dataset_ref": self._dataset_ref,
            "generator_config": self._generator_config,
        }

        resolved = {
            "n_variants": len(self._variants),
            "variants": self._variants,
        }

        # Compute fingerprints
        plan_content = canonical_json({
            "inputs": inputs,
            "resolved": resolved,
        })
        plan_fingerprint = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

        provenance = {
            "generator": {
                "name": self._generator_name,
                "version": self._generator_version,
            },
            "created_at": created_at,
            "seed": self._seed,
            "fingerprints": {
                "plan": f"sha256:{plan_fingerprint}",
            },
        }

        return {
            "inputs": inputs,
            "resolved": resolved,
            "provenance": provenance,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/services/testing/test_plan_builder.py -v`
Expected: PASS

**Step 5: Update `__init__.py` exports**

Add to `app/services/testing/__init__.py`:
```python
from app.services.testing.plan_builder import PlanBuilder
```

**Step 6: Commit**

```bash
git add app/services/testing/plan_builder.py tests/unit/services/testing/test_plan_builder.py app/services/testing/__init__.py
git commit -m "feat: add PlanBuilder for constructing plan JSONB"
```

---

## Task 7: Update RunOrchestrator to persist results

**Files:**
- Modify: `app/services/testing/run_orchestrator.py`
- Test: `tests/unit/services/testing/test_run_orchestrator_persist.py`

**Step 1: Write the failing test**

```python
"""Unit tests for RunOrchestrator persistence."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.testing.models import (
    RunPlan,
    RunVariant,
    RunResultStatus,
)
from app.services.testing.run_orchestrator import RunOrchestrator


@pytest.fixture
def mock_events_repo():
    repo = MagicMock()
    repo.insert = AsyncMock()
    return repo


@pytest.fixture
def mock_run_plans_repo():
    repo = MagicMock()
    repo.create_run_plan = AsyncMock(return_value=uuid4())
    repo.update_run_plan_status = AsyncMock()
    repo.complete_run_plan = AsyncMock()
    return repo


@pytest.fixture
def mock_backtest_repo():
    repo = MagicMock()
    repo.create_variant_run = AsyncMock(return_value=uuid4())
    repo.update_variant_completed = AsyncMock()
    repo.update_variant_skipped = AsyncMock()
    repo.update_variant_failed = AsyncMock()
    return repo


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def orchestrator(mock_events_repo, mock_run_plans_repo, mock_backtest_repo, mock_runner):
    return RunOrchestrator(
        events_repo=mock_events_repo,
        run_plans_repo=mock_run_plans_repo,
        backtest_repo=mock_backtest_repo,
        runner=mock_runner,
    )


class TestExecutePersistence:
    """Tests for execute method persistence."""

    @pytest.mark.asyncio
    async def test_execute_creates_run_plan(
        self, orchestrator, mock_run_plans_repo
    ):
        """execute creates run_plan in DB at start."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={"strategy_name": "test"},
            variants=[],
            dataset_ref="test.csv",
        )

        csv_content = b"ts,open,high,low,close,volume\n2024-01-01T00:00:00Z,100,101,99,100.5,1000\n2024-01-02T00:00:00Z,100.5,102,100,101,1100"

        await orchestrator.execute(run_plan, csv_content)

        mock_run_plans_repo.create_run_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_completes_run_plan(
        self, orchestrator, mock_run_plans_repo
    ):
        """execute calls complete_run_plan at end."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={"strategy_name": "test", "risk": {"dollars_per_trade": 100, "max_positions": 1}},
            variants=[
                RunVariant(
                    variant_id="abc123",
                    label="baseline",
                    spec_overrides={},
                )
            ],
            dataset_ref="test.csv",
        )

        csv_content = b"ts,open,high,low,close,volume\n2024-01-01T00:00:00Z,100,101,99,100.5,1000\n2024-01-02T00:00:00Z,100.5,102,100,101,1100"

        await orchestrator.execute(run_plan, csv_content)

        mock_run_plans_repo.complete_run_plan.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/services/testing/test_run_orchestrator_persist.py -v`
Expected: FAIL (missing repo parameters or method not implemented)

**Step 3: Update RunOrchestrator constructor and execute method**

Update `app/services/testing/run_orchestrator.py`:

```python
class RunOrchestrator:
    """Orchestrates execution of RunPlan variants with persistence."""

    def __init__(
        self,
        events_repo,
        runner,
        run_plans_repo=None,
        backtest_repo=None,
    ):
        """Initialize orchestrator.

        Args:
            events_repo: TradeEventsRepository for journaling events
            runner: StrategyRunner for evaluating strategies
            run_plans_repo: RunPlansRepository for plan persistence (optional)
            backtest_repo: BacktestRepository for variant persistence (optional)
        """
        self._events_repo = events_repo
        self._runner = runner
        self._run_plans_repo = run_plans_repo
        self._backtest_repo = backtest_repo
```

Then update `execute` method to:
1. Create run_plan in DB at start
2. Create backtest_run for each variant
3. Update variant result on completion
4. Call complete_run_plan at end

(Full implementation in step 3 of actual execution)

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/services/testing/test_run_orchestrator_persist.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/testing/run_orchestrator.py tests/unit/services/testing/test_run_orchestrator_persist.py
git commit -m "feat: add persistence to RunOrchestrator"
```

---

## Task 8: Update testing router to use repositories

**Files:**
- Modify: `app/routers/testing.py`

**Step 1: Add repository initialization**

Update router to initialize repositories and pass to orchestrator:

```python
from app.repositories.run_plans import RunPlansRepository
from app.repositories.backtests import BacktestRepository

def _get_run_plans_repo() -> RunPlansRepository:
    """Get run plans repository."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return RunPlansRepository(_db_pool)


def _get_backtest_repo() -> BacktestRepository:
    """Get backtest repository."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return BacktestRepository(_db_pool)
```

**Step 2: Update generate_and_execute endpoint**

```python
@router.post("/run-plans/generate-and-execute", ...)
async def generate_and_execute_run_plan(...):
    # ... existing code ...

    # Execute run plan with all repositories
    events_repo = _get_events_repo()
    run_plans_repo = _get_run_plans_repo()
    backtest_repo = _get_backtest_repo()
    runner = StrategyRunner()

    orchestrator = RunOrchestrator(
        events_repo=events_repo,
        run_plans_repo=run_plans_repo,
        backtest_repo=backtest_repo,
        runner=runner,
    )

    # ... rest unchanged ...
```

**Step 3: Run existing tests**

Run: `pytest tests/unit/routers/test_testing.py -v`
Expected: PASS (may need mock updates)

**Step 4: Commit**

```bash
git add app/routers/testing.py
git commit -m "feat: wire repositories into testing router"
```

---

## Task 9: Add admin verification endpoints

**Files:**
- Modify: `app/admin/router.py`
- Test: `tests/unit/admin/test_run_plans_endpoints.py`

**Step 1: Write the failing test**

```python
"""Unit tests for admin run plans endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestGetRunPlan:
    """Tests for GET /admin/run-plans/{id}."""

    def test_get_run_plan_returns_plan(self, client):
        """GET /admin/run-plans/{id} returns plan data."""
        plan_id = uuid4()

        with patch("app.admin.router._get_run_plans_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_run_plan = AsyncMock(return_value={
                "id": plan_id,
                "status": "completed",
                "n_variants": 10,
                "plan": {"inputs": {}, "resolved": {}, "provenance": {}},
            })
            mock_get_repo.return_value = mock_repo

            response = client.get(
                f"/admin/run-plans/{plan_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(plan_id)
        assert data["status"] == "completed"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/admin/test_run_plans_endpoints.py -v`
Expected: FAIL with 404 (endpoint doesn't exist)

**Step 3: Add endpoints to admin router**

Add to `app/admin/router.py`:

```python
from app.repositories.run_plans import RunPlansRepository

def _get_run_plans_repo() -> RunPlansRepository:
    """Get run plans repository."""
    if _db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return RunPlansRepository(_db_pool)


@router.get("/run-plans/{plan_id}")
async def get_run_plan(
    plan_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Get a run plan by ID (verification endpoint)."""
    repo = _get_run_plans_repo()
    plan = await repo.get_run_plan(plan_id)

    if not plan:
        raise HTTPException(status_code=404, detail="Run plan not found")

    return plan


@router.get("/run-plans/{plan_id}/runs")
async def get_run_plan_runs(
    plan_id: UUID,
    limit: int = 100,
    offset: int = 0,
    _: str = Depends(require_admin_token),
):
    """List backtest_runs for a run plan (no large blobs)."""
    repo = _get_run_plans_repo()

    # Verify plan exists
    plan = await repo.get_run_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Run plan not found")

    runs, total = await repo.list_runs_for_plan(plan_id, limit=limit, offset=offset)

    return {
        "runs": runs,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/admin/test_run_plans_endpoints.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/admin/router.py tests/unit/admin/test_run_plans_endpoints.py
git commit -m "feat: add admin verification endpoints for run plans"
```

---

## Task 10: Update event journaling to use correct types

**Files:**
- Modify: `app/services/testing/run_orchestrator.py`

**Step 1: Update _journal_run_event to use proper event types**

Replace placeholder INTENT_EMITTED with actual run event types:

```python
from app.schemas import TradeEvent, TradeEventType

async def _journal_run_event(
    self, run_plan: RunPlan, event_type: str, payload: dict
) -> None:
    """Journal a run-level event."""
    # Map event type string to enum
    event_type_map = {
        "RUN_STARTED": TradeEventType.RUN_STARTED,
        "RUN_COMPLETED": TradeEventType.RUN_COMPLETED,
        "RUN_FAILED": TradeEventType.RUN_FAILED,
        "RUN_CANCELLED": TradeEventType.RUN_CANCELLED,
    }

    trade_event_type = event_type_map.get(event_type)
    if not trade_event_type:
        logger.warning("Unknown run event type", event_type=event_type)
        return

    event = TradeEvent(
        correlation_id=str(run_plan.run_plan_id),
        workspace_id=run_plan.workspace_id,
        event_type=trade_event_type,
        payload=payload,
    )

    try:
        await self._events_repo.insert(event)
    except Exception as e:
        logger.warning(
            "failed_to_journal_run_event",
            event_type=event_type,
            error=str(e),
        )
```

**Step 2: Run tests**

Run: `pytest tests/unit/services/testing/ -v`
Expected: PASS

**Step 3: Commit**

```bash
git add app/services/testing/run_orchestrator.py
git commit -m "fix: use proper TradeEventType enums for run events"
```

---

## Task 11: Integration test for full persistence flow

**Files:**
- Create: `tests/integration/test_run_persistence.py`

**Step 1: Write integration test**

```python
"""Integration tests for run plan persistence."""

import pytest
from uuid import uuid4

from app.services.testing import (
    TestGenerator,
    RunOrchestrator,
    GeneratorConstraints,
)
from app.services.strategy.models import ExecutionSpec


@pytest.mark.integration
@pytest.mark.requires_db
class TestRunPersistenceFlow:
    """Integration tests for complete persistence flow."""

    @pytest.mark.asyncio
    async def test_full_persistence_flow(
        self,
        db_pool,
        workspace_id,
    ):
        """Test complete flow: generate -> execute -> persist."""
        from app.repositories.run_plans import RunPlansRepository
        from app.repositories.backtests import BacktestRepository
        from app.repositories.trade_events import TradeEventsRepository
        from app.services.strategy.runner import StrategyRunner

        # Setup repos
        run_plans_repo = RunPlansRepository(db_pool)
        backtest_repo = BacktestRepository(db_pool)
        events_repo = TradeEventsRepository(db_pool)
        runner = StrategyRunner()

        # Generate plan
        base_spec = ExecutionSpec(
            strategy_id=uuid4(),
            name="test_strategy",
            symbol="BTC",
            timeframe="1h",
            params={"entry": {"lookback_days": 200}},
            risk={"dollars_per_trade": 1000, "max_positions": 1},
        )

        constraints = GeneratorConstraints(
            lookback_days_values=[200, 252],
            max_variants=5,
        )

        generator = TestGenerator()
        run_plan = generator.generate(
            base_spec=base_spec,
            dataset_ref="test.csv",
            constraints=constraints,
        )

        # CSV data
        csv_content = b"ts,open,high,low,close,volume\n2024-01-01T00:00:00Z,100,101,99,100.5,1000\n2024-01-02T00:00:00Z,100.5,102,100,101,1100"

        # Execute with persistence
        orchestrator = RunOrchestrator(
            events_repo=events_repo,
            run_plans_repo=run_plans_repo,
            backtest_repo=backtest_repo,
            runner=runner,
        )

        results = await orchestrator.execute(run_plan, csv_content)

        # Verify plan was persisted
        persisted_plan = await run_plans_repo.get_run_plan(run_plan.run_plan_id)
        assert persisted_plan is not None
        assert persisted_plan["status"] == "completed"
        assert persisted_plan["n_variants"] == len(run_plan.variants)

        # Verify variants were persisted
        runs, total = await run_plans_repo.list_runs_for_plan(run_plan.run_plan_id)
        assert total == len(run_plan.variants)
```

**Step 2: Run integration test (requires DB)**

Run: `pytest tests/integration/test_run_persistence.py -v -m "requires_db"`
Expected: PASS (with real DB) or SKIP (without DB)

**Step 3: Commit**

```bash
git add tests/integration/test_run_persistence.py
git commit -m "test: add integration test for run persistence flow"
```

---

## Task 12: Final PR preparation

**Files:**
- Update: `CLAUDE.md` (document new tables)

**Step 1: Update CLAUDE.md with new schema info**

Add to database schema section:

```markdown
**run_plans** - Orchestration grouping for test/tune runs
- `id`, `workspace_id`, `strategy_entity_id`
- `status` (pending|running|completed|failed|cancelled)
- `objective_name`, `n_variants`, `n_completed`, `n_failed`, `n_skipped`
- `best_backtest_run_id`, `best_objective_score`
- `plan` (jsonb) - immutable: inputs, resolved variants, provenance
- `created_at`, `completed_at`

**backtest_runs** - Extended with variant tracking
- New: `run_plan_id`, `variant_index`, `variant_fingerprint`
- New: `run_kind` (backtest|tune_variant|test_variant)
- New: `objective_score`, `skip_reason`
- New: `has_equity_curve`, `has_trades`, `equity_points`, `trade_count`
- New: `artifacts_ref` (jsonb) - future S3 refs
```

**Step 2: Run all tests**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

**Step 3: Run linting**

Run: `black --check app/ tests/ && flake8 app/ tests/ --max-line-length=100 && mypy app/`
Expected: PASS

**Step 4: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with results persistence schema"
```

**Step 5: Create PR**

```bash
git push -u origin feat/results-persistence
gh pr create --title "feat: Results Persistence (#6)" --body "$(cat <<'EOF'
## Summary
- Add `run_plans` table as orchestration grouping container
- Extend `backtest_runs` with variant tracking columns
- Persist run plans and variant results on completion
- Dual-write lifecycle events as breadcrumbs
- Add admin verification endpoints

## Test plan
- [ ] Unit tests pass
- [ ] Integration tests pass (with DB)
- [ ] Manual test: run generate-and-execute, verify DB rows
- [ ] Manual test: check admin endpoints return data

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Summary

| Task | Description | Estimated Complexity |
|------|-------------|---------------------|
| 1 | Create run_plans migration | Low |
| 2 | Extend backtest_runs migration | Low |
| 3 | Add event types | Low |
| 4 | Create RunPlansRepository | Medium |
| 5 | Extend BacktestRepository | Medium |
| 6 | Create PlanBuilder | Low |
| 7 | Update RunOrchestrator | High |
| 8 | Update testing router | Low |
| 9 | Add admin endpoints | Medium |
| 10 | Fix event journaling | Low |
| 11 | Integration test | Medium |
| 12 | PR preparation | Low |

**Total: 12 tasks, ~2-3 hours of implementation**
