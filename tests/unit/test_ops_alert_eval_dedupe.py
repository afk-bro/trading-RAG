"""Unit tests for ops alert eval job dedupe logic.

Tests verify:
1. 15-minute bucket calculation is correct
2. Dedupe key format matches expected pattern
3. Same bucket returns same job ID (dedupe works)
4. Different bucket returns new job ID

Run SQL tests with: pytest tests/unit/test_ops_alert_eval_dedupe.py -v -m requires_db
"""

import os

import pytest

# Mark all SQL-dependent tests
pytestmark = [pytest.mark.requires_db]


@pytest.fixture
async def db_conn():
    """Create database connection for testing SQL functions.

    Requires DATABASE_URL environment variable.
    Uses statement_cache_size=0 for pgbouncer compatibility.
    """
    import asyncpg

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not configured")

    conn = await asyncpg.connect(database_url, statement_cache_size=0)
    try:
        yield conn
    finally:
        # Cleanup test jobs
        await conn.execute(
            """
            DELETE FROM jobs
            WHERE type = 'ops_alert_eval'
              AND payload->>'triggered_by' = 'test'
            """
        )
        await conn.close()


class TestEnqueueAlertEvaluatorJob:
    """Test the enqueue_alert_evaluator_job SQL function."""

    @pytest.mark.asyncio
    async def test_bucket_format_on_hour(self, db_conn):
        """Bucket at :00 minutes formats correctly."""
        t = "2026-01-19 14:00:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )
        assert job_id is not None

        # Check dedupe key format
        row = await db_conn.fetchrow(
            "SELECT dedupe_key FROM jobs WHERE id = $1", job_id
        )
        assert row["dedupe_key"] == "ops_alerts.evaluate:2026-01-19T14:00"

    @pytest.mark.asyncio
    async def test_bucket_format_at_15(self, db_conn):
        """Bucket at :15 minutes formats correctly."""
        t = "2026-01-19 14:15:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )
        assert job_id is not None

        row = await db_conn.fetchrow(
            "SELECT dedupe_key FROM jobs WHERE id = $1", job_id
        )
        assert row["dedupe_key"] == "ops_alerts.evaluate:2026-01-19T14:15"

    @pytest.mark.asyncio
    async def test_bucket_format_at_30(self, db_conn):
        """Bucket at :30 minutes formats correctly."""
        t = "2026-01-19 14:30:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )
        assert job_id is not None

        row = await db_conn.fetchrow(
            "SELECT dedupe_key FROM jobs WHERE id = $1", job_id
        )
        assert row["dedupe_key"] == "ops_alerts.evaluate:2026-01-19T14:30"

    @pytest.mark.asyncio
    async def test_bucket_format_at_45(self, db_conn):
        """Bucket at :45 minutes formats correctly."""
        t = "2026-01-19 14:45:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )
        assert job_id is not None

        row = await db_conn.fetchrow(
            "SELECT dedupe_key FROM jobs WHERE id = $1", job_id
        )
        assert row["dedupe_key"] == "ops_alerts.evaluate:2026-01-19T14:45"

    @pytest.mark.asyncio
    async def test_bucket_floors_to_15m_boundary(self, db_conn):
        """Minutes 0-14 floor to :00, 15-29 floor to :15, etc."""
        # :07 should floor to :00
        t1 = "2026-01-19 14:07:00+00"
        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t1
        )
        row = await db_conn.fetchrow(
            "SELECT dedupe_key FROM jobs WHERE id = $1", job_id
        )
        assert row["dedupe_key"] == "ops_alerts.evaluate:2026-01-19T14:00"

        # :22 should floor to :15
        t2 = "2026-01-19 14:22:00+00"
        # Need different bucket for new job, use different hour
        job_id2 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t2
        )
        row2 = await db_conn.fetchrow(
            "SELECT dedupe_key FROM jobs WHERE id = $1", job_id2
        )
        assert row2["dedupe_key"] == "ops_alerts.evaluate:2026-01-19T14:15"

    @pytest.mark.asyncio
    async def test_dedupe_same_bucket_returns_same_id(self, db_conn):
        """Two enqueues in same 15m bucket return same job ID (dedupe works)."""
        t1 = "2026-01-19 15:02:00+00"  # Bucket: 15:00
        t2 = "2026-01-19 15:12:00+00"  # Bucket: 15:00 (same)

        id1 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t1
        )
        id2 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t2
        )

        assert id1 == id2, "Same bucket should return same job ID (dedupe)"

    @pytest.mark.asyncio
    async def test_dedupe_different_bucket_returns_new_id(self, db_conn):
        """Two enqueues in different 15m buckets return different job IDs."""
        t1 = "2026-01-19 16:07:00+00"  # Bucket: 16:00
        t2 = "2026-01-19 16:17:00+00"  # Bucket: 16:15 (different)

        id1 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t1
        )
        id2 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t2
        )

        assert id1 != id2, "Different buckets should return different job IDs"

    @pytest.mark.asyncio
    async def test_workspace_scoped_dedupe_key(self, db_conn):
        """Workspace-scoped dedupe key differs from all-workspace key."""
        from uuid import uuid4

        workspace_id = uuid4()
        t = "2026-01-19 17:00:00+00"

        # All-workspace job
        id1 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )

        # Workspace-scoped job (same bucket, different dedupe key)
        id2 = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job($1, 'test', $2::timestamptz)",
            workspace_id,
            t,
        )

        assert id1 != id2, "Workspace-scoped key should differ from all-workspace key"

        # Verify dedupe key format includes workspace ID
        row = await db_conn.fetchrow("SELECT dedupe_key FROM jobs WHERE id = $1", id2)
        expected_key = f"ops_alerts.evaluate:2026-01-19T17:00:{workspace_id}"
        assert row["dedupe_key"] == expected_key

    @pytest.mark.asyncio
    async def test_job_payload_contains_triggered_by(self, db_conn):
        """Job payload contains triggered_by field."""
        t = "2026-01-19 18:00:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )

        row = await db_conn.fetchrow("SELECT payload FROM jobs WHERE id = $1", job_id)
        assert row["payload"]["triggered_by"] == "test"

    @pytest.mark.asyncio
    async def test_job_priority_is_50(self, db_conn):
        """Scheduled jobs have priority 50 (higher than default 100)."""
        t = "2026-01-19 19:00:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )

        row = await db_conn.fetchrow("SELECT priority FROM jobs WHERE id = $1", job_id)
        assert row["priority"] == 50

    @pytest.mark.asyncio
    async def test_job_type_is_ops_alert_eval(self, db_conn):
        """Job type is 'ops_alert_eval'."""
        t = "2026-01-19 20:00:00+00"

        job_id = await db_conn.fetchval(
            "SELECT enqueue_alert_evaluator_job(NULL, 'test', $1::timestamptz)", t
        )

        row = await db_conn.fetchrow("SELECT type FROM jobs WHERE id = $1", job_id)
        assert row["type"] == "ops_alert_eval"
