-- Migration: 054_retention_functions
-- Purpose: Batch delete functions for data retention with audit logging
-- Design: Uses LIMIT loops for safe batch deletes, avoids long locks

-- ===========================================
-- Retention job log table for auditing
-- ===========================================

CREATE TABLE IF NOT EXISTS retention_job_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    rows_deleted INTEGER NOT NULL DEFAULT 0,
    cutoff_ts TIMESTAMPTZ NOT NULL,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    ok BOOLEAN NOT NULL DEFAULT FALSE,
    error TEXT
);

-- Index for querying recent runs
CREATE INDEX IF NOT EXISTS idx_retention_job_log_job_started
    ON retention_job_log(job_name, started_at DESC);

COMMENT ON TABLE retention_job_log IS 'Audit log for retention job executions';

-- ===========================================
-- Additional indexes for retention queries
-- ===========================================

-- trade_events: created_at index for batch deletes (may already exist via 040)
CREATE INDEX IF NOT EXISTS idx_trade_events_created_at
    ON trade_events(created_at);

-- job_runs: started_at index for batch deletes
CREATE INDEX IF NOT EXISTS idx_job_runs_started_at
    ON job_runs(started_at);

-- match_runs: resolved status + created_at for retention queries
-- Only prune resolved items older than cutoff
CREATE INDEX IF NOT EXISTS idx_match_runs_resolved_created
    ON match_runs(created_at)
    WHERE coverage_status = 'resolved';

-- ===========================================
-- Batch delete function: trade_events
-- Prunes RUN_* events only (not ORDER_* which are source of truth)
-- ===========================================

CREATE OR REPLACE FUNCTION retention_prune_trade_events(
    p_cutoff INTERVAL DEFAULT '90 days',
    p_batch_size INT DEFAULT 10000,
    p_dry_run BOOLEAN DEFAULT FALSE
)
RETURNS TABLE(deleted_count BIGINT, job_log_id UUID) AS $$
DECLARE
    v_log_id UUID;
    v_cutoff TIMESTAMPTZ := NOW() - p_cutoff;
    v_total_deleted BIGINT := 0;
    v_batch_deleted INT;
    -- Positive allowlist of event types safe to prune (RUN_* breadcrumbs)
    -- ORDER_* events are source of truth and must be retained forever
    v_prune_types TEXT[] := ARRAY[
        'run_started',
        'run_completed',
        'run_failed',
        'run_cancelled'
    ];
BEGIN
    -- Create log entry
    INSERT INTO retention_job_log (job_name, cutoff_ts, dry_run)
    VALUES ('trade_events', v_cutoff, p_dry_run)
    RETURNING id INTO v_log_id;

    IF p_dry_run THEN
        -- Count only, no delete
        SELECT COUNT(*) INTO v_total_deleted
        FROM trade_events
        WHERE created_at < v_cutoff
          AND event_type = ANY(v_prune_types)
          AND (pinned IS NULL OR pinned = FALSE);
    ELSE
        -- Batch delete loop to avoid long locks
        -- Uses ORDER BY for deterministic behavior across runs
        LOOP
            WITH candidates AS (
                SELECT id FROM trade_events
                WHERE created_at < v_cutoff
                  AND event_type = ANY(v_prune_types)
                  AND (pinned IS NULL OR pinned = FALSE)
                ORDER BY created_at ASC
                LIMIT p_batch_size
            ),
            deleted AS (
                DELETE FROM trade_events
                WHERE id IN (SELECT id FROM candidates)
                RETURNING 1
            )
            SELECT COUNT(*) INTO v_batch_deleted FROM deleted;

            v_total_deleted := v_total_deleted + v_batch_deleted;

            EXIT WHEN v_batch_deleted < p_batch_size;

            -- Small sleep between batches to reduce lock contention
            PERFORM pg_sleep(0.1);
        END LOOP;
    END IF;

    -- Update log entry
    UPDATE retention_job_log
    SET finished_at = NOW(), rows_deleted = v_total_deleted, ok = TRUE
    WHERE id = v_log_id;

    RETURN QUERY SELECT v_total_deleted, v_log_id;

EXCEPTION WHEN OTHERS THEN
    UPDATE retention_job_log
    SET finished_at = NOW(), ok = FALSE, error = SQLERRM
    WHERE id = v_log_id;
    RAISE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION retention_prune_trade_events IS 'Prune old RUN_* trade events. ORDER_* events retained forever as source of truth.';

-- ===========================================
-- Batch delete function: job_runs
-- Prunes all job_runs older than cutoff
-- ===========================================

CREATE OR REPLACE FUNCTION retention_prune_job_runs(
    p_cutoff INTERVAL DEFAULT '30 days',
    p_batch_size INT DEFAULT 10000,
    p_dry_run BOOLEAN DEFAULT FALSE
)
RETURNS TABLE(deleted_count BIGINT, job_log_id UUID) AS $$
DECLARE
    v_log_id UUID;
    v_cutoff TIMESTAMPTZ := NOW() - p_cutoff;
    v_total_deleted BIGINT := 0;
    v_batch_deleted INT;
BEGIN
    -- Create log entry
    INSERT INTO retention_job_log (job_name, cutoff_ts, dry_run)
    VALUES ('job_runs', v_cutoff, p_dry_run)
    RETURNING id INTO v_log_id;

    IF p_dry_run THEN
        -- Count only
        SELECT COUNT(*) INTO v_total_deleted
        FROM job_runs
        WHERE started_at < v_cutoff;
    ELSE
        -- Batch delete loop
        LOOP
            WITH candidates AS (
                SELECT id FROM job_runs
                WHERE started_at < v_cutoff
                ORDER BY started_at ASC
                LIMIT p_batch_size
            ),
            deleted AS (
                DELETE FROM job_runs
                WHERE id IN (SELECT id FROM candidates)
                RETURNING 1
            )
            SELECT COUNT(*) INTO v_batch_deleted FROM deleted;

            v_total_deleted := v_total_deleted + v_batch_deleted;

            EXIT WHEN v_batch_deleted < p_batch_size;

            PERFORM pg_sleep(0.1);
        END LOOP;
    END IF;

    -- Update log entry
    UPDATE retention_job_log
    SET finished_at = NOW(), rows_deleted = v_total_deleted, ok = TRUE
    WHERE id = v_log_id;

    RETURN QUERY SELECT v_total_deleted, v_log_id;

EXCEPTION WHEN OTHERS THEN
    UPDATE retention_job_log
    SET finished_at = NOW(), ok = FALSE, error = SQLERRM
    WHERE id = v_log_id;
    RAISE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION retention_prune_job_runs IS 'Prune old job_runs records (default 30 days).';

-- ===========================================
-- Batch delete function: match_runs
-- Prunes only RESOLVED match_runs older than cutoff
-- Open/acknowledged items retained for triage
-- ===========================================

CREATE OR REPLACE FUNCTION retention_prune_match_runs(
    p_cutoff INTERVAL DEFAULT '180 days',
    p_batch_size INT DEFAULT 10000,
    p_dry_run BOOLEAN DEFAULT FALSE
)
RETURNS TABLE(deleted_count BIGINT, job_log_id UUID) AS $$
DECLARE
    v_log_id UUID;
    v_cutoff TIMESTAMPTZ := NOW() - p_cutoff;
    v_total_deleted BIGINT := 0;
    v_batch_deleted INT;
BEGIN
    -- Create log entry
    INSERT INTO retention_job_log (job_name, cutoff_ts, dry_run)
    VALUES ('match_runs', v_cutoff, p_dry_run)
    RETURNING id INTO v_log_id;

    IF p_dry_run THEN
        -- Count only
        SELECT COUNT(*) INTO v_total_deleted
        FROM match_runs
        WHERE created_at < v_cutoff
          AND coverage_status = 'resolved';
    ELSE
        -- Batch delete loop
        LOOP
            WITH candidates AS (
                SELECT id FROM match_runs
                WHERE created_at < v_cutoff
                  AND coverage_status = 'resolved'
                ORDER BY created_at ASC
                LIMIT p_batch_size
            ),
            deleted AS (
                DELETE FROM match_runs
                WHERE id IN (SELECT id FROM candidates)
                RETURNING 1
            )
            SELECT COUNT(*) INTO v_batch_deleted FROM deleted;

            v_total_deleted := v_total_deleted + v_batch_deleted;

            EXIT WHEN v_batch_deleted < p_batch_size;

            PERFORM pg_sleep(0.1);
        END LOOP;
    END IF;

    -- Update log entry
    UPDATE retention_job_log
    SET finished_at = NOW(), rows_deleted = v_total_deleted, ok = TRUE
    WHERE id = v_log_id;

    RETURN QUERY SELECT v_total_deleted, v_log_id;

EXCEPTION WHEN OTHERS THEN
    UPDATE retention_job_log
    SET finished_at = NOW(), ok = FALSE, error = SQLERRM
    WHERE id = v_log_id;
    RAISE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION retention_prune_match_runs IS 'Prune old resolved match_runs (default 180 days). Open/acknowledged items retained.';

-- ===========================================
-- Batch delete function: idempotency_keys
-- Prunes expired idempotency keys (already have expires_at column)
-- ===========================================

CREATE OR REPLACE FUNCTION retention_prune_idempotency_keys(
    p_batch_size INT DEFAULT 10000,
    p_dry_run BOOLEAN DEFAULT FALSE
)
RETURNS TABLE(deleted_count BIGINT, job_log_id UUID) AS $$
DECLARE
    v_log_id UUID;
    v_total_deleted BIGINT := 0;
    v_batch_deleted INT;
BEGIN
    -- Create log entry (cutoff is NOW since we use expires_at)
    INSERT INTO retention_job_log (job_name, cutoff_ts, dry_run)
    VALUES ('idempotency_keys', NOW(), p_dry_run)
    RETURNING id INTO v_log_id;

    IF p_dry_run THEN
        -- Count only
        SELECT COUNT(*) INTO v_total_deleted
        FROM idempotency_keys
        WHERE expires_at < NOW();
    ELSE
        -- Batch delete loop
        LOOP
            WITH candidates AS (
                SELECT id FROM idempotency_keys
                WHERE expires_at < NOW()
                ORDER BY expires_at ASC
                LIMIT p_batch_size
            ),
            deleted AS (
                DELETE FROM idempotency_keys
                WHERE id IN (SELECT id FROM candidates)
                RETURNING 1
            )
            SELECT COUNT(*) INTO v_batch_deleted FROM deleted;

            v_total_deleted := v_total_deleted + v_batch_deleted;

            EXIT WHEN v_batch_deleted < p_batch_size;

            PERFORM pg_sleep(0.1);
        END LOOP;
    END IF;

    -- Update log entry
    UPDATE retention_job_log
    SET finished_at = NOW(), rows_deleted = v_total_deleted, ok = TRUE
    WHERE id = v_log_id;

    RETURN QUERY SELECT v_total_deleted, v_log_id;

EXCEPTION WHEN OTHERS THEN
    UPDATE retention_job_log
    SET finished_at = NOW(), ok = FALSE, error = SQLERRM
    WHERE id = v_log_id;
    RAISE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION retention_prune_idempotency_keys IS 'Prune expired idempotency keys (7-day default expiry).';

-- ===========================================
-- Notes on VACUUM strategy
-- ===========================================
-- After large deletes, tables may have dead tuples. Options:
-- 1. Let autovacuum handle it (default, usually sufficient)
-- 2. Run VACUUM manually during maintenance windows
-- 3. Consider VACUUM (VERBOSE) to monitor dead tuple counts
--
-- For very large deletes (>1M rows), consider:
-- - Running retention during low-traffic hours (pg_cron at 3 AM)
-- - Increasing autovacuum_vacuum_scale_factor temporarily
-- - Manual VACUUM ANALYZE after completion
