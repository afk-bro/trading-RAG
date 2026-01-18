-- Migration: 058_pine_archiver_pgcron
-- Purpose: SQL function and pg_cron job for archiving stale Pine scripts
-- Design: Idempotent - archives scripts not seen in N days
-- Note: pg_cron may not be available on all Supabase plans

-- ===========================================
-- Archive function for stale Pine scripts
-- ===========================================

CREATE OR REPLACE FUNCTION pine_archive_stale_scripts(
    p_older_than_days INT DEFAULT 7,
    p_batch_limit INT DEFAULT 1000,
    p_dry_run BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    archived_count INT,
    workspace_ids UUID[],
    dry_run BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_archived_count INT := 0;
    v_workspace_ids UUID[];
    v_cutoff TIMESTAMPTZ;
BEGIN
    v_cutoff := NOW() - (p_older_than_days || ' days')::INTERVAL;

    IF p_dry_run THEN
        -- Dry run: count without archiving
        SELECT COUNT(*), ARRAY_AGG(DISTINCT workspace_id)
        INTO v_archived_count, v_workspace_ids
        FROM strategy_scripts
        WHERE status != 'archived'
          AND last_seen_at < v_cutoff;

        RETURN QUERY SELECT v_archived_count, v_workspace_ids, TRUE;
        RETURN;
    END IF;

    -- Get workspace IDs before update
    SELECT ARRAY_AGG(DISTINCT workspace_id)
    INTO v_workspace_ids
    FROM strategy_scripts
    WHERE status != 'archived'
      AND last_seen_at < v_cutoff;

    -- Archive stale scripts
    WITH archived AS (
        UPDATE strategy_scripts
        SET status = 'archived'
        WHERE status != 'archived'
          AND last_seen_at < v_cutoff
        RETURNING id
    )
    SELECT COUNT(*)::INT INTO v_archived_count FROM archived;

    -- Log to retention_job_log if table exists
    IF v_archived_count > 0 THEN
        BEGIN
            INSERT INTO retention_job_log (
                job_name,
                started_at,
                finished_at,
                rows_deleted,
                status
            ) VALUES (
                'pine_archive_stale_scripts',
                NOW(),
                NOW(),
                v_archived_count,
                'success'
            );
        EXCEPTION WHEN undefined_table THEN
            -- retention_job_log may not exist, skip logging
            NULL;
        END;
    END IF;

    RETURN QUERY SELECT v_archived_count, v_workspace_ids, FALSE;
END;
$$;

COMMENT ON FUNCTION pine_archive_stale_scripts IS
'Archive Pine scripts not seen in N days. Idempotent - only archives non-archived scripts.
Usage: SELECT * FROM pine_archive_stale_scripts(7, 1000, FALSE);
Dry run: SELECT * FROM pine_archive_stale_scripts(7, 1000, TRUE);';

-- ===========================================
-- pg_cron job scheduling (conditional)
-- ===========================================

DO $$
DECLARE
    v_jobid INT;
BEGIN
    -- Check if pg_cron extension is available
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        RAISE NOTICE 'pg_cron extension not available. Pine archiver must be triggered manually via admin endpoint or external scheduler.';
        RETURN;
    END IF;

    -- Unschedule existing job (idempotent)
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'pine_archive_stale_scripts';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: pine_archive_stale_scripts (jobid=%)', v_jobid;
    END IF;

    -- Schedule new job at 3:35 AM UTC daily (staggered from retention jobs)
    -- Archives scripts not seen in 7 days
    PERFORM cron.schedule(
        'pine_archive_stale_scripts',
        '35 3 * * *',  -- 3:35 AM UTC daily
        $$SELECT * FROM pine_archive_stale_scripts(7, 1000, FALSE)$$
    );
    RAISE NOTICE 'Scheduled job: pine_archive_stale_scripts at 3:35 AM UTC daily';

END $$;

-- ===========================================
-- Verification queries
-- ===========================================
-- Check scheduled job:
-- SELECT jobid, jobname, schedule, command, active
-- FROM cron.job
-- WHERE jobname = 'pine_archive_stale_scripts';

-- Dry run to preview:
-- SELECT * FROM pine_archive_stale_scripts(7, 1000, TRUE);

-- Manual archive:
-- SELECT * FROM pine_archive_stale_scripts(7, 1000, FALSE);
