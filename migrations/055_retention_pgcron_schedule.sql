-- Migration: 055_retention_pgcron_schedule
-- Purpose: Schedule retention jobs via pg_cron (if available)
-- Design: Idempotent - unschedules existing jobs before rescheduling
-- Note: pg_cron may not be available on all Supabase plans

-- ===========================================
-- pg_cron job scheduling (conditional)
-- ===========================================

DO $$
DECLARE
    v_jobid INT;
BEGIN
    -- Check if pg_cron extension is available
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        RAISE NOTICE 'pg_cron extension not available. Retention jobs must be triggered manually via admin endpoints or external scheduler.';
        RETURN;
    END IF;

    -- ===========================================
    -- Unschedule existing jobs (idempotent)
    -- pg_cron's unschedule() takes jobid (int), not jobname
    -- Must look up jobid first for safe re-scheduling
    -- ===========================================

    -- Unschedule trade_events retention
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'retention_prune_trade_events';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: retention_prune_trade_events (jobid=%)', v_jobid;
    END IF;

    -- Unschedule job_runs retention
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'retention_prune_job_runs';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: retention_prune_job_runs (jobid=%)', v_jobid;
    END IF;

    -- Unschedule match_runs retention
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'retention_prune_match_runs';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: retention_prune_match_runs (jobid=%)', v_jobid;
    END IF;

    -- Unschedule idempotency_keys retention
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'retention_prune_idempotency_keys';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: retention_prune_idempotency_keys (jobid=%)', v_jobid;
    END IF;

    -- ===========================================
    -- Schedule new jobs (staggered to avoid lock contention)
    -- Runs at 3:15 AM, 3:20 AM, 3:25 AM, 3:30 AM UTC
    -- ===========================================

    -- trade_events: 90-day retention for RUN_* events
    PERFORM cron.schedule(
        'retention_prune_trade_events',
        '15 3 * * *',  -- 3:15 AM UTC daily
        $$SELECT * FROM retention_prune_trade_events('90 days', 10000, FALSE)$$
    );
    RAISE NOTICE 'Scheduled job: retention_prune_trade_events at 3:15 AM UTC daily';

    -- job_runs: 30-day retention
    PERFORM cron.schedule(
        'retention_prune_job_runs',
        '20 3 * * *',  -- 3:20 AM UTC daily
        $$SELECT * FROM retention_prune_job_runs('30 days', 10000, FALSE)$$
    );
    RAISE NOTICE 'Scheduled job: retention_prune_job_runs at 3:20 AM UTC daily';

    -- match_runs: 180-day retention for resolved items
    PERFORM cron.schedule(
        'retention_prune_match_runs',
        '25 3 * * *',  -- 3:25 AM UTC daily
        $$SELECT * FROM retention_prune_match_runs('180 days', 10000, FALSE)$$
    );
    RAISE NOTICE 'Scheduled job: retention_prune_match_runs at 3:25 AM UTC daily';

    -- idempotency_keys: prune expired keys (7-day expiry built into table)
    PERFORM cron.schedule(
        'retention_prune_idempotency_keys',
        '30 3 * * *',  -- 3:30 AM UTC daily
        $$SELECT * FROM retention_prune_idempotency_keys(10000, FALSE)$$
    );
    RAISE NOTICE 'Scheduled job: retention_prune_idempotency_keys at 3:30 AM UTC daily';

    RAISE NOTICE 'All retention jobs scheduled successfully.';
END $$;

-- ===========================================
-- Verification query (run manually to check schedule)
-- ===========================================
-- SELECT jobid, jobname, schedule, command, active
-- FROM cron.job
-- WHERE jobname LIKE 'retention_prune_%'
-- ORDER BY jobname;

-- ===========================================
-- Manual execution examples
-- ===========================================
-- Dry run (count only, no delete):
-- SELECT * FROM retention_prune_trade_events('90 days', 10000, TRUE);
-- SELECT * FROM retention_prune_job_runs('30 days', 10000, TRUE);
-- SELECT * FROM retention_prune_match_runs('180 days', 10000, TRUE);
-- SELECT * FROM retention_prune_idempotency_keys(10000, TRUE);
--
-- Actual prune:
-- SELECT * FROM retention_prune_trade_events('90 days', 10000, FALSE);
-- SELECT * FROM retention_prune_job_runs('30 days', 10000, FALSE);
-- SELECT * FROM retention_prune_match_runs('180 days', 10000, FALSE);
-- SELECT * FROM retention_prune_idempotency_keys(10000, FALSE);
--
-- Check retention log:
-- SELECT * FROM retention_job_log ORDER BY started_at DESC LIMIT 20;
