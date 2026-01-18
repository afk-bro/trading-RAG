-- Migration: 068_data_sync_pgcron_schedule
-- Purpose: Schedule automatic data sync jobs via pg_cron (if available)
-- Design: Idempotent - unschedules existing jobs before rescheduling
-- Note: pg_cron may not be available on all Supabase plans

-- ===========================================
-- Function to enqueue a data sync job
-- ===========================================

CREATE OR REPLACE FUNCTION enqueue_data_sync_job(
    p_exchange_id TEXT DEFAULT NULL,
    p_mode TEXT DEFAULT 'incremental'
)
RETURNS UUID AS $$
DECLARE
    v_job_id UUID;
BEGIN
    INSERT INTO jobs (type, payload, priority)
    VALUES (
        'data_sync',
        jsonb_build_object(
            'exchange_id', p_exchange_id,
            'mode', p_mode,
            'triggered_by', 'pg_cron'
        ),
        50  -- Higher priority than ad-hoc jobs (lower number = higher priority)
    )
    RETURNING id INTO v_job_id;

    RETURN v_job_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION enqueue_data_sync_job IS 'Enqueue a data_sync job for OHLCV data refresh. Called by pg_cron schedules.';

-- ===========================================
-- pg_cron job scheduling (conditional)
-- ===========================================

DO $$
DECLARE
    v_jobid INT;
BEGIN
    -- Check if pg_cron extension is available
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        RAISE NOTICE 'pg_cron extension not available. Data sync jobs must be triggered manually via API or external scheduler.';
        RETURN;
    END IF;

    -- ===========================================
    -- Unschedule existing jobs (idempotent)
    -- pg_cron's unschedule() takes jobid (int), not jobname
    -- Must look up jobid first for safe re-scheduling
    -- ===========================================

    -- Unschedule hourly incremental sync
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'data_sync_hourly_incremental';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: data_sync_hourly_incremental (jobid=%)', v_jobid;
    END IF;

    -- Unschedule daily full sync
    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'data_sync_daily_full';
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: data_sync_daily_full (jobid=%)', v_jobid;
    END IF;

    -- ===========================================
    -- Schedule new jobs
    -- Timing chosen to:
    -- - Hourly at :05 - frequent enough for active trading, offset from hour boundary
    -- - Daily at 3:45 AM UTC - catches gaps, staggered from retention jobs (3:15-3:30)
    -- ===========================================

    -- Hourly incremental sync at minute 5 of each hour
    -- Keeps data fresh for active trading strategies
    PERFORM cron.schedule(
        'data_sync_hourly_incremental',
        '5 * * * *',  -- Every hour at :05
        $$SELECT enqueue_data_sync_job(NULL, 'incremental')$$
    );
    RAISE NOTICE 'Scheduled job: data_sync_hourly_incremental at :05 every hour';

    -- Daily full sync at 3:45 AM UTC
    -- Catches any gaps from missed incremental syncs
    -- Rebuilds full dataset for data integrity
    PERFORM cron.schedule(
        'data_sync_daily_full',
        '45 3 * * *',  -- Daily at 3:45 AM UTC
        $$SELECT enqueue_data_sync_job(NULL, 'full')$$
    );
    RAISE NOTICE 'Scheduled job: data_sync_daily_full at 3:45 AM UTC daily';

    RAISE NOTICE 'All data sync jobs scheduled successfully.';
END $$;

-- ===========================================
-- Verification query (run manually to check schedule)
-- ===========================================
-- SELECT jobid, jobname, schedule, command, active
-- FROM cron.job
-- WHERE jobname LIKE 'data_sync_%'
-- ORDER BY jobname;

-- ===========================================
-- Manual execution examples
-- ===========================================
-- Incremental sync (default):
-- SELECT enqueue_data_sync_job();
-- SELECT enqueue_data_sync_job(NULL, 'incremental');
--
-- Full sync:
-- SELECT enqueue_data_sync_job(NULL, 'full');
--
-- Exchange-specific sync:
-- SELECT enqueue_data_sync_job('binance', 'incremental');
-- SELECT enqueue_data_sync_job('coinbase', 'full');
--
-- Check queued jobs:
-- SELECT id, type, status, priority, payload, created_at
-- FROM jobs
-- WHERE type = 'data_sync'
-- ORDER BY created_at DESC
-- LIMIT 10;
