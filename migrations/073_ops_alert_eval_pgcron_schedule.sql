-- Migration: 073_ops_alert_eval_pgcron_schedule
-- Purpose: Schedule automatic ops alert evaluation jobs via pg_cron (if available)
-- Design: Idempotent - unschedules existing jobs before rescheduling
-- Note: pg_cron may not be available on all Supabase plans

-- ===========================================
-- Function to enqueue an ops alert eval job
-- ===========================================

CREATE OR REPLACE FUNCTION enqueue_alert_evaluator_job(
    p_workspace_id UUID DEFAULT NULL,
    p_triggered_by TEXT DEFAULT 'schedule',
    p_now TIMESTAMPTZ DEFAULT NULL  -- For testing; NULL = use current time
)
RETURNS UUID AS $$
DECLARE
    v_job_id UUID;
    v_dedupe_key TEXT;
    v_now_utc TIMESTAMPTZ;
    v_bucket_ts TIMESTAMP;  -- No tz needed, just for formatting
    v_bucket TEXT;
BEGIN
    -- Single timestamp source (testable via p_now param)
    v_now_utc := COALESCE(p_now, now());

    -- 15-minute bucket: truncate to hour in UTC, add floored 15m interval
    -- Results in :00, :15, :30, :45 boundaries
    v_bucket_ts := date_trunc('hour', timezone('UTC', v_now_utc))
                   + make_interval(mins => (extract(minute FROM timezone('UTC', v_now_utc))::int / 15) * 15);
    v_bucket := to_char(v_bucket_ts, 'YYYY-MM-DD"T"HH24:MI');

    -- Dedupe key format: ops_alerts.evaluate:<bucket>[:workspace_id]
    IF p_workspace_id IS NULL THEN
        v_dedupe_key := 'ops_alerts.evaluate:' || v_bucket;
    ELSE
        v_dedupe_key := 'ops_alerts.evaluate:' || v_bucket || ':' || p_workspace_id::text;
    END IF;

    INSERT INTO jobs (type, payload, priority, dedupe_key)
    VALUES (
        'ops_alert_eval',
        jsonb_build_object(
            'workspace_id', p_workspace_id,
            'triggered_by', p_triggered_by
        ),
        50  -- Higher priority (lower number = higher priority)
    )
    ON CONFLICT (dedupe_key) WHERE dedupe_key IS NOT NULL
    DO UPDATE SET dedupe_key = jobs.dedupe_key  -- No-op, return existing
    RETURNING id INTO v_job_id;

    RETURN v_job_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION enqueue_alert_evaluator_job IS 'Enqueue an ops_alert_eval job for health/coverage/drift rule evaluation. Called by pg_cron schedules. Uses 15-minute bucket dedupe to prevent double-runs.';

-- ===========================================
-- pg_cron job scheduling (conditional)
-- ===========================================

DO $$
DECLARE
    v_jobid INT;
BEGIN
    -- Check if pg_cron extension is available
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        RAISE NOTICE 'pg_cron extension not available. Ops alert evaluation must be triggered manually via API or external scheduler.';
        RETURN;
    END IF;

    -- ===========================================
    -- Unschedule existing jobs (idempotent)
    -- pg_cron's unschedule() takes jobid (int), not jobname
    -- Must look up jobid first for safe re-scheduling
    -- ===========================================

    SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'ops_alert_eval_scheduled' LIMIT 1;
    IF v_jobid IS NOT NULL THEN
        PERFORM cron.unschedule(v_jobid);
        RAISE NOTICE 'Unscheduled existing job: ops_alert_eval_scheduled (jobid=%)', v_jobid;
    END IF;

    -- ===========================================
    -- Schedule new job
    -- Timing: every 15 minutes (matches dedupe bucket for clarity)
    -- Each cron fire creates one job; dedupe prevents double-runs
    -- ===========================================

    PERFORM cron.schedule(
        'ops_alert_eval_scheduled',
        '*/15 * * * *',  -- Every 15 minutes at :00, :15, :30, :45
        $$SELECT enqueue_alert_evaluator_job(NULL, 'schedule')$$
    );
    RAISE NOTICE 'Scheduled job: ops_alert_eval_scheduled every 15 minutes';

    RAISE NOTICE 'Ops alert eval scheduling configured successfully.';
END $$;

-- ===========================================
-- Verification query (run manually to check schedule)
-- ===========================================
-- SELECT jobid, jobname, schedule, command, active
-- FROM cron.job
-- WHERE jobname = 'ops_alert_eval_scheduled';

-- ===========================================
-- Manual execution examples
-- ===========================================
-- Evaluate all workspaces (default):
-- SELECT enqueue_alert_evaluator_job();
-- SELECT enqueue_alert_evaluator_job(NULL, 'manual');
--
-- Evaluate specific workspace:
-- SELECT enqueue_alert_evaluator_job('a1b2c3d4-...'::uuid, 'admin');
--
-- Test with explicit timestamp (for deterministic tests):
-- SELECT enqueue_alert_evaluator_job(NULL, 'test', '2026-01-19 14:07:00+00'::timestamptz);
--
-- Check queued jobs:
-- SELECT id, type, status, priority, dedupe_key, payload, created_at
-- FROM jobs
-- WHERE type = 'ops_alert_eval'
-- ORDER BY created_at DESC
-- LIMIT 10;
