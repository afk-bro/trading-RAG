-- Migration: 025_run_plans_started_at
-- Add started_at column to run_plans for duration tracking

ALTER TABLE run_plans
ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ NULL;

COMMENT ON COLUMN run_plans.started_at IS 'When execution began (first variant started)';
