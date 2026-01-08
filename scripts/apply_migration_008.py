#!/usr/bin/env python3
"""Apply migration 008: backtest_runs table."""
import asyncio
import asyncpg
import os

MIGRATION = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,
    strategy_spec_id UUID REFERENCES kb_strategy_specs(id) ON DELETE SET NULL,
    spec_version INTEGER,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    params JSONB NOT NULL DEFAULT '{}',
    engine TEXT NOT NULL DEFAULT 'backtesting.py',
    dataset_meta JSONB NOT NULL DEFAULT '{}',
    summary JSONB,
    equity_curve JSONB,
    trades JSONB,
    warnings JSONB DEFAULT '[]',
    error TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_workspace ON backtest_runs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_entity_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created ON backtest_runs(created_at DESC);
"""

async def main():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"], statement_cache_size=0)
    try:
        await conn.execute(MIGRATION)
        print("Migration 008 applied: backtest_runs table created")

        # Verify
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'backtest_runs'"
        )
        print(f"Table has {count} columns")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
