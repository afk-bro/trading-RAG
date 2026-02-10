# Database Migrations

PostgreSQL schema migrations for the RAG system, applied against a Supabase-hosted database.

## Overview

These migrations define the full schema: core RAG tables (documents, chunks, vectors),
backtest/tuning infrastructure, strategy lifecycle, knowledge-base entities, execution
tracking, operational alerts, and supporting views/indexes/triggers.

## Naming Convention

Files follow the pattern `NNN_descriptive_name.sql` where `NNN` is a three-digit
zero-padded sequence number (e.g., `001_initial_schema.sql`, `076_strategy_versions.sql`).

## How to Apply

Migrations are applied **manually** via the Supabase SQL Editor
(Dashboard > SQL Editor). Copy-paste each file in order, starting from the first
unapplied migration.

The convenience file `apply_012_to_019.sql` wraps migrations 012 through 019 in a
single transaction. It is safe to run multiple times.

There is no automated migration runner. Order matters -- always apply in ascending
numeric order.

## Numbering Gaps

Some sequence numbers are intentionally skipped:

| Gap       | Reason                                      |
|-----------|---------------------------------------------|
| 010 - 011 | Drafted but never shipped                   |
| 034 - 038 | Drafted but never shipped                   |

This is by design. **Never reuse a skipped number.** Always use the next number after
the highest existing migration.

## Creating New Migrations

1. Use the next available number after the current maximum (currently `085`).
2. Keep migrations idempotent where possible:
   - `CREATE TABLE IF NOT EXISTS`
   - `ADD COLUMN IF NOT EXISTS`
   - `CREATE INDEX IF NOT EXISTS`
   - `DROP CONSTRAINT IF EXISTS` before re-adding constraints
3. Each file should be self-contained -- do not rely on transaction state from
   other files.
4. Include a comment header describing the migration purpose.

## Key Tables

| Table                    | Purpose                                       |
|--------------------------|-----------------------------------------------|
| `workspaces`             | Multi-tenant control plane, per-workspace config |
| `documents`              | Ingested source metadata and content hashes   |
| `chunks`                 | Token-aware text segments with metadata arrays |
| `chunk_vectors`          | Embedding records per model/collection        |
| `backtest_runs`          | Backtest execution records                    |
| `backtest_tunes`         | Parameter tuning sessions                     |
| `backtest_tune_runs`     | Individual runs within a tuning session       |
| `wfo_runs`               | Walk-forward optimization runs                |
| `strategies`             | Strategy registry                             |
| `strategy_versions`      | Immutable config snapshots with state machine |
| `strategy_intel_snapshots` | Regime + confidence time series             |
| `run_plans`              | Execution plan tracking                       |
| `trade_events`           | Trade event log with retention policies       |
| `jobs` / `job_events`    | Async job orchestration                       |
| `ops_alerts`             | Operational alert management                  |
| `kb_entities`            | Knowledge-base entity registry                |
