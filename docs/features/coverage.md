# Coverage Triage Workflow

Admin endpoints for managing coverage gaps in the cockpit UI.

## Architecture (`app/admin/coverage.py`, `app/services/coverage_gap/repository.py`)

```
Match Run (weak_coverage=true)
       │
       ▼
  Coverage Status ──► open → acknowledged → resolved
       │
       ▼
  Priority Score (deterministic ranking)
```

## Status Lifecycle

- `open` - New coverage gap, needs attention (default)
- `acknowledged` - Someone is investigating
- `resolved` - Gap addressed (strategy added, false positive, etc.)

## Priority Score Formula

Higher = more urgent:

| Component | Value | Condition |
|-----------|-------|-----------|
| Base | `0.5 - best_score` | Clamped to [0, 0.5] |
| No results | +0.2 | `num_above_threshold == 0` |
| NO_MATCHES | +0.15 | Reason code present |
| NO_STRONG_MATCHES | +0.1 | Reason code present |
| Recency | +0.05 | Created in last 24h |

## Endpoints

**List**: `GET /admin/coverage/weak?workspace_id=...&status=open`
- `status`: `open` (default), `acknowledged`, `resolved`, `all`
- `include_candidate_cards=true` - Hydrate strategy cards
- Results sorted by `priority_score` descending

**Update**: `PATCH /admin/coverage/weak/{run_id}`
- Body: `{"status": "acknowledged|resolved", "note": "optional"}`
- Tracks `acknowledged_at/by`, `resolved_at/by`, `resolution_note`

**Response Fields**:
- `coverage_status`, `priority_score`
- `strategy_cards_by_id`, `missing_strategy_ids`

## Resolution Guard

Cannot mark `resolved` without at least one of:
- `candidate_strategy_ids` present
- `resolution_note` provided

Returns 400 if guard fails.

## Auto-Resolve on Success

When `/youtube/match-pine` produces `weak_coverage=false`:
1. Find all `open`/`acknowledged` runs with same `intent_signature`
2. Auto-resolve with `resolved_by='system'`
3. Set `resolution_note='Auto-resolved by successful match'`

## LLM-Powered Explanation

`POST /admin/coverage/explain` - Generate explanation of strategy match

- Request: `{run_id, strategy_id}` + `workspace_id` query param
- Response: `{explanation, model, provider, latency_ms}`
- Requires LLM configuration
- Returns 503 if unconfigured, 404 if not found

## Cockpit UI (`/admin/coverage/cockpit`)

- Two-panel layout: queue (left) + detail (right)
- Status tabs: Open, Acknowledged, Resolved, All
- Priority badges: P1 (>=0.75), P2 (>=0.40), P3 (<0.40)
- Strategy cards with tags, backtest status, OOS score
- "Explain Match" button generates LLM explanation
- Deep link support: `/admin/coverage/cockpit/{run_id}`
- Triage controls: Acknowledge, Resolve, Reopen
