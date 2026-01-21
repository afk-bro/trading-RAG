# Parallel Minimal Alerts + Telegram Design

**Date**: 2026-01-19
**Status**: Draft
**Authors**: Human + Claude

## Goal and Scope

Implement a lightweight alert system that provides:
- **Durable storage** for triggered alerts (audit trail, dedupe, debugging)
- **Telegram delivery** for immediate operator notification
- **Minimal API** for inspection (no UI yet)

This is "Option C: Parallel minimal" - immediate Telegram value with proper foundation, deferring full UI.

### In Scope (v1)
- `alert_events` table with deduplication
- AlertEvaluatorJob wired to write events
- 4 rule types: health_degraded, weak_coverage, drift_spike, confidence_drop
- Telegram notifier (new alerts + recovery)
- Read-only admin endpoints (list, detail, acknowledge, resolve)

### Deferred
- `alert_rules` table (per-workspace thresholds)
- Full admin UI (Recent Alerts panel, list page, detail page)
- Webhook sink (generic HTTP delivery)
- Alert history charts/trends

---

## Data Model

### Migration: 071_alert_events.sql

```sql
-- migrations/071_alert_events.sql
-- Alert events for ops notifications (health, coverage, drift)

CREATE TABLE IF NOT EXISTS alert_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID NOT NULL REFERENCES workspaces(id),

    -- Classification
    rule_type       TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'medium'
                    CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'resolved')),
    rule_version    TEXT NOT NULL DEFAULT 'v1',

    -- Deduplication (unique per workspace)
    dedupe_key      TEXT NOT NULL,

    -- Context
    payload         JSONB NOT NULL DEFAULT '{}',
    source          TEXT NOT NULL DEFAULT 'alert_evaluator',
    job_run_id      UUID,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,

    -- Acknowledgment (metadata, not status)
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,

    UNIQUE(workspace_id, dedupe_key)
);

-- Primary query: recent alerts for workspace
CREATE INDEX IF NOT EXISTS idx_alert_events_workspace_recent
    ON alert_events(workspace_id, created_at DESC);

-- Active alerts (ordered by last_seen for "still happening")
CREATE INDEX IF NOT EXISTS idx_alert_events_active
    ON alert_events(workspace_id, last_seen_at DESC) WHERE status = 'active';

-- Severity filtering
CREATE INDEX IF NOT EXISTS idx_alert_events_severity
    ON alert_events(workspace_id, severity, created_at DESC);

-- Job run linkage
CREATE INDEX IF NOT EXISTS idx_alert_events_job_run
    ON alert_events(job_run_id) WHERE job_run_id IS NOT NULL;

COMMENT ON TABLE alert_events IS 'Durable storage for triggered alerts with deduplication';
COMMENT ON COLUMN alert_events.dedupe_key IS 'Unique per workspace: {rule_type}:{bucket_key}';
COMMENT ON COLUMN alert_events.last_seen_at IS 'Updated each time condition is still true (heartbeat)';
COMMENT ON COLUMN alert_events.rule_version IS 'Version of rule logic that created this event';
```

### Upsert Pattern

```sql
INSERT INTO alert_events (
    workspace_id, rule_type, severity, dedupe_key,
    payload, source, job_run_id, rule_version
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (workspace_id, dedupe_key) DO UPDATE SET
    last_seen_at = NOW(),
    payload = EXCLUDED.payload,
    severity = EXCLUDED.severity,
    job_run_id = EXCLUDED.job_run_id,
    source = EXCLUDED.source
WHERE alert_events.status = 'active'
RETURNING id, (xmax = 0) AS is_new, severity AS current_severity;
```

- `is_new = True` → new alert, send Telegram
- `is_new = False` → heartbeat update only
- Track `current_severity` vs `EXCLUDED.severity` for escalation detection

### Dedupe Key Patterns

| Rule Type | Dedupe Key | Granularity |
|-----------|------------|-------------|
| health_degraded | `health_degraded:{date}` | Daily |
| weak_coverage | `weak_coverage:{P1\|P2}:{date}` | Per priority tier, daily |
| drift_spike | `drift_spike:{strategy}:{regime}:{hour}` | Hourly |
| confidence_drop | `confidence_drop:{strategy}:{hour}` | Hourly |

---

## Evaluator Changes

### Flow

```
POST /admin/jobs/evaluate-alerts?workspace_id=...
                │
                ▼
        AlertEvaluatorJob.run()
                │
    ┌───────────┼───────────┬──────────────┐
    ▼           ▼           ▼              ▼
 health     coverage     drift      confidence
 snapshot   scan         check      check
    │           │           │              │
    └───────────┴───────────┴──────────────┘
                │
                ▼
        evaluate_conditions()
                │
                ▼
        for each triggered condition:
          ├─ build dedupe_key
          ├─ upsert to alert_events
          └─ collect is_new + escalation flags
                │
                ▼
        resolution_pass()  ──► resolve cleared conditions
                │
                ▼
        notify_alerts(new + escalated + recovered)  ──► Telegram
                │
                ▼
        return job_result (metrics)
```

### Condition Definitions (v1 Hardcoded)

| Rule Type | Condition | Severity | Payload Fields |
|-----------|-----------|----------|----------------|
| health_degraded | overall_status in ('degraded','error','halted') | critical=error/halted, high=degraded | overall_status, failed_checks[], thresholds |
| weak_coverage | count(open, priority>=tier) > 0 | high=P1, medium=P2 | count, top_strategies[], worst_score, worst_run_id |
| drift_spike | drift_score >= 0.30 for 2 consecutive buckets | medium | drift_score, threshold, recent_values[], bucket_end |
| confidence_drop | trend_delta <= -0.05 | medium | delta, first_half_avg, second_half_avg, bucket_end |

### Resolution Pass

After upserting triggered conditions, resolve any that cleared:

```python
async def resolution_pass(
    self,
    workspace_id: UUID,
    triggered_keys: set[str],
    today: str
) -> list[AlertEvent]:
    """Resolve alerts whose condition cleared this run."""

    # Check daily singleton rules
    singleton_rules = ["health_degraded", "weak_coverage:P1", "weak_coverage:P2"]
    recovered = []

    for rule in singleton_rules:
        key = f"{rule}:{today}"
        if key not in triggered_keys:
            event = await self.resolve_if_active(workspace_id, key)
            if event:
                recovered.append(event)

    return recovered
```

### Job Result Structure

```python
{
    "status": "completed",
    "lock_acquired": True,
    "metrics": {
        "conditions_evaluated": 4,
        "alerts_triggered": 2,
        "alerts_new": 1,
        "alerts_updated": 1,
        "alerts_resolved": 1,
        "alerts_escalated": 0,
        "telegram_sent": 2
    },
    "by_rule_type": {
        "health_degraded": {"triggered": False, "resolved": True, "notified": True},
        "weak_coverage:P1": {"triggered": True, "new": True, "notified": True},
        "weak_coverage:P2": {"triggered": False},
        "drift_spike": {"triggered": False},
        "confidence_drop": {"triggered": False}
    },
    "job_run_id": "uuid"
}
```

---

## Telegram Notifier

### Configuration

```bash
# Required
TELEGRAM_BOT_TOKEN=123456:ABC-xyz...
TELEGRAM_CHAT_ID=-100123456789

# Optional
TELEGRAM_ENABLED=true              # Kill switch (default: true if token set)
TELEGRAM_TIMEOUT_SECS=10           # Request timeout (default: 10)
ADMIN_BASE_URL=https://rag.example.com  # For clickable links
```

### Notifier Implementation

```python
class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        base_url: str | None = None,
        timeout: float = 10.0
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = base_url
        self.timeout = timeout

    async def send_alert(
        self,
        event: AlertEvent,
        is_recovery: bool = False
    ) -> bool:
        """Send alert to Telegram. Returns True if successful."""
        message = self._format_message(event, is_recovery)

        # Truncate if too long (Telegram limit ~4096)
        if len(message) > 4000:
            message = message[:3950] + "\n\n(truncated...)"

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        for attempt in range(2):  # Retry once on transient errors
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                        json=payload
                    )
                    if resp.status_code == 200:
                        return True
                    if resp.status_code == 400:  # Bad request, don't retry
                        logger.warning("telegram_bad_request", detail=resp.text)
                        return False
                    # 429/5xx → retry
                    logger.warning("telegram_error", status=resp.status_code)
            except Exception as e:
                logger.warning("telegram_exception", attempt=attempt, error=str(e))

            if attempt == 0:
                await asyncio.sleep(1)  # Brief backoff before retry

        return False
```

### Notification Conditions

| Condition | Send Telegram? |
|-----------|----------------|
| New alert (`is_new = True`) | Yes |
| Recovery (resolved) | Yes |
| Severity escalation (e.g., medium→high) | Yes |
| Heartbeat update only | No |

### Message Format (HTML)

**Alert:**
```
🔴 <b>ALERT: Health Degraded</b>
━━━━━━━━━━━━━━━━━━━━━
Workspace: <code>trading-prod</code>
Severity: critical
Rule: <code>health_degraded</code>
Date: 2026-01-19

<b>Failed checks:</b>
• db: connection timeout
• qdrant: unhealthy
(+2 more…)

<a href="https://rag.example.com/admin/jobs/runs/{run_id}/detail">View Job Run</a>
Event: <code>{event_id[:8]}</code>
```

**Recovery:**
```
🟢 <b>RECOVERED: Health Degraded</b>
━━━━━━━━━━━━━━━━━━━━━
Workspace: <code>trading-prod</code>
Date: 2026-01-19

All checks passing.

<a href="https://rag.example.com/admin/jobs/runs/{run_id}/detail">View Job Run</a>
Event: <code>{event_id[:8]}</code>
```

**Severity Emoji:**
- critical: 🔴
- high: 🟠
- medium: 🟡
- recovery: 🟢

---

## Admin Endpoints

All endpoints require `X-Admin-Token` header.

### List Alerts

```
GET /admin/alerts
    ?workspace_id={uuid}           # Required
    &status=active,resolved        # Comma-separated, default: all
    &severity=critical,high        # Comma-separated filter
    &rule_type=health_degraded,weak_coverage  # Comma-separated filter
    &limit=50                      # Max 100, default 50
    &offset=0
```

**Ordering:**
- `status=active` → `last_seen_at DESC` (still happening first)
- Otherwise → `created_at DESC`

**Response:**
```json
{
    "items": [
        {
            "id": "uuid",
            "workspace_id": "uuid",
            "rule_type": "health_degraded",
            "severity": "critical",
            "status": "active",
            "rule_version": "v1",
            "dedupe_key": "health_degraded:2026-01-19",
            "payload": {
                "overall_status": "error",
                "failed_checks": ["db", "qdrant"],
                "thresholds": {"degraded_if": ["degraded", "error", "halted"]}
            },
            "source": "alert_evaluator",
            "job_run_id": "uuid",
            "created_at": "2026-01-19T10:15:00Z",
            "last_seen_at": "2026-01-19T10:30:00Z",
            "resolved_at": null,
            "acknowledged_at": null,
            "acknowledged_by": null
        }
    ],
    "total": 42,
    "limit": 50,
    "offset": 0
}
```

### Get Alert Detail

```
GET /admin/alerts/{id}
```

Returns single alert with full payload.

### Acknowledge Alert

```
POST /admin/alerts/{id}/acknowledge
    ?acknowledged_by=operator_name  # Optional
```

**Idempotent:** If already acknowledged, returns 200 with unchanged row.

**Response:**
```json
{
    "id": "uuid",
    "acknowledged_at": "2026-01-19T11:00:00Z",
    "acknowledged_by": "operator_name",
    "was_already_acknowledged": false
}
```

### Resolve Alert

```
POST /admin/alerts/{id}/resolve
```

**Idempotent:** If already resolved, returns 200 with unchanged row.

**Response:**
```json
{
    "id": "uuid",
    "resolved_at": "2026-01-19T11:05:00Z",
    "was_already_resolved": false
}
```

---

## Tests

### Unit Tests

1. **Upsert semantics**
   - New event creates row, returns `is_new=True`
   - Duplicate dedupe_key updates `last_seen_at`, returns `is_new=False`
   - Resolved event not updated (WHERE status='active')
   - Severity escalation detected

2. **Condition evaluation**
   - Health: degraded/error/halted → triggers, ok → doesn't trigger
   - Coverage: P1 count > 0 → triggers with high severity
   - Each rule type builds correct dedupe_key

3. **Resolution pass**
   - Clears active alerts when condition no longer met
   - Doesn't touch resolved alerts
   - Returns list of recovered events

4. **Telegram notifier**
   - Formats messages correctly (HTML escaping)
   - Truncates long messages
   - Retries on 5xx, doesn't retry on 400
   - Returns False on failure (doesn't raise)

### Integration Tests

1. **Full evaluator flow**
   - Mock health/coverage checks
   - Verify events created in DB
   - Verify Telegram called for new events only

2. **Idempotent actions**
   - Double-acknowledge returns 200 both times
   - Double-resolve returns 200 both times

---

## Rollout Plan

### Phase 1: Migration + Evaluator (no notifications)

1. Apply migration 071_alert_events.sql
2. Wire AlertEvaluatorJob to write events
3. Add `GET /admin/alerts` endpoint
4. Test: run evaluator, verify events in DB

### Phase 2: Telegram Delivery

1. Set `TELEGRAM_ENABLED=false` initially
2. Deploy TelegramNotifier
3. Test with `TELEGRAM_ENABLED=true` in staging
4. Enable in production

### Phase 3: Scheduling

1. Add pg_cron schedule (every 15 minutes):
   ```sql
   SELECT cron.schedule('evaluate-alerts', '*/15 * * * *',
       $$SELECT net.http_post(
           url := 'http://localhost:8000/admin/jobs/evaluate-alerts?workspace_id=...',
           headers := '{"X-Admin-Token": "..."}'::jsonb
       )$$);
   ```
2. Or use external scheduler (GHA cron, systemd timer)

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `TELEGRAM_ENABLED` | false | Master kill switch for Telegram delivery |
| `ALERT_EVALUATOR_DRY_RUN` | false | Evaluate but don't write events |

---

## Acceptance Criteria

- [ ] Migration applied, `alert_events` table exists
- [ ] AlertEvaluatorJob writes events on triggered conditions
- [ ] Dedupe keys prevent duplicate alerts within time bucket
- [ ] `last_seen_at` updated on heartbeat evaluations
- [ ] Resolution pass clears alerts when conditions clear
- [ ] Telegram sent for new alerts (when enabled)
- [ ] Telegram sent for recovery events (when enabled)
- [ ] Telegram not sent for heartbeat-only updates
- [ ] `GET /admin/alerts` returns filtered, paginated list
- [ ] Acknowledge/resolve endpoints are idempotent
- [ ] Job result includes detailed metrics
- [ ] All unit tests pass
- [ ] Integration test verifies full flow

---

## Future Enhancements (Deferred)

- `alert_rules` table for per-workspace threshold configuration
- Full admin UI (Recent Alerts panel, list page, detail page)
- Webhook sink for generic HTTP delivery (Slack, PagerDuty, etc.)
- Alert grouping/correlation
- Notification preferences per user
- Alert history trends/charts
