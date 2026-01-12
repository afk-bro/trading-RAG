# PR12: In-App Alerts Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add analytics alerting to surface drift spikes, confidence drops, and combo degradations proactively.

**Architecture:** Sink-based alert system with scheduled evaluation. AlertRule defines thresholds, AlertEvent tracks occurrences with state transitions, InAppSink persists to DB for UI display.

**Tech Stack:** PostgreSQL (alert_rules, alert_events tables), FastAPI endpoints, Jinja2 templates, PR11 JobRunner for scheduled evaluation.

---

## Scope

### In Scope (PR12)
- 3 rule types: Drift Spike, Confidence Drop, Combo
- In-app delivery only (InAppSink)
- Scheduled evaluation every 15 minutes
- Recent Alerts panel in Analytics Overview
- Full alert list + detail pages
- Acknowledge/unacknowledge actions

### Deferred (PR12.5+)
- Webhook sink (external notifications)
- Saved Views (filter bookmarks)
- Tier Fallback / Coverage Gap rules

---

## Data Model

### alert_rules (definitions)

```sql
CREATE TABLE alert_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    rule_type TEXT NOT NULL CHECK (rule_type IN ('drift_spike', 'confidence_drop', 'combo')),
    strategy_entity_id UUID,  -- NULL = all strategies
    regime_key TEXT,          -- NULL = all regimes
    timeframe TEXT,           -- NULL = default workspace timeframe
    enabled BOOLEAN DEFAULT true,
    config JSONB NOT NULL DEFAULT '{}',
    cooldown_minutes INT DEFAULT 60,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alert_rules_workspace ON alert_rules(workspace_id, enabled);
```

### alert_events (occurrences)

```sql
CREATE TABLE alert_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    rule_id UUID NOT NULL REFERENCES alert_rules(id),
    strategy_entity_id UUID NOT NULL,
    regime_key TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('drift_spike', 'confidence_drop', 'combo')),

    -- State (orthogonal concerns)
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved')),
    severity TEXT NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high')),

    -- Acknowledgment (separate from status)
    acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,

    -- Timestamps
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- Reset on resolved->active
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,

    -- Context for display + deep-link
    context_json JSONB NOT NULL DEFAULT '{}',
    fingerprint TEXT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(workspace_id, strategy_entity_id, regime_key, timeframe, rule_type, fingerprint)
);

-- Consistency constraints
ALTER TABLE alert_events ADD CONSTRAINT chk_ack_consistency CHECK (
    (acknowledged = false AND acknowledged_at IS NULL AND acknowledged_by IS NULL)
    OR (acknowledged = true AND acknowledged_at IS NOT NULL)
);

ALTER TABLE alert_events ADD CONSTRAINT chk_resolved_consistency CHECK (
    (status = 'active' AND resolved_at IS NULL)
    OR (status = 'resolved' AND resolved_at IS NOT NULL)
);

-- Indexes
CREATE INDEX idx_alert_events_active ON alert_events(workspace_id, status) WHERE status = 'active';
CREATE INDEX idx_alert_events_list ON alert_events(workspace_id, last_seen DESC);
CREATE INDEX idx_alert_events_filtered ON alert_events(workspace_id, status, severity, last_seen DESC);
CREATE INDEX idx_alert_events_needs_attention ON alert_events(workspace_id, last_seen DESC)
    WHERE status = 'active' AND acknowledged = false;
```

---

## Rule Types

### A) Drift Spike

**Input:** drift_score per bucket (already computed for drift panel)

**Config:**
```json
{
    "drift_threshold": 0.30,
    "consecutive_buckets": 2,
    "resolve_consecutive_buckets": 2,
    "hysteresis": 0.05
}
```

**Trigger:** drift_score >= drift_threshold for N consecutive buckets

**Resolve:** drift_score < (drift_threshold - hysteresis) for M consecutive buckets

**Severity:** medium

### B) Confidence Drop

**Input:** avg_confidence per bucket, first-half vs second-half delta (already computed)

**Config:**
```json
{
    "trend_threshold": 0.05
}
```

**Trigger:** confidence_trend <= -trend_threshold (second half avg is 5pts worse than first half)

**Resolve:** confidence_trend >= (-trend_threshold + hysteresis)

**Severity:** medium

### C) Combo (Drift + Confidence)

**Trigger:** Both Drift Spike and Confidence Drop conditions are met simultaneously

**Resolve:** Either underlying condition clears (OR logic, less sticky)

**Severity:** high (strongest signal - regime shift happening)

---

## Evaluation Engine

### EvalResult Structure

```python
@dataclass
class EvalResult:
    condition_met: bool       # Currently in alert condition
    condition_clear: bool     # Currently in clear/resolved condition
    trigger_value: float      # Value that triggered
    context: dict             # Additional context for display
    insufficient_data: bool = False
```

### Rule Evaluation (pure, stateless)

```python
def evaluate_drift_spike(self, buckets, config) -> EvalResult:
    activate_n = config.get("consecutive_buckets", 2)
    resolve_n = config.get("resolve_consecutive_buckets", activate_n)
    threshold = config.get("drift_threshold", 0.30)
    hysteresis = config.get("hysteresis", 0.05)

    if len(buckets) < max(activate_n, resolve_n):
        return EvalResult(insufficient_data=True, ...)

    recent_activate = buckets[-activate_n:]
    recent_resolve = buckets[-resolve_n:]

    condition_met = all(b.drift_score >= threshold for b in recent_activate)
    condition_clear = all(b.drift_score < (threshold - hysteresis) for b in recent_resolve)

    # Tie-break: prioritize alerting
    if condition_met:
        condition_clear = False

    return EvalResult(
        condition_met=condition_met,
        condition_clear=condition_clear,
        trigger_value=buckets[-1].drift_score,
        context={"threshold": threshold, "recent_values": [b.drift_score for b in recent_activate]}
    )

def evaluate_combo(self, drift_eval, confidence_eval) -> EvalResult:
    condition_met = drift_eval.condition_met and confidence_eval.condition_met
    condition_clear = drift_eval.condition_clear or confidence_eval.condition_clear

    if condition_met:
        condition_clear = False

    return EvalResult(
        condition_met=condition_met,
        condition_clear=condition_clear,
        context={"drift": drift_eval.context, "confidence": confidence_eval.context}
    )
```

### Transition Layer (handles cooldown + DB)

```python
def process_evaluation(self, eval_result, existing_event, rule, now):
    if eval_result.insufficient_data:
        return  # No change, don't update last_seen

    if eval_result.condition_met:
        if existing_event and existing_event.status == "active":
            # Still active: just update last_seen (no cooldown check)
            update_last_seen(existing_event.id, now)
            return

        # Potential activation (new or reactivation)
        if existing_event and (now - existing_event.activated_at).total_seconds() < rule.cooldown_minutes * 60:
            return  # Suppress reactivation during cooldown

        # UPSERT: activate (reset acknowledged on reactivation)
        upsert_activate(
            workspace_id=rule.workspace_id,
            strategy_entity_id=...,
            regime_key=...,
            timeframe=...,
            rule_type=rule.rule_type,
            fingerprint=...,
            context_json=eval_result.context,
            severity=get_severity(rule.rule_type),
            activated_at=now,
            last_seen=now,
            acknowledged=False  # Reset on reactivation
        )

    elif eval_result.condition_clear:
        if existing_event and existing_event.status == "active":
            resolve(existing_event.id, now)

    # else: unchanged -> no write
```

---

## Job Integration

### AlertEvaluatorJob

Integrates with PR11's JobRunner infrastructure.

**Job name:** `evaluate_alerts`

**Endpoint:** `POST /admin/jobs/evaluate-alerts?workspace_id={uuid}`

**Lock granularity:** Per-job per-workspace
```python
lock_key = hash(f"evaluate_alerts:{workspace_id}")
```

### Job Flow

1. Acquire advisory lock
2. Load enabled rules for workspace
3. Expand rule scope (NULL strategy → enumerate all strategies)
4. Group by (strategy_entity_id, timeframe) for batch fetching
5. Fetch drift/confidence series (one query per group)
6. Evaluate all (strategy, regime, timeframe) tuples
7. Batch upsert alert events (single transaction)
8. Record job_run with metrics

### Job Metrics

```json
{
    "rules_loaded": 5,
    "tuples_evaluated": 42,
    "tuples_skipped_insufficient_data": 3,
    "activations_suppressed_cooldown": 1,
    "alerts_activated": 2,
    "alerts_resolved": 1,
    "db_upserts": 2,
    "db_updates": 1
}
```

### Schedule

- **Default:** Every 15 minutes via external cron/GHA
- **Pattern:** Scheduler enumerates active workspaces, calls endpoint per workspace
- **Optional:** "Evaluate Now" button for admin debugging (same endpoint, manual trigger)

---

## API Endpoints

### Alert Rules (CRUD)

```
GET    /admin/alerts/rules?workspace_id={uuid}           # List rules
POST   /admin/alerts/rules?workspace_id={uuid}           # Create rule
GET    /admin/alerts/rules/{id}                          # Get rule
PATCH  /admin/alerts/rules/{id}                          # Update rule
DELETE /admin/alerts/rules/{id}                          # Delete rule
```

### Alert Events (read + actions)

```
GET    /admin/alerts?workspace_id={uuid}                 # List events
GET    /admin/alerts/{id}                                # Event detail
POST   /admin/alerts/{id}/acknowledge                    # Acknowledge
POST   /admin/alerts/{id}/unacknowledge                  # Unacknowledge
```

### List Filters (GET /admin/alerts)

| Param | Values | Default |
|-------|--------|---------|
| workspace_id | UUID | required |
| status | active, resolved, all | all |
| severity | low, medium, high | - |
| acknowledged | true, false | - |
| rule_type | drift_spike, confidence_drop, combo | - |
| timeframe | 5m, 1h, 1d, ... | - |
| strategy_entity_id | UUID | - |
| regime_key | string | - |
| from, to | ISO timestamps | - |
| limit | 1-100 | 50 |
| offset | int | 0 |

### Response (includes joined data)

```json
{
    "items": [
        {
            "id": "uuid",
            "rule_type": "drift_spike",
            "strategy_name": "bb_reversal",
            "regime_key": "high_vol/uptrend",
            "timeframe": "1h",
            "status": "active",
            "severity": "medium",
            "acknowledged": false,
            "first_seen": "2026-01-12T10:15:00Z",
            "last_seen": "2026-01-12T10:30:00Z",
            "context_json": {...}
        }
    ],
    "total": 42,
    "limit": 50,
    "offset": 0
}
```

### Rule Config Validation

Enforce schema per rule_type on POST/PATCH:

- **drift_spike:** drift_threshold (float), consecutive_buckets (int), resolve_consecutive_buckets (int), hysteresis (float)
- **confidence_drop:** trend_threshold (float), hysteresis (float)
- **combo:** No additional config (composes drift + confidence)

---

## UI Components

### 1. Recent Alerts Panel (Analytics Overview)

**Location:** Top-right card, near drift summary

**Features:**
- Toggle: Active (default) / All (last 7d)
- Shows 5-7 alerts, newest first
- Row: severity badge, rule type, strategy, regime, status, timestamp
- Row click → /admin/alerts/{id}
- Hover reveals "View in Analytics" secondary link
- Footer: "View all →" link to /admin/alerts

**Nav badge:** Count of active+high alerts (optional dot for active+medium)

### 2. Alert List Page (/admin/alerts)

**Default filter:** status=active, acknowledged=false ("Needs attention")

**Quick filter chips:** Needs attention | Active | Resolved | All

**Filter bar:** Status, Severity, Rule Type, Strategy dropdown, Date range

**Table columns:**
- Last seen
- Severity (badge)
- Rule type
- Strategy
- Regime
- Timeframe
- Status
- Acknowledged
- Actions (Ack button)

**Behavior:**
- Row clickable → detail page
- Ack button: stopPropagation, inline action
- Pagination: limit/offset with Previous/Next

### 3. Alert Detail Page (/admin/alerts/{id})

**Header:** Severity badge + Rule type + Status badge

**"Why it triggered" card:**

For drift_spike:
```
Drift Score: 0.35 (threshold: 0.30)
Consecutive buckets: 2 of 2
Recent values: [0.32, 0.35]
```

For confidence_drop:
```
Trend Delta: -0.08 (threshold: -0.05)
First half avg: 0.72
Second half avg: 0.64
```

For combo:
```
[Drift context]
[Confidence context]
```

**Sparkline:** Last K buckets (reuse existing mini-chart style)

**Timeline:**
- Activated: Jan 11, 14:15
- Last seen: Jan 11, 14:30
- (if resolved) Resolved: Jan 11, 15:00

**Actions:**
- "View in Analytics" button (deep-link using context_json)
- "Acknowledge" / "Unacknowledge" button

### Deep-link Behavior

"View in Analytics" lands on analytics page with:
- strategy_entity_id filter applied
- timeframe selected
- regime_key filter applied (or highlight in Top Drift Drivers if no regime filter UI)

---

## context_json Schema

### drift_spike

```json
{
    "drift_threshold": 0.30,
    "consecutive_buckets": 2,
    "hysteresis": 0.05,
    "current_drift": 0.35,
    "recent_drifts": [0.32, 0.35],
    "deep_link": {
        "strategy_entity_id": "uuid",
        "timeframe": "1h",
        "regime_key": "high_vol/uptrend"
    }
}
```

### confidence_drop

```json
{
    "trend_threshold": 0.05,
    "trend_delta": -0.08,
    "first_half_avg": 0.72,
    "second_half_avg": 0.64,
    "deep_link": {
        "strategy_entity_id": "uuid",
        "timeframe": "1h",
        "regime_key": "high_vol/uptrend"
    }
}
```

### combo

```json
{
    "drift_context": { ... },
    "confidence_context": { ... },
    "deep_link": { ... }
}
```

---

## Severities

| Rule Type | Severity |
|-----------|----------|
| drift_spike | medium |
| confidence_drop | medium |
| combo | high |

---

## Testing Strategy

### Unit Tests
- Rule evaluation logic (each rule type)
- Transition layer (cooldown, state changes)
- Config validation per rule type

### Integration Tests
- Job execution with mocked DB
- API endpoints (CRUD, list filters, ack/unack)
- Upsert idempotency

### E2E Tests
- Create rule → run job → verify alert appears
- Alert lifecycle: active → ack → resolve
- UI panel + list + detail navigation

---

## Migration Plan

1. Apply migrations (alert_rules, alert_events tables)
2. Deploy code (endpoints, job, UI)
3. Create default rules per workspace (optional bootstrap)
4. Enable scheduled job in cron/GHA
5. Monitor job metrics + alert quality

---

## Future (PR12.5+)

- **Webhook sink:** POST to user-configured URL with retry/backoff
- **Saved Views:** Bookmark filters, optionally create alert from view
- **Tier Fallback rule:** Alert when tier_3 usage exceeds threshold
- **Coverage Gap rule:** Alert when regime has insufficient data
