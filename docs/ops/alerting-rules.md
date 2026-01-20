# Alerting Rules

Minimum viable alerts for Trading RAG service monitoring.

## Prometheus Alerting Rules

Prometheus alerting rules for Grafana Alerting or Alertmanager. Copy to your Prometheus rules directory or configure via Grafana UI.

### Critical Alerts

```yaml
groups:
  - name: trading-rag-critical
    rules:
      # High 5xx error rate indicates service degradation
      - alert: TradingRAG_HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High 5xx error rate ({{ $value | humanizePercentage }})"
          description: "Error rate exceeds 5% for 2+ minutes"
          runbook_url: "/docs/ops/runbooks.md#high-error-rate"

      # Service not responding
      - alert: TradingRAG_ServiceDown
        expr: up{job="trading-rag"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Trading RAG service is down"
          description: "Prometheus cannot scrape /metrics endpoint"

      # Database connection pool exhausted
      - alert: TradingRAG_DBPoolExhausted
        expr: |
          histogram_quantile(0.95, sum by (le) (rate(db_pool_acquire_seconds_bucket[5m]))) > 1
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "DB pool acquire P95 > 1s ({{ $value | humanizeDuration }})"
          description: "Connection pool may be exhausted"

      # Qdrant errors sustained
      - alert: TradingRAG_QdrantErrors
        expr: increase(qdrant_errors_total[5m]) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High Qdrant error rate ({{ $value }} in 5m)"
          description: "Vector database may be unavailable"
```

### Warning Alerts

```yaml
      # High latency indicates performance degradation
      - alert: TradingRAG_HighLatencyP95
        expr: |
          histogram_quantile(0.95, sum by (route) (rate(http_request_duration_seconds_bucket[5m]))) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency on {{ $labels.route }} ({{ $value | humanizeDuration }})"
          description: "Response time degraded for 5+ minutes"

      # LLM fallback rate elevated
      - alert: TradingRAG_LLMDegradedHigh
        expr: |
          sum(rate(llm_degraded_total[5m])) / sum(rate(llm_requests_total[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM fallback rate > 10% ({{ $value | humanizePercentage }})"
          description: "LLM provider may be rate-limited or unavailable"

      # KB recommend quality degraded
      - alert: TradingRAG_KBWeakCoverageHigh
        expr: |
          sum(rate(kb_recommend_requests_total{status="none"}[15m]))
          / sum(rate(kb_recommend_requests_total[15m])) > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "KB weak coverage rate > 20% ({{ $value | humanizePercentage }})"
          description: "High percentage of requests returning no recommendations"

      # Tune failure rate elevated
      - alert: TradingRAG_TuneFailureRateHigh
        expr: |
          increase(tune_runs_total{status="failed"}[1h])
          / increase(tune_runs_total{status=~"completed|failed"}[1h]) > 0.2
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Tune failure rate > 20% ({{ $value | humanizePercentage }})"
          description: "Backtest tuning may have systemic issues"

      # Embedding errors
      - alert: TradingRAG_EmbeddingErrors
        expr: increase(kb_embed_errors_total[15m]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Embedding errors detected ({{ $value }} in 15m)"
          description: "Check embedding provider availability"

      # SSE queue drops
      - alert: TradingRAG_SSEQueueDrops
        expr: increase(sse_queue_drops_total[5m]) > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "SSE events being dropped"
          description: "Client queues may be full, check subscriber health"

      # Retention job failures
      - alert: TradingRAG_RetentionJobFailed
        expr: increase(retention_job_runs_total{status="failure"}[24h]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Retention job failed"
          description: "Check pg_cron logs and retention_job_log table"
```

### Info Alerts

```yaml
      # Low confidence recommendations
      - alert: TradingRAG_KBLowConfidence
        expr: |
          sum(rate(kb_recommend_requests_total{confidence_bucket=~"low|none"}[15m]))
          / sum(rate(kb_recommend_requests_total[15m])) > 0.4
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "High rate of low-confidence recommendations"
          description: "Consider expanding KB coverage or adjusting thresholds"

      # High tune duration
      - alert: TradingRAG_TuneDurationHigh
        expr: |
          histogram_quantile(0.95, sum by (le) (rate(tune_run_duration_seconds_bucket[1h]))) > 1800
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "Tune duration P95 > 30min"
          description: "Tunes taking longer than expected"
```

---

## Sentry Alerts

### Critical Alerts (PagerDuty / On-call)

| Alert Name | Fingerprint / Query | Threshold | Action |
|------------|---------------------|-----------|--------|
| KB Recommend Timeout | `fingerprint:["kb", "recommend_timeout"]` | >3 in 5min | Check Qdrant connectivity, resource usage |
| Qdrant Error | `fingerprint:["kb", "qdrant_error"]` | >5 in 5min | Verify Qdrant health, check disk space |
| Embed Error | `fingerprint:["kb", "embed_error"]` | >5 in 5min | Check Ollama service, model availability |
| Readiness Failing | `/ready` returns 503 | >3min | Check all dependencies via /ready response |

### Warning Alerts (Slack #ops)

| Alert Name | Query | Threshold | Action |
|------------|-------|-----------|--------|
| KB Status None Rate | `tag:kb_status=none` | >20% of requests in 15min | Review filter thresholds, check data quality |
| KB Status Degraded Rate | `tag:kb_status=degraded` | >30% of requests in 15min | May need filter relaxation |
| P95 Latency High | `measurement:kb.total_ms` | P95 > 5000ms | Check Qdrant index, embedding service |
| Low Confidence | `tag:kb_confidence=low OR tag:kb_confidence=none` | >40% in 15min | Data quality issue |

## Sentry Tag Reference

Use these tags for filtering and dashboard queries:

| Tag | Values | Purpose |
|-----|--------|---------|
| `service` | `trading-rag` | Service identification |
| `kb_status` | `ok`, `degraded`, `none` | Recommendation result status |
| `kb_confidence` | `high`, `medium`, `low`, `none` | Bucketed confidence (0.7+, 0.4+, <0.4, null) |
| `strategy` | Strategy name | Filter by strategy type |
| `workspace_id` | UUID | Multi-tenant filtering |
| `embed_model` | Model name | Track model performance |
| `collection` | Collection name | Track collection usage |
| `objective` | Objective type | Filter by optimization goal |

## Sentry Measurement Reference

Use for performance dashboards:

| Measurement | Unit | Description |
|-------------|------|-------------|
| `kb.total_ms` | millisecond | End-to-end recommend time |
| `kb.embed_ms` | millisecond | Query embedding time |
| `kb.qdrant_ms` | millisecond | Vector search time |
| `kb.rerank_ms` | millisecond | Reranking time |
| `kb.regime_ms` | millisecond | Regime computation time |
| `kb.aggregate_ms` | millisecond | Parameter aggregation time |

## Prometheus Metrics (Optional)

If Prometheus is deployed, scrape `/metrics` endpoint:

| Metric | Type | Labels | Alert Threshold |
|--------|------|--------|-----------------|
| `kb_recommend_total` | Counter | `status`, `confidence` | Rate change alerts |
| `kb_recommend_duration_seconds` | Histogram | `status` | P95 > 5s |
| `kb_embed_errors_total` | Counter | - | Rate > 1/min |
| `kb_qdrant_errors_total` | Counter | - | Rate > 1/min |

## Example Sentry Alert Configuration

```yaml
# Sentry Alert Rule - KB Recommend Timeout
name: "KB Recommend Timeout"
conditions:
  - type: event_frequency
    interval: 5m
    value: 3
filters:
  - type: tagged_event
    key: fingerprint
    match: "kb, recommend_timeout"
actions:
  - type: notify_integration
    integration: pagerduty
    priority: high
```

```yaml
# Sentry Alert Rule - High None Rate
name: "KB Status None Rate High"
conditions:
  - type: event_frequency_percent
    interval: 15m
    value: 20
    comparison_type: percent
filters:
  - type: tagged_event
    key: kb_status
    match: "none"
actions:
  - type: notify_integration
    integration: slack
    channel: "#ops-alerts"
```

## Dashboard Queries

### Performance Overview
```
# P95 latency by status
measurement:kb.total_ms group by tag:kb_status percentile(95)

# Status distribution
count() by tag:kb_status
```

### Error Analysis
```
# Errors by fingerprint
fingerprint:kb.* count() group by fingerprint

# Errors by workspace
tag:fingerprint:kb.* count() group by tag:workspace_id
```

---

## Internal Ops Alerts System

The internal ops alerts system (`app/services/ops_alerts/`) provides business-level alerting with Telegram notifications. These are higher-level alerts that evaluate domain-specific conditions beyond infrastructure metrics.

### Rule Types

| Rule Type | Severity | Description | Data Source |
|-----------|----------|-------------|-------------|
| `health_degraded` | HIGH/CRITICAL | System health is degraded or in error state | Health snapshot |
| `weak_coverage:P1` | HIGH | P1 priority coverage gaps exist | Coverage stats |
| `weak_coverage:P2` | MEDIUM | P2 priority coverage gaps exist | Coverage stats |
| `drift_spike` | MEDIUM | Match quality drifted from baseline | Match run stats |
| `confidence_drop` | MEDIUM | Match confidence below threshold | Match run stats |
| `strategy_confidence_low` | MEDIUM/HIGH | Strategy version confidence score low | Strategy intel |

### Strategy Confidence Alert (`strategy_confidence_low`)

Triggers when a strategy version's confidence score drops below thresholds for consecutive snapshots.

**Thresholds:**
- **Warn (MEDIUM)**: `confidence_score < 0.35`
- **Critical (HIGH)**: `confidence_score < 0.20`

**Persistence Gate:** Requires 2 consecutive snapshots below threshold to reduce noise from transient dips.

**Auto-Resolution (Hysteresis):**
- Clear warn: `confidence_score > 0.40`
- Clear critical: `confidence_score > 0.25`

**Dedupe Key Format:** `strategy_confidence_low:{version_id}:{severity_bucket}:{date}`

**Alert Payload:**
```json
{
  "version_id": "uuid",
  "strategy_id": "uuid",
  "strategy_name": "Strategy Name",
  "version_number": 1,
  "confidence_score": 0.28,
  "regime": "trend-up|volatility-normal",
  "as_of_ts": "2026-01-19T14:00:00Z",
  "computed_at": "2026-01-19T14:05:00Z",
  "weak_components": [
    {"name": "drawdown", "score": 0.2},
    {"name": "stability", "score": 0.3},
    {"name": "data_freshness", "score": 0.4}
  ],
  "consecutive_low_count": 2,
  "thresholds": {
    "warn": 0.35,
    "critical": 0.20,
    "persistence_required": 2
  }
}
```

**Response Actions:**
1. Review weak components to identify root cause
2. Check recent market regime changes
3. Review backtest performance vs live conditions
4. Consider pausing version if critical persists

### Telegram Notification Routing

Alerts are routed to Telegram forum topics based on category:

| Rule Category | Topic |
|---------------|-------|
| Health alerts | Health topic |
| Strategy alerts (`strategy_confidence_low`) | Strategy topic |
| Coverage alerts | Coverage topic |
| Drift/Confidence alerts | Quality topic |

### Admin Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /admin/ops-alerts` | List active alerts |
| `POST /admin/ops-alerts/{id}/acknowledge` | Acknowledge alert |
| `POST /admin/ops-alerts/{id}/resolve` | Mark alert resolved |
| `POST /admin/ops-alerts/{id}/reopen` | Reopen resolved alert |
