# Alerting Rules

Minimum viable alerts for Trading RAG service monitoring.

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
