# Grafana Dashboards

Pre-built Grafana dashboard provisioning files for Trading RAG observability.

## Dashboards

| Dashboard | UID | Purpose |
|-----------|-----|---------|
| Platform Overview | `trading-rag-overview` | RPS, latency, errors, DB pool, Qdrant health |
| RAG Quality & Coverage | `trading-rag-quality` | KB recommend quality, LLM fallbacks, embedding errors |
| Backtest Ops | `trading-rag-backtest` | Tune throughput, duration, trials, retention jobs |

## Installation

### Option 1: Grafana Provisioning (Recommended)

Mount the dashboards directory in your Grafana container:

```yaml
# docker-compose.yml
services:
  grafana:
    volumes:
      - ./dashboards:/etc/grafana/provisioning/dashboards/trading-rag
```

Create a provisioning config at `/etc/grafana/provisioning/dashboards/trading-rag.yaml`:

```yaml
apiVersion: 1
providers:
  - name: 'trading-rag'
    orgId: 1
    folder: 'Trading RAG'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards/trading-rag
```

### Option 2: Manual Import

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload JSON file or paste contents
4. Select your Prometheus datasource

## Prerequisites

### Prometheus Datasource

All dashboards use a `${datasource}` variable. Configure your Prometheus datasource in Grafana and select it from the dropdown.

### Required Metrics

These metrics must be exposed by the Trading RAG `/metrics` endpoint:

**Platform Overview**:
- `http_requests_total{route, method, status}`
- `http_request_duration_seconds_bucket{route, method}`
- `db_pool_acquire_seconds_bucket`
- `qdrant_request_duration_seconds_bucket{op}`
- `qdrant_errors_total{op}`
- `llm_degraded_total{reason_code}`
- `tune_runs_total{status}`

**RAG Quality & Coverage**:
- `kb_recommend_requests_total{status, confidence_bucket}`
- `kb_recommend_fallback_total{type}`
- `kb_recommend_latency_ms_bucket`
- `kb_qdrant_latency_ms_bucket`
- `kb_embed_latency_ms_bucket`
- `kb_embed_errors_total`
- `kb_qdrant_errors_total`
- `llm_requests_total{provider, status, reason_code}`
- `llm_degraded_total{reason_code}`
- `embedding_duration_seconds_bucket{provider}`

**Backtest Ops**:
- `tune_runs_total{status}` (started, completed, failed, cancelled)
- `tune_run_duration_seconds_bucket`
- `tune_trials_total{status}` (completed, failed)
- `retention_rows_deleted_total{table}`
- `retention_job_runs_total{job_name, status}`

## Panel Reference

### Platform Overview

| Panel | Query | Purpose |
|-------|-------|---------|
| Request Rate | `sum(rate(http_requests_total[5m])) by (route)` | Traffic by endpoint |
| 5xx Error Rate | `sum(rate(http_requests_total{status=~"5.."}[5m]))` | Server error rate |
| P95 Latency by Route | `histogram_quantile(0.95, ...)` | Response time hotspots |
| DB Pool Acquire P95 | `histogram_quantile(0.95, ...)` | Connection pool health |
| Qdrant P95 by Op | `histogram_quantile(0.95, ...)` | Vector DB performance |
| Qdrant Errors | `increase(qdrant_errors_total[1h])` | Vector DB reliability |
| LLM Degraded Events | `sum by (reason_code) (rate(...))` | LLM fallback visibility |
| Tune Runs by Status | `increase(tune_runs_total[1h])` | Backtest throughput |

### RAG Quality & Coverage

| Panel | Query | Purpose |
|-------|-------|---------|
| KB Recommend by Status | `sum(rate(kb_recommend_requests_total[5m])) by (status)` | Quality funnel |
| Confidence Distribution | `sum by (confidence_bucket) (rate(...))` | Recommendation quality |
| Fallback Types | `sum by (type) (rate(kb_recommend_fallback_total[5m]))` | Degradation visibility |
| LLM Degraded by Reason | `sum by (reason_code) (rate(...))` | LLM failure modes |
| LLM Fallback Rate | `sum(rate(llm_degraded_total[5m])) / sum(rate(llm_requests_total[5m]))` | Overall LLM health |
| KB Errors (1h) | `increase(kb_embed_errors_total[1h])` | Embedding failures |
| KB Recommend Latency | `histogram_quantile(0.95, ...)` | E2E latency |
| Component Latency P95 | Qdrant + Embed latencies | Bottleneck identification |

### Backtest Ops

| Panel | Query | Purpose |
|-------|-------|---------|
| Tunes (24h) | `increase(tune_runs_total[24h])` | Daily throughput |
| Tune Duration P95 | `histogram_quantile(0.95, ...)` | Run time monitoring |
| Active Tunes | `started - (completed + failed + cancelled)` | Concurrency estimate |
| Failure Rate (24h) | `failed / (completed + failed)` | Reliability tracking |
| Tune Runs by Status | `increase(tune_runs_total[1h])` | Status breakdown |
| Tune Duration Distribution | P50/P95/P99 histograms | Duration visibility |
| Trials by Status | `increase(tune_trials_total[1h])` | Trial throughput |
| Trial Failure Rate | `failed / total` | Trial reliability |
| Retention Rows Deleted | `increase(retention_rows_deleted_total[24h])` | Cleanup activity |
| Retention Job Runs | Success/failure by job | Job reliability |

## Customization

### Adding Workspace Filter

To add workspace-scoped filtering, add a variable:

```json
{
  "name": "workspace_id",
  "type": "query",
  "query": "label_values(http_requests_total, workspace_id)"
}
```

Then update queries to include `{workspace_id="$workspace_id"}`.

### Adjusting Thresholds

Panel thresholds are configured in `fieldConfig.defaults.thresholds`. Adjust values based on your SLOs:

```json
"thresholds": {
  "mode": "absolute",
  "steps": [
    { "color": "green", "value": null },
    { "color": "yellow", "value": 0.5 },
    { "color": "red", "value": 1.0 }
  ]
}
```

## Alerting

For alerting rules, see `docs/ops/alerting-rules.md`. Alerts are configured separately from dashboards in Grafana or via external alert managers.
