# RAG Core Prometheus Alerting

Production-ready alerting rules for rag-core service monitoring.

## Quick Start

```bash
# Copy rules to Prometheus rules directory
cp rules/rag_core_alerts.yml /etc/prometheus/rules/

# Or add to prometheus.yml
rule_files:
  - /path/to/rag_core_alerts.yml

# Reload Prometheus
curl -X POST http://localhost:9090/-/reload
```

## Required Metrics

The alerts expect these metrics to be exposed at `/metrics`:

### Platform Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | `route`, `status`, `method` | HTTP request count |
| `http_request_duration_seconds` | Histogram | `route`, `method` | Request latency |
| `up` | Gauge | `job`, `instance` | Scrape target health |

### Database Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `db_pool_acquire_seconds` | Histogram | - | Connection acquire time |
| `db_connection_errors_total` | Counter | `error_type` | Connection failures |

### Qdrant Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `qdrant_errors_total` | Counter | `op`, `collection` | Qdrant operation errors |
| `qdrant_request_duration_seconds` | Histogram | `op`, `collection` | Qdrant request latency |

### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_requests_total` | Counter | `provider`, `model` | LLM request count |
| `llm_degraded_total` | Counter | `reason_code` | Fallback/degraded responses |
| `llm_errors_total` | Counter | `provider`, `error_type` | LLM errors |
| `llm_timeout_total` | Counter | `provider` | LLM timeouts |

### KB Recommend Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `kb_recommend_requests_total` | Counter | `status`, `confidence_bucket` | Recommend requests |
| `kb_embed_errors_total` | Counter | - | Embedding failures |
| `kb_recommend_timeout_total` | Counter | - | Recommend timeouts |

### Backtest Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tune_runs_total` | Counter | `status` | Tune run completions |
| `tune_run_duration_seconds` | Histogram | - | Tune execution time |
| `run_plan_failures_total` | Counter | - | Run plan execution failures |

### Retention Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `retention_job_runs_total` | Counter | `status`, `table` | Retention job runs |
| `retention_job_last_run_timestamp` | Gauge | - | Last successful run time |
| `pg_table_size_bytes` | Gauge | `table` | PostgreSQL table sizes |

### Idempotency Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `idempotency_keys_total` | Gauge | - | Total idempotency keys in table |
| `idempotency_expired_pending_total` | Gauge | - | Expired keys pending prune |
| `idempotency_pending_requests_total` | Gauge | - | Active pending requests |
| `idempotency_oldest_pending_age_minutes` | Gauge | - | Age of oldest pending request |
| `idempotency_oldest_expired_age_hours` | Gauge | - | Age of oldest expired key (pg_cron health) |

### SSE Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `sse_queue_drops_total` | Counter | `topic` | Dropped events |
| `sse_subscribers_count` | Gauge | `topic` | Active subscribers |

### Ingestion Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `ingest_failures_total` | Counter | `source_type` | Ingestion failures |
| `ingest_queue_pending_count` | Gauge | - | Pending ingestion jobs |

### Pine Discovery Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pine_scripts_pending_ingest` | Gauge | - | Scripts needing ingest/re-ingest |
| `pine_discovery_last_run_timestamp` | Gauge | - | Unix epoch of last discovery run |
| `pine_discovery_last_success_timestamp` | Gauge | - | Unix epoch of last successful run |
| `pine_ingest_failed_total` | Counter | - | Pine script ingest failures |
| `pine_ingest_chunks_total` | Counter | - | Chunks created from Pine scripts |
| `pine_scripts_total` | Gauge | `status` | Scripts by discovery status |

### Pine Repos Metrics (GitHub Registry)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pine_repos_total` | Gauge | - | Total registered GitHub repos |
| `pine_repos_enabled` | Gauge | - | Enabled repos (scanned) |
| `pine_repos_pull_failed` | Gauge | - | Repos with git pull failures |
| `pine_repos_stale` | Gauge | - | Repos not scanned in 7+ days |
| `pine_repos_oldest_scan_age_hours` | Gauge | - | Age of oldest repo scan |
| `pine_repo_scan_runs_total` | Counter | `status` | Repo scan run completions |
| `pine_repo_scan_duration_seconds` | Histogram | - | Repo scan duration |
| `pine_repo_scripts_discovered_total` | Counter | `change_type` | Scripts discovered (new/updated/deleted) |

## Label Assumptions

The rules assume these label conventions:

- `job="rag-core"` - Prometheus job name for the service
- `status` - HTTP status code (e.g., "200", "500") or result status ("ok", "none", "degraded")
- `route` - API route path (e.g., "/query", "/kb/trials/recommend")
- `op` - Operation type (e.g., "search", "upsert", "delete")
- `confidence_bucket` - Bucketed confidence: "high" (>=0.7), "medium" (>=0.4), "low" (<0.4), "none"

## Threshold Tuning

These are starting defaults - tune based on your traffic patterns:

| Alert | Default | Tune When |
|-------|---------|-----------|
| `HighErrorRate` | >5% for 2m | Adjust for baseline error rate |
| `HighLatencyP95` | >2s for 5m | Lower for latency-sensitive endpoints |
| `DBPoolExhausted` | P95 >1s for 3m | Adjust based on pool size |
| `LLMDegradedHigh` | >10% for 5m | Lower if zero degradation expected |
| `KBWeakCoverageHigh` | >20% for 10m | Lower for mature KB |
| `TuneFailureRateHigh` | >20% for 30m | Lower for stable strategies |
| `RetentionJobNotRunning` | >48h | Match your cron schedule |
| `IdempotencyExpiredPending` | >100 for 30m | Lower if strict hygiene required |
| `IdempotencyPruneStale` | >48h | Match pg_cron schedule (daily) |
| `PineDiscoveryPendingHigh` | >50 for 15m | Lower if ingest should be immediate |
| `PineDiscoveryStale` | >1h | Match your discovery cron schedule |
| `PineIngestErrorsCritical` | >10/h | Lower if zero errors expected |
| `PineRepoPullFailed` | >0 for 5m | Immediate action on any failure |
| `PineRepoStale` | >0 for 1h | Match your repo scan schedule |
| `PineRepoOldestScanAged` | >168h (7d) | Lower for active repos |
| `PineRepoScanErrors` | >3/h | Lower for strict error handling |

## Loading Rules

### Prometheus (Direct)

```yaml
# prometheus.yml
rule_files:
  - /etc/prometheus/rules/rag_core_alerts.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Grafana Alerting

1. Navigate to Alerting > Alert rules
2. Click "Import"
3. Select `rag_core_alerts.yml`
4. Map to your Grafana data sources

### Kubernetes (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: rag-core-alerts
  namespace: monitoring
  labels:
    prometheus: kube-prometheus
    role: alert-rules
spec:
  # Paste contents of rag_core_alerts.yml groups here
  groups:
    - name: rag_core.platform
      rules:
        # ... rules from rag_core_alerts.yml
```

Or use `yq` to convert:

```bash
yq eval '.groups' rules/rag_core_alerts.yml | \
  yq eval '{"apiVersion": "monitoring.coreos.com/v1", "kind": "PrometheusRule", "metadata": {"name": "rag-core-alerts", "namespace": "monitoring"}, "spec": {"groups": .}}' - \
  > rag-core-prometheusrule.yaml
```

## Alert Routing (Alertmanager)

Example routing by severity:

```yaml
# alertmanager.yml
route:
  receiver: 'slack-ops'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-oncall'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-ops'
    - match:
        severity: info
      receiver: 'slack-info'

receivers:
  - name: 'pagerduty-oncall'
    pagerduty_configs:
      - service_key: '<PD_SERVICE_KEY>'
  - name: 'slack-ops'
    slack_configs:
      - api_url: '<SLACK_WEBHOOK>'
        channel: '#ops-alerts'
  - name: 'slack-info'
    slack_configs:
      - api_url: '<SLACK_WEBHOOK>'
        channel: '#ops-info'
```

## Subsystem Reference

Each alert has a `subsystem` label for filtering:

| Subsystem | Scope |
|-----------|-------|
| `platform` | HTTP/service health |
| `db` | PostgreSQL/connection pool |
| `qdrant` | Vector database |
| `llm` | Language model providers |
| `kb` | Knowledge base recommendations |
| `backtests` | Tune/run plan execution |
| `retention` | Data cleanup jobs |
| `idempotency` | Idempotency key hygiene |
| `sse` | Server-Sent Events |
| `ingestion` | Document ingestion |
| `pine_discovery` | Pine script discovery & KB ingest |

## Related Documentation

- [Runbooks](../../docs/ops/runbooks.md) - Incident response procedures
- [Alerting Rules (Sentry)](../../docs/ops/alerting-rules.md) - Sentry-specific alerts
- [System Health](../../docs/ops/system-health.md) - Dashboard reference
