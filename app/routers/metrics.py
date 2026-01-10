"""Prometheus metrics endpoint for Trading RAG Pipeline."""

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

router = APIRouter()

# Request metrics
REQUEST_COUNT = Counter(
    "trading_rag_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "trading_rag_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Ingestion metrics
DOCUMENTS_INGESTED = Counter(
    "trading_rag_documents_ingested_total",
    "Total number of documents ingested",
    ["source_type", "status"],
)

CHUNKS_CREATED = Counter(
    "trading_rag_chunks_created_total",
    "Total number of chunks created",
    ["source_type"],
)

EMBEDDINGS_GENERATED = Counter(
    "trading_rag_embeddings_generated_total",
    "Total number of embeddings generated",
    ["embed_model"],
)

# Query metrics
QUERIES_TOTAL = Counter(
    "trading_rag_queries_total",
    "Total number of queries",
    ["mode", "status"],
)

QUERY_RESULTS = Histogram(
    "trading_rag_query_results_count",
    "Number of results returned per query",
    buckets=[0, 1, 2, 3, 5, 10, 15, 20, 30, 50],
)

# Service health metrics
SERVICE_UP = Gauge(
    "trading_rag_service_up",
    "Service availability (1=up, 0=down)",
    ["component"],
)

# Connection pool metrics
DB_POOL_SIZE = Gauge(
    "trading_rag_db_pool_size",
    "Current database connection pool size",
)

DB_POOL_AVAILABLE = Gauge(
    "trading_rag_db_pool_available",
    "Available connections in database pool",
)

# Qdrant metrics
QDRANT_VECTORS_COUNT = Gauge(
    "trading_rag_qdrant_vectors_total",
    "Total vectors in Qdrant collection",
    ["collection"],
)

# =============================================================================
# Trading KB Metrics
# =============================================================================

KB_RECOMMEND_REQUESTS = Counter(
    "kb_recommend_requests_total",
    "Total KB recommend requests",
    [
        "status",
        "confidence_bucket",
    ],  # confidence_bucket: high (>0.7), medium (0.4-0.7), low (<0.4), none
)

KB_RECOMMEND_FALLBACK = Counter(
    "kb_recommend_fallback_total",
    "KB recommend fallback usage",
    ["type"],  # relaxed, metadata_only, repaired, incomplete_regime
)

KB_RECOMMEND_LATENCY = Histogram(
    "kb_recommend_latency_ms",
    "KB recommend total latency in milliseconds",
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

KB_QDRANT_LATENCY = Histogram(
    "kb_qdrant_latency_ms",
    "KB Qdrant query latency in milliseconds",
    buckets=[10, 25, 50, 100, 250, 500, 1000],
)

KB_EMBED_LATENCY = Histogram(
    "kb_embed_latency_ms",
    "KB embedding latency in milliseconds",
    buckets=[10, 25, 50, 100, 250, 500],
)

KB_EMBED_ERRORS = Counter(
    "kb_embed_errors_total",
    "KB embedding errors",
)

KB_QDRANT_ERRORS = Counter(
    "kb_qdrant_errors_total",
    "KB Qdrant query errors",
)


def record_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record request metrics."""
    REQUEST_COUNT.labels(
        method=method, endpoint=endpoint, status_code=status_code
    ).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_ingestion(source_type: str, status: str, chunks_count: int = 0):
    """Record document ingestion metrics."""
    DOCUMENTS_INGESTED.labels(source_type=source_type, status=status).inc()
    if chunks_count > 0:
        CHUNKS_CREATED.labels(source_type=source_type).inc(chunks_count)


def record_embeddings(embed_model: str, count: int = 1):
    """Record embedding generation metrics."""
    EMBEDDINGS_GENERATED.labels(embed_model=embed_model).inc(count)


def record_query(mode: str, status: str, results_count: int = 0):
    """Record query metrics."""
    QUERIES_TOTAL.labels(mode=mode, status=status).inc()
    QUERY_RESULTS.observe(results_count)


def set_service_health(component: str, is_up: bool):
    """Set service component health status."""
    SERVICE_UP.labels(component=component).set(1 if is_up else 0)


def set_db_pool_metrics(pool_size: int, available: int):
    """Set database pool metrics."""
    DB_POOL_SIZE.set(pool_size)
    DB_POOL_AVAILABLE.set(available)


def set_qdrant_vectors(collection: str, count: int):
    """Set Qdrant vector count metric."""
    QDRANT_VECTORS_COUNT.labels(collection=collection).set(count)


# =============================================================================
# KB Metric Recording Functions
# =============================================================================


def _confidence_bucket(confidence: float | None) -> str:
    """Convert confidence score to bucket label."""
    if confidence is None:
        return "none"
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def record_kb_recommend(
    status: str,
    confidence: float | None,
    total_ms: float,
    qdrant_ms: float = 0.0,
    embed_ms: float = 0.0,
    used_relaxed: bool = False,
    used_metadata_fallback: bool = False,
    params_repaired: bool = False,
    incomplete_regime: bool = False,
):
    """Record KB recommend request metrics."""
    # Request counter with status and confidence bucket
    bucket = _confidence_bucket(confidence)
    KB_RECOMMEND_REQUESTS.labels(status=status, confidence_bucket=bucket).inc()

    # Latency histograms
    KB_RECOMMEND_LATENCY.observe(total_ms)
    if qdrant_ms > 0:
        KB_QDRANT_LATENCY.observe(qdrant_ms)
    if embed_ms > 0:
        KB_EMBED_LATENCY.observe(embed_ms)

    # Fallback counters
    if used_relaxed:
        KB_RECOMMEND_FALLBACK.labels(type="relaxed").inc()
    if used_metadata_fallback:
        KB_RECOMMEND_FALLBACK.labels(type="metadata_only").inc()
    if params_repaired:
        KB_RECOMMEND_FALLBACK.labels(type="repaired").inc()
    if incomplete_regime:
        KB_RECOMMEND_FALLBACK.labels(type="incomplete_regime").inc()


def record_kb_embed_error():
    """Record KB embedding error."""
    KB_EMBED_ERRORS.inc()


def record_kb_qdrant_error():
    """Record KB Qdrant query error."""
    KB_QDRANT_ERRORS.inc()


@router.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    This endpoint is excluded from OpenAPI docs.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
