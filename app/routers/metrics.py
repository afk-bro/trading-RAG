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


def record_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
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
