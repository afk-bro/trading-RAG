"""Integration tests for the Trading RAG API endpoints.

These tests verify the main API flows work end-to-end.
Some tests require external services (Supabase, Qdrant, Ollama) to be running.

Run with: pytest tests/integration/ -v
Skip tests requiring DB: pytest tests/integration/ -v -m "not requires_db"
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import uuid


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        # Mock the database and Qdrant connections for health endpoint tests
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 even with degraded status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        """Test that health response has all required fields."""
        response = client.get("/health")
        data = response.json()

        required_fields = [
            "status", "qdrant", "supabase", "ollama",
            "active_collection", "embed_model", "latency_ms", "version"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_health_includes_request_id_header(self, client):
        """Test that response includes X-Request-ID header."""
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_health_includes_response_time_header(self, client):
        """Test that response includes X-Response-Time-Ms header."""
        response = client.get("/health")
        assert "x-response-time-ms" in response.headers


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_service_info(self, client):
        """Test that root endpoint has service information."""
        response = client.get("/")
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert data["service"] == "Trading RAG Pipeline"


class TestIngestEndpoint:
    """Tests for the /ingest endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_ingest_validates_required_fields(self, client):
        """Test that ingest validates required fields."""
        # Missing workspace_id
        response = client.post("/ingest", json={
            "source": {"type": "article", "url": "https://example.com"},
            "content": "Test content"
        })
        assert response.status_code == 422

    def test_ingest_validates_workspace_id_format(self, client):
        """Test that workspace_id must be a valid UUID."""
        response = client.post("/ingest", json={
            "workspace_id": "not-a-uuid",
            "source": {"type": "article", "url": "https://example.com"},
            "content": "Test content"
        })
        assert response.status_code == 422

    def test_ingest_validates_source_type(self, client):
        """Test that source type must be valid."""
        response = client.post("/ingest", json={
            "workspace_id": str(uuid.uuid4()),
            "source": {"type": "invalid_type", "url": "https://example.com"},
            "content": "Test content"
        })
        assert response.status_code == 422

    def test_ingest_returns_503_without_db(self, client):
        """Test that ingest returns 503 when DB is not available."""
        response = client.post("/ingest", json={
            "workspace_id": str(uuid.uuid4()),
            "source": {"type": "article", "url": "https://example.com"},
            "content": "Test content for ingestion",
            "metadata": {"title": "Test Article"}
        })
        assert response.status_code == 503
        assert "Database" in response.json().get("detail", "")


class TestYouTubeEndpoint:
    """Tests for the /sources/youtube/ingest endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_youtube_validates_required_fields(self, client):
        """Test that YouTube ingest validates required fields."""
        # Missing url
        response = client.post("/sources/youtube/ingest", json={
            "workspace_id": str(uuid.uuid4())
        })
        assert response.status_code == 422

    def test_youtube_invalid_url_returns_error(self, client):
        """Test that invalid URL returns error response."""
        response = client.post("/sources/youtube/ingest", json={
            "workspace_id": str(uuid.uuid4()),
            "url": "not-a-youtube-url"
        })
        assert response.status_code == 200  # Returns 200 with error status
        data = response.json()
        assert data.get("status") == "error"
        assert data.get("retryable") == False
        assert "Could not extract video ID" in data.get("error_reason", "")

    def test_youtube_error_includes_retryable_flag(self, client):
        """Test that YouTube error response includes retryable flag."""
        response = client.post("/sources/youtube/ingest", json={
            "workspace_id": str(uuid.uuid4()),
            "url": "invalid"
        })
        data = response.json()
        assert "retryable" in data


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_query_validates_required_fields(self, client):
        """Test that query validates required fields."""
        # Missing workspace_id
        response = client.post("/query", json={
            "question": "What is the market outlook?"
        })
        assert response.status_code == 422

    def test_query_validates_mode(self, client):
        """Test that query mode must be valid."""
        response = client.post("/query", json={
            "workspace_id": str(uuid.uuid4()),
            "question": "What is the market outlook?",
            "mode": "invalid_mode"
        })
        assert response.status_code == 422

    def test_query_returns_503_without_db(self, client):
        """Test that query returns 503 when DB is not available."""
        response = client.post("/query", json={
            "workspace_id": str(uuid.uuid4()),
            "question": "What is the market outlook?",
            "mode": "retrieve"
        })
        assert response.status_code == 503


class TestJobsEndpoint:
    """Tests for the /jobs endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_jobs_unknown_id_returns_404(self, client):
        """Test that unknown job ID returns 404."""
        response = client.get(f"/jobs/{uuid.uuid4()}")
        assert response.status_code == 404


class TestReembedEndpoint:
    """Tests for the /reembed endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_reembed_validates_required_fields(self, client):
        """Test that reembed validates required fields."""
        # Missing workspace_id
        response = client.post("/reembed", json={
            "target_collection": "test_collection"
        })
        assert response.status_code == 422

    def test_reembed_returns_503_without_db(self, client):
        """Test that reembed returns 503 when DB is not available."""
        response = client.post("/reembed", json={
            "workspace_id": str(uuid.uuid4()),
            "target_collection": "test_collection",
            "embed_provider": "ollama",
            "embed_model": "nomic-embed-text"
        })
        assert response.status_code == 503


class TestCORSHeaders:
    """Tests for CORS headers."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_cors_headers_on_options(self, client):
        """Test that OPTIONS request returns CORS headers."""
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_all_origins(self, client):
        """Test that CORS allows all origins (development mode)."""
        response = client.get("/health", headers={
            "Origin": "http://example.com"
        })
        assert response.headers.get("access-control-allow-origin") == "*"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch("app.main._db_pool", None), \
             patch("app.main._qdrant_client", None):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

    def test_malformed_json_returns_422(self, client):
        """Test that malformed JSON returns 422."""
        response = client.post(
            "/ingest",
            content="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_error_response_has_detail(self, client):
        """Test that error responses have detail field."""
        response = client.post("/ingest", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_404_for_unknown_endpoint(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404


# Tests that require full infrastructure (marked to skip without proper config)
@pytest.mark.requires_db
class TestIngestFlow:
    """Integration tests for the full ingest flow.

    These tests require Supabase, Qdrant, and Ollama to be running.
    """

    @pytest.fixture
    def client(self):
        """Create test client with real connections."""
        # These tests require actual services running
        from app.main import app
        with TestClient(app) as client:
            yield client

    @pytest.mark.skip(reason="Requires Supabase connection")
    def test_ingest_creates_document(self, client):
        """Test that ingest creates a document with chunks and vectors."""
        response = client.post("/ingest", json={
            "workspace_id": str(uuid.uuid4()),
            "source": {"type": "article", "url": "https://example.com/test"},
            "content": "This is a test article about AAPL stock. The Fed announced new policy.",
            "metadata": {"title": "Test Article", "author": "Test Author"}
        })

        assert response.status_code in [200, 201]
        data = response.json()
        assert "doc_id" in data
        assert data["chunks_created"] > 0
        assert data["vectors_created"] > 0


@pytest.mark.requires_db
class TestQueryFlow:
    """Integration tests for the query flow.

    These tests require Supabase, Qdrant, and Ollama to be running.
    """

    @pytest.mark.skip(reason="Requires Supabase connection")
    def test_query_returns_results(self, client):
        """Test that query returns relevant results."""
        # First ingest some content
        # Then query for it
        pass


@pytest.mark.requires_db
class TestYouTubeFlow:
    """Integration tests for the YouTube ingestion flow.

    These tests require Supabase, Qdrant, Ollama, and YouTube API access.
    """

    @pytest.mark.skip(reason="Requires Supabase and YouTube API")
    def test_youtube_ingest_creates_document(self, client):
        """Test that YouTube ingest creates a document with timestamps."""
        pass
