"""
Visual smoke tests - verify key pages load with correct CSS.

These tests check that:
1. Pages return 200 OK
2. Design system CSS files are linked
3. Key DOM elements exist
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestDesignSystemLoaded:
    """Verify design system CSS is properly linked."""

    def test_landing_page_loads(self, client):
        """Landing page loads with consistent dark theme."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Landing uses inline styles with same CSS vars
        assert "--bg:" in html
        assert "--text:" in html
        assert "--accent:" in html

    def test_landing_page_has_key_sections(self, client):
        """Landing page has hero, features, how-it-works."""
        response = client.get("/")
        html = response.text

        assert 'class="hero"' in html
        assert 'class="features"' in html
        assert 'class="how-it-works"' in html


class TestStaticAssets:
    """Verify static CSS files are served correctly."""

    def test_tokens_css_served(self, client):
        """tokens.css is served with correct content type."""
        response = client.get("/static/admin/css/tokens.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        # Check key token exists
        assert "--bg:" in response.text
        assert "--text:" in response.text

    def test_components_css_served(self, client):
        """components.css is served with correct content type."""
        response = client.get("/static/admin/css/components.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        # Check key component exists
        assert ".btn" in response.text
        assert ".card" in response.text

    def test_sparklines_js_served(self, client):
        """sparklines.js is served."""
        response = client.get("/static/admin/js/sparklines.js")
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")

    def test_compare_tray_js_served(self, client):
        """compare_tray.js is served."""
        response = client.get("/static/admin/js/compare_tray.js")
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")


class TestKeyPagesStructure:
    """Verify key page DOM structure (without requiring auth)."""

    def test_landing_has_nav_structure(self, client):
        """Landing page has expected structure."""
        response = client.get("/")
        html = response.text

        # Key structural elements
        assert "<title>" in html
        assert "Trading RAG" in html

    def test_docs_page_loads(self, client):
        """API docs page loads."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestHTMXSupport:
    """Verify HTMX is properly integrated."""

    def test_layout_template_includes_htmx(self):
        """Layout template includes HTMX script tag."""
        import os

        layout_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "app",
            "admin",
            "templates",
            "layout.html",
        )
        with open(layout_path) as f:
            html = f.read()
        assert "htmx.org" in html
