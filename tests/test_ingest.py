#!/usr/bin/env python3
"""Test script to verify the ingest endpoint and chunk_vectors table."""

import httpx
import json
import time

BASE_URL = "http://localhost:8000"


def test_ingest():
    """Test the /ingest endpoint."""
    payload = {
        "workspace_id": "00000000-0000-0000-0000-000000000001",
        "idempotency_key": f"test-chunk-vectors-{int(time.time())}",
        "source": {
            "url": f"https://example.com/test-article-{int(time.time())}",
            "type": "article",
        },
        "content": """The Federal Reserve has announced a significant change in monetary policy.
        Chairman Powell discussed the implications for AAPL, GOOGL, and MSFT stock prices.
        The interest rate decision will impact the macro economic outlook.
        Inflation concerns remain elevated while employment data shows resilience.
        The FOMC meeting revealed important insights about future rate paths.
        This is additional content to ensure we have enough text to create meaningful chunks.
        The market reacted positively to the news, with tech stocks leading the gains.
        Analysts expect continued volatility as investors digest the policy implications.""",
        "metadata": {"title": "Fed Policy Impact Analysis", "author": "Test Author"},
    }

    print("Testing POST /ingest...")
    response = httpx.post(f"{BASE_URL}/ingest", json=payload, timeout=60.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    test_ingest()
