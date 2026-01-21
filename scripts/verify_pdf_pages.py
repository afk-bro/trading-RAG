#!/usr/bin/env python3
"""Script to verify PDF page locator label tests via API calls."""

import time
import requests
import json

def main():
    timestamp = int(time.time())
    base_url = "http://localhost:8000"

    # Step 1: POST to /ingest
    payload = {
        "workspace_id": "00000000-0000-0000-0000-000000000001",
        "idempotency_key": f"pdf-page-test-{timestamp}",
        "source": {
            "url": f"https://example.com/test-pdf-pages-{timestamp}.pdf",
            "type": "pdf"
        },
        "content": "Full document content placeholder for PDF testing.",
        "metadata": {
            "title": "PDF Page Test Document",
            "author": "Test PDF Author"
        },
        "chunks": [
            {"content": "This is page 1 content about market analysis.", "page_start": 1, "page_end": 1},
            {"content": "Page 2 covers detailed analysis.", "page_start": 2, "page_end": 2},
            {"content": "This chunk spans pages 3 to 5.", "page_start": 3, "page_end": 5},
            {"content": "Pages 6 through 7 discuss compliance.", "page_start": 6, "page_end": 7}
        ]
    }

    print("=" * 60)
    print("Step 1: POST /ingest")
    print("=" * 60)

    resp = requests.post(f"{base_url}/ingest", json=payload)
    print(f"Status: {resp.status_code}")
    result = resp.json()
    print(json.dumps(result, indent=2))

    doc_id = result.get("doc_id")
    print(f"\nExtracted doc_id: {doc_id}")

    if not doc_id:
        print("ERROR: No doc_id returned!")
        return

    # Step 2: GET /debug/chunks
    print("\n" + "=" * 60)
    print("Step 2: GET /debug/chunks")
    print("=" * 60)

    chunks_resp = requests.get(f"{base_url}/debug/chunks?doc_id={doc_id}")
    print(f"Status: {chunks_resp.status_code}")
    chunks_data = chunks_resp.json()

    if not chunks_data.get("success"):
        print(f"ERROR: {chunks_data}")
        return

    chunks = chunks_data.get("chunks", [])
    print(f"Found {len(chunks)} chunks")

    # Step 3: Verification
    print("\n" + "=" * 60)
    print("Step 3: Verification Results")
    print("=" * 60)

    test1_pass = True  # Single page format: "p. X"
    test2_pass = True  # Multi-page format: "pp. X-Y"

    for chunk in chunks:
        content = chunk.get("content_preview", "")[:40]
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        locator_label = chunk.get("locator_label")

        if page_start is not None and page_end is not None:
            if page_start == page_end:
                expected = f"p. {page_start}"
                test_name = "Single page format"
                if locator_label != expected:
                    test1_pass = False
            else:
                expected = f"pp. {page_start}-{page_end}"
                test_name = "Multi-page format"
                if locator_label != expected:
                    test2_pass = False

            passed = locator_label == expected
            status = "PASS" if passed else "FAIL"

            print(f"\n{status}: {test_name}")
            print(f"  Content: \"{content}...\"")
            print(f"  page_start: {page_start}, page_end: {page_end}")
            print(f"  Expected locator_label: \"{expected}\"")
            print(f"  Actual locator_label:   \"{locator_label}\"")

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"\nTest 1: 'Locator label for PDF uses page numbers'")
    print(f"        Should use 'p. X' format for single page chunks")
    print(f"        Result: {'PASS' if test1_pass else 'FAIL'}")

    print(f"\nTest 2: 'Multi-page PDF chunk spans tracked correctly'")
    print(f"        Should use 'pp. X-Y' format for multi-page chunks")
    print(f"        Result: {'PASS' if test2_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    if test1_pass and test2_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()
