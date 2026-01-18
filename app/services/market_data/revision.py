"""Data revision computation for drift detection."""

import hashlib
from typing import Sequence

from app.repositories.ohlcv import Candle


def compute_checksum(candles: Sequence[Candle]) -> str:
    """Compute a deterministic checksum from candle data.

    Algorithm:
    - Sample first 10 + last 10 + every 1000th candle
    - Format: ISO timestamp + OHLCV with 10 decimal places
    - SHA256 truncated to 16 chars
    """
    if not candles:
        return "empty"

    # Sampling logic
    n = len(candles)
    sample_indices = set()

    # First 10
    for i in range(min(10, n)):
        sample_indices.add(i)

    # Last 10
    for i in range(max(0, n - 10), n):
        sample_indices.add(i)

    # Every 1000th
    for i in range(0, n, 1000):
        sample_indices.add(i)

    # Sort for determinism
    sample_indices_list = sorted(sample_indices)

    # Build content string
    content_parts = []
    for idx in sample_indices_list:
        c = candles[idx]
        part = (
            f"{c.ts.isoformat()}|"
            f"{c.open:.10f}|{c.high:.10f}|{c.low:.10f}|{c.close:.10f}|{c.volume:.10f}"
        )
        content_parts.append(part)

    content = "\n".join(content_parts)

    # Hash and truncate
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return h[:16]
