#!/usr/bin/env python3
"""Generate synthetic OHLCV data with predictable trend/chop/reversal segments.

Creates a fixture that reliably generates trades across many strategy types:
- Segment A (0-30%): mild uptrend + small noise (momentum strategies trade)
- Segment B (30-60%): sideways sine-wave chop (mean reversion strategies trade)
- Segment C (60-100%): downtrend + larger oscillations (both types trade)

Usage:
    python tests/unit/fixtures/generate_synth_ohlcv.py

Output:
    tests/unit/fixtures/ohlcv_synth_trendy_range.csv
"""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
NUM_BARS = 1000
START_DATE = datetime(2023, 1, 1, 9, 30)  # Market open
INTERVAL = timedelta(hours=1)  # 1-hour bars
BASE_PRICE = 100.0
BASE_VOLUME = 1_000_000

# Segment boundaries (as fractions)
SEG_A_END = 0.30  # Uptrend ends
SEG_B_END = 0.60  # Chop ends, downtrend begins


def generate_ohlcv(num_bars: int = NUM_BARS) -> list[dict]:
    """Generate synthetic OHLCV bars with regime changes."""
    bars = []
    price = BASE_PRICE

    for i in range(num_bars):
        t = i / num_bars  # Progress through data (0 to 1)

        # Determine segment and price dynamics
        if t < SEG_A_END:
            # Segment A: Mild uptrend with small noise
            # Price drifts up ~0.05% per bar with small oscillations
            trend = 0.0005 * price
            noise_amp = 0.002 * price
            noise = noise_amp * math.sin(i * 0.5)
            price = price + trend + noise
            volatility = 0.005  # Low volatility

        elif t < SEG_B_END:
            # Segment B: Sideways sine-wave chop (mean reversion territory)
            # Price oscillates around a stable mean
            cycle_period = 40  # bars per cycle
            amplitude = 0.03 * BASE_PRICE  # 3% swings
            center = price  # Hold roughly stable
            offset = amplitude * math.sin(2 * math.pi * i / cycle_period)
            price = center + offset * 0.1  # Damped update to avoid drift
            # Add smaller high-frequency noise
            price += 0.001 * price * math.sin(i * 2.1)
            volatility = 0.008  # Medium volatility

        else:
            # Segment C: Downtrend with larger oscillations
            # Price drifts down with bigger swings (both momentum and reversal opportunities)
            trend = -0.0004 * price
            noise_amp = 0.004 * price
            noise = noise_amp * math.sin(i * 0.3) + noise_amp * 0.5 * math.cos(i * 0.7)
            price = price + trend + noise
            volatility = 0.012  # Higher volatility

        # Ensure price stays positive
        price = max(price, 1.0)

        # Generate OHLC from close price and volatility
        close = price

        # Intrabar movement: open, high, low
        bar_range = volatility * close

        # Randomish but deterministic intrabar dynamics
        phase = math.sin(i * 1.7) * 0.5 + 0.5  # 0 to 1

        if i % 3 == 0:
            # Bullish bar
            open_price = close - bar_range * phase
            high = close + bar_range * (1 - phase) * 0.5
            low = open_price - bar_range * phase * 0.3
        elif i % 3 == 1:
            # Bearish bar
            open_price = close + bar_range * phase
            high = open_price + bar_range * (1 - phase) * 0.3
            low = close - bar_range * phase * 0.5
        else:
            # Doji-ish
            open_price = close + bar_range * 0.1 * math.sin(i)
            high = max(open_price, close) + bar_range * 0.4
            low = min(open_price, close) - bar_range * 0.4

        # Ensure OHLC validity: low <= open,close <= high
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        low = max(low, 0.01)  # Floor at 1 cent

        # Volume: higher in volatile segments, spikes on trend changes
        segment_vol_mult = 1.0 + volatility * 50
        spike = 1.5 if (abs(t - SEG_A_END) < 0.02 or abs(t - SEG_B_END) < 0.02) else 1.0
        volume = int(
            BASE_VOLUME
            * segment_vol_mult
            * spike
            * (0.8 + 0.4 * abs(math.sin(i * 0.1)))
        )

        # Timestamp
        timestamp = START_DATE + i * INTERVAL

        bars.append(
            {
                "date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": volume,
            }
        )

    return bars


def write_csv(bars: list[dict], output_path: Path) -> None:
    """Write bars to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(bars)
    print(f"Wrote {len(bars)} bars to {output_path}")


def main():
    # Generate data
    bars = generate_ohlcv()

    # Summary stats
    prices = [b["close"] for b in bars]
    print(f"Price range: {min(prices):.2f} - {max(prices):.2f}")
    print(f"Start: {prices[0]:.2f}, End: {prices[-1]:.2f}")
    print(f"Bars: {len(bars)}")

    # Segment breakdown
    seg_a = int(NUM_BARS * SEG_A_END)
    seg_b = int(NUM_BARS * SEG_B_END)
    print(
        f"Segment A (uptrend): bars 0-{seg_a}, close {prices[0]:.2f} -> {prices[seg_a-1]:.2f}"
    )
    print(
        f"Segment B (chop): bars {seg_a}-{seg_b}, close {prices[seg_a]:.2f} -> {prices[seg_b-1]:.2f}"
    )
    print(
        f"Segment C (downtrend): bars {seg_b}-{NUM_BARS}, close {prices[seg_b]:.2f} -> {prices[-1]:.2f}"
    )

    # Write output
    output_path = Path(__file__).parent / "ohlcv_synth_trendy_range.csv"
    write_csv(bars, output_path)


if __name__ == "__main__":
    main()
