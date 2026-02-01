"""Instrument metadata: point values, tick sizes, contract specs."""

# Point values per contract
POINT_VALUE_NQ = 20.0  # Full NQ = $20 per point
POINT_VALUE_ES = 50.0  # Full ES = $50 per point


def get_point_value(symbol: str) -> float:
    """Get point value for the instrument.

    Micro contracts are 1/10th of the full contract.
    Order matters: check MNQ before NQ, MES before ES.

    Args:
        symbol: Trading symbol (e.g., "MNQ", "NQ", "MES1!", "ES")

    Returns:
        Dollar value per point for the instrument.
    """
    symbol_upper = symbol.upper()
    # Check micro contracts first (MNQ before NQ, MES before ES)
    if "MNQ" in symbol_upper:
        return POINT_VALUE_NQ / 10  # Micro NQ = $2 per point
    elif "NQ" in symbol_upper:
        return POINT_VALUE_NQ  # Full NQ = $20 per point
    elif "MES" in symbol_upper:
        return POINT_VALUE_ES / 10  # Micro ES = $5 per point
    elif "ES" in symbol_upper:
        return POINT_VALUE_ES  # Full ES = $50 per point
    else:
        return POINT_VALUE_NQ  # Default to NQ
