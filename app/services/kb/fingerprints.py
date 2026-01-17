"""
Regime fingerprint service for instant similarity queries.

Materializes regime vectors as 32-byte hashes for O(1) exact matching
and stores raw vectors for similarity computation.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import structlog
from asyncpg import Connection

from app.services.kb.constants import REGIME_SCHEMA_VERSION

logger = structlog.get_logger(__name__)


@dataclass
class RegimeVector:
    """Numeric regime features for similarity computation."""

    atr_norm: float  # ATR as fraction of close (0-0.05 typical)
    rsi: float  # RSI value (0-100)
    bb_width: float  # Bollinger Band width as fraction (0-0.10 typical)
    efficiency: float  # Kaufman Efficiency Ratio (0-1)
    trend_strength: float  # Trend strength (0-1)
    zscore: float  # Price z-score (-3 to +3 typical)

    def to_array(self) -> list[float]:
        """Convert to array for storage."""
        return [
            self.atr_norm,
            self.rsi,
            self.bb_width,
            self.efficiency,
            self.trend_strength,
            self.zscore,
        ]

    @classmethod
    def from_array(cls, arr: list[float]) -> "RegimeVector":
        """Reconstruct from array."""
        if len(arr) != 6:
            raise ValueError(f"Expected 6 elements, got {len(arr)}")
        return cls(
            atr_norm=arr[0],
            rsi=arr[1],
            bb_width=arr[2],
            efficiency=arr[3],
            trend_strength=arr[4],
            zscore=arr[5],
        )

    def compute_fingerprint(self) -> bytes:
        """
        Compute 32-byte SHA256 fingerprint from regime vector.

        Uses canonical representation with 4 decimal precision
        for stability across floating-point variations.
        """
        # Round to 4 decimals and join with pipes
        canonical = "|".join(f"{v:.4f}" for v in self.to_array())
        return hashlib.sha256(canonical.encode("utf-8")).digest()


@dataclass
class RegimeFingerprint:
    """Materialized regime fingerprint record."""

    id: UUID
    tune_id: UUID
    run_id: UUID
    fingerprint: bytes
    regime_vector: list[float]
    trend_tag: Optional[str]
    vol_tag: Optional[str]
    efficiency_tag: Optional[str]
    regime_schema_version: str


class RegimeFingerprintRepository:
    """Repository for regime fingerprint operations."""

    def __init__(self, conn: Connection):
        """Initialize with database connection."""
        self.conn = conn

    async def upsert(
        self,
        tune_id: UUID,
        run_id: UUID,
        vector: RegimeVector,
        trend_tag: Optional[str] = None,
        vol_tag: Optional[str] = None,
        efficiency_tag: Optional[str] = None,
    ) -> UUID:
        """
        Insert or update regime fingerprint for a run.

        Args:
            tune_id: Parent tune ID
            run_id: Trial run ID
            vector: Regime vector features
            trend_tag: Optional trend classification tag
            vol_tag: Optional volatility classification tag
            efficiency_tag: Optional efficiency classification tag

        Returns:
            ID of the created/updated fingerprint record.
        """
        fingerprint = vector.compute_fingerprint()
        regime_array = vector.to_array()

        result = await self.conn.fetchrow(
            """
            INSERT INTO regime_fingerprints (
                tune_id, run_id, fingerprint, regime_vector,
                trend_tag, vol_tag, efficiency_tag, regime_schema_version
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (run_id) DO UPDATE SET
                fingerprint = EXCLUDED.fingerprint,
                regime_vector = EXCLUDED.regime_vector,
                trend_tag = EXCLUDED.trend_tag,
                vol_tag = EXCLUDED.vol_tag,
                efficiency_tag = EXCLUDED.efficiency_tag,
                regime_schema_version = EXCLUDED.regime_schema_version,
                updated_at = NOW()
            RETURNING id
            """,
            tune_id,
            run_id,
            fingerprint,
            regime_array,
            trend_tag,
            vol_tag,
            efficiency_tag,
            REGIME_SCHEMA_VERSION,
        )

        fp_id = result["id"]
        logger.debug(
            "regime_fingerprint_upserted",
            fingerprint_id=str(fp_id),
            run_id=str(run_id),
            fingerprint_hex=fingerprint.hex()[:16],
        )
        return fp_id

    async def find_by_fingerprint(
        self,
        fingerprint: bytes,
        limit: int = 100,
    ) -> list[RegimeFingerprint]:
        """
        Find all runs with exact fingerprint match.

        Uses hash index for O(1) lookup.

        Args:
            fingerprint: 32-byte SHA256 hash
            limit: Maximum results to return

        Returns:
            List of matching fingerprint records.
        """
        rows = await self.conn.fetch(
            """
            SELECT id, tune_id, run_id, fingerprint, regime_vector,
                   trend_tag, vol_tag, efficiency_tag, regime_schema_version
            FROM regime_fingerprints
            WHERE fingerprint = $1
            LIMIT $2
            """,
            fingerprint,
            limit,
        )

        return [
            RegimeFingerprint(
                id=row["id"],
                tune_id=row["tune_id"],
                run_id=row["run_id"],
                fingerprint=bytes(row["fingerprint"]),
                regime_vector=list(row["regime_vector"]),
                trend_tag=row["trend_tag"],
                vol_tag=row["vol_tag"],
                efficiency_tag=row["efficiency_tag"],
                regime_schema_version=row["regime_schema_version"],
            )
            for row in rows
        ]

    async def find_similar(
        self,
        vector: RegimeVector,
        max_distance: float = 0.5,
        trend_tag: Optional[str] = None,
        vol_tag: Optional[str] = None,
        limit: int = 50,
    ) -> list[tuple[RegimeFingerprint, float]]:
        """
        Find runs with similar regime vectors.

        Pre-filters by tags if provided, then computes distance.

        Args:
            vector: Reference regime vector
            max_distance: Maximum Euclidean distance threshold
            trend_tag: Optional filter by trend tag
            vol_tag: Optional filter by volatility tag
            limit: Maximum results to return

        Returns:
            List of (fingerprint, distance) tuples, sorted by distance.
        """
        regime_array = vector.to_array()

        # Build query with optional tag filters
        conditions = ["TRUE"]
        params: list = [regime_array, max_distance, limit]
        param_idx = 4

        if trend_tag:
            conditions.append(f"trend_tag = ${param_idx}")
            params.append(trend_tag)
            param_idx += 1

        if vol_tag:
            conditions.append(f"vol_tag = ${param_idx}")
            params.append(vol_tag)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        rows = await self.conn.fetch(
            f"""
            SELECT id, tune_id, run_id, fingerprint, regime_vector,
                   trend_tag, vol_tag, efficiency_tag, regime_schema_version,
                   regime_distance(regime_vector, $1) AS distance
            FROM regime_fingerprints
            WHERE {where_clause}
              AND regime_distance(regime_vector, $1) <= $2
            ORDER BY distance ASC
            LIMIT $3
            """,
            *params,
        )

        return [
            (
                RegimeFingerprint(
                    id=row["id"],
                    tune_id=row["tune_id"],
                    run_id=row["run_id"],
                    fingerprint=bytes(row["fingerprint"]),
                    regime_vector=list(row["regime_vector"]),
                    trend_tag=row["trend_tag"],
                    vol_tag=row["vol_tag"],
                    efficiency_tag=row["efficiency_tag"],
                    regime_schema_version=row["regime_schema_version"],
                ),
                row["distance"],
            )
            for row in rows
        ]

    async def get_by_run_id(self, run_id: UUID) -> Optional[RegimeFingerprint]:
        """Get fingerprint for a specific run."""
        row = await self.conn.fetchrow(
            """
            SELECT id, tune_id, run_id, fingerprint, regime_vector,
                   trend_tag, vol_tag, efficiency_tag, regime_schema_version
            FROM regime_fingerprints
            WHERE run_id = $1
            """,
            run_id,
        )

        if not row:
            return None

        return RegimeFingerprint(
            id=row["id"],
            tune_id=row["tune_id"],
            run_id=row["run_id"],
            fingerprint=bytes(row["fingerprint"]),
            regime_vector=list(row["regime_vector"]),
            trend_tag=row["trend_tag"],
            vol_tag=row["vol_tag"],
            efficiency_tag=row["efficiency_tag"],
            regime_schema_version=row["regime_schema_version"],
        )

    async def count_by_fingerprint(self, fingerprint: bytes) -> int:
        """Count runs with exact fingerprint match."""
        result = await self.conn.fetchval(
            "SELECT COUNT(*) FROM regime_fingerprints WHERE fingerprint = $1",
            fingerprint,
        )
        return result or 0

    async def delete_by_run_id(self, run_id: UUID) -> bool:
        """Delete fingerprint for a run."""
        result = await self.conn.execute(
            "DELETE FROM regime_fingerprints WHERE run_id = $1",
            run_id,
        )
        return result == "DELETE 1"
