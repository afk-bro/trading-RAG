"""
TrialDoc conversion utilities for the Trading Knowledge Base.

Provides functions for converting TrialDoc to:
- Text representation (for embedding)
- Metadata dict (for Qdrant payload)
- Building TrialDoc from tune_run records
"""

from typing import Any, Optional
from uuid import UUID

from app.services.kb.types import (
    TrialDoc,
    RegimeSnapshot,
    compute_trial_warnings,
    compute_overfit_gap,
    utc_now_iso,
    clean_nan_for_json,
)
from app.services.kb.constants import (
    KB_TRIALS_DOC_TYPE,
    TRIAL_DOC_SCHEMA_VERSION,
)


# =============================================================================
# Text Templates
# =============================================================================


def regime_to_text(r: RegimeSnapshot) -> str:
    """
    Convert RegimeSnapshot to text for embedding (query context).

    Used when querying the KB with current market conditions.

    Args:
        r: RegimeSnapshot

    Returns:
        Text representation for embedding
    """
    if r is None:
        return "Regime: unknown."

    # Tags
    regime_str = ", ".join(r.regime_tags) if r.regime_tags else "neutral"

    # Key metrics
    metrics = []
    if r.atr_pct > 0:
        metrics.append(f"ATR {r.atr_pct*100:.1f}%")
    if r.trend_strength > 0:
        direction = "up" if r.trend_dir > 0 else "down" if r.trend_dir < 0 else "flat"
        metrics.append(f"trend {direction} ({r.trend_strength:.2f})")
    if r.rsi != 50:
        metrics.append(f"RSI {r.rsi:.0f}")
    if r.efficiency_ratio != 0.5:
        metrics.append(f"efficiency {r.efficiency_ratio:.2f}")

    metrics_str = ", ".join(metrics) if metrics else "neutral conditions"

    # Timeframe and instrument
    context = []
    if r.instrument:
        context.append(r.instrument)
    if r.timeframe:
        context.append(r.timeframe)
    context_str = " ".join(context) if context else "unknown"

    return f"""Regime: {regime_str}.
Market conditions for {context_str}: {metrics_str}."""


def trial_to_text(t: TrialDoc) -> str:
    """
    Convert TrialDoc to text for embedding.

    Deterministic format that captures:
    - Dataset provenance
    - Regime characteristics
    - Strategy and parameters
    - Performance metrics
    - Quality notes

    Args:
        t: TrialDoc

    Returns:
        Text representation for embedding
    """
    # Provenance
    dataset_label = t.dataset_id or t.instrument or "unknown"
    provenance = f"Dataset: {dataset_label} {t.timeframe or ''}."
    if t.has_oos:
        provenance += " OOS enabled."

    # Regime
    regime_str = ", ".join(t.regime_tags) if t.regime_tags else "neutral"

    # Performance (prefer OOS)
    if t.has_oos and t.sharpe_oos is not None:
        perf = f"OOS Sharpe {t.sharpe_oos:.2f}"
        if t.return_frac_oos is not None:
            perf += f", return {t.return_frac_oos*100:.1f}%"
        if t.max_dd_frac_oos is not None:
            perf += f", max DD {t.max_dd_frac_oos*100:.1f}%"
    elif t.sharpe_is is not None:
        perf = f"IS Sharpe {t.sharpe_is:.2f}"
        if t.return_frac_is is not None:
            perf += f", return {t.return_frac_is*100:.1f}%"
        if t.max_dd_frac_is is not None:
            perf += f", max DD {t.max_dd_frac_is*100:.1f}%"
    else:
        perf = "metrics unavailable"

    # Objective
    obj_str = f"Objective: {t.objective_type}"
    if t.objective_score is not None:
        obj_str += f" (score {t.objective_score:.2f})"

    # Params - sorted for determinism
    params_str = ", ".join(f"{k}={v}" for k, v in sorted(t.params.items()))

    # Quality
    quality_notes = [w.replace("_", " ") for w in t.warnings]
    quality_str = f" ({', '.join(quality_notes)})" if quality_notes else ""

    return f"""{provenance}
Regime: {regime_str}.
Strategy: {t.strategy_name} with {params_str}.
Performance: {perf}. {obj_str}.{quality_str}"""


# =============================================================================
# Metadata Conversion
# =============================================================================


def trial_to_metadata(
    t: TrialDoc,
    embedding_model_id: str | None = None,
    vector_dim: int | None = None,
) -> dict[str, Any]:
    """
    Convert TrialDoc to Qdrant payload metadata.

    Includes all fields needed for filtering and retrieval.

    Args:
        t: TrialDoc
        embedding_model_id: Optional embedding model identifier (for forward compatibility)
        vector_dim: Optional vector dimension (for forward compatibility)

    Returns:
        Dict suitable for Qdrant payload (NaN → null)
    """
    payload = {
        # Schema
        "schema_version": t.schema_version,
        # Embedding info (for forward compatibility)
        "embedding_model_id": embedding_model_id,
        "vector_dim": vector_dim,
        # Identity
        "doc_type": t.doc_type,
        "tune_run_id": str(t.tune_run_id),
        "tune_id": str(t.tune_id),
        "workspace_id": str(t.workspace_id),
        "dataset_id": t.dataset_id,
        "instrument": t.instrument,
        "timeframe": t.timeframe,
        "exchange": t.exchange,
        # Strategy
        "strategy_name": t.strategy_name,
        "params": t.params,
        # Performance (numeric filters)
        "sharpe_is": t.sharpe_is,
        "sharpe_oos": t.sharpe_oos,
        "return_frac_is": t.return_frac_is,
        "return_frac_oos": t.return_frac_oos,
        "max_dd_frac_is": t.max_dd_frac_is,
        "max_dd_frac_oos": t.max_dd_frac_oos,
        "n_trades_is": t.n_trades_is,
        "n_trades_oos": t.n_trades_oos,
        "overfit_gap": t.overfit_gap,
        # Regime - tags for filtering
        "regime_tags": t.regime_tags,
        "regime_tags_is": t.regime_tags_is,
        "regime_tags_oos": t.regime_tags_oos,
        "regime_tags_str": " ".join(t.regime_tags),  # For text search
        # Regime - full snapshots for analysis (NaN cleaned)
        "regime_snapshot_is": t.regime_is.to_dict() if t.regime_is else None,
        "regime_snapshot_oos": t.regime_oos.to_dict() if t.regime_oos else None,
        # Quality
        "has_oos": t.has_oos,
        "is_valid": t.is_valid,
        "warnings": t.warnings,
        # Scoring
        "objective_type": t.objective_type,
        "objective_score": t.objective_score,
        "created_at": t.created_at,
    }
    # Clean any remaining NaN values at the top level
    return clean_nan_for_json(payload)


# =============================================================================
# Factory Functions
# =============================================================================


def build_trial_doc_from_tune_run(
    tune_run: dict,
    tune: dict,
    workspace_id: UUID | str,
) -> Optional[TrialDoc]:
    """
    Build a TrialDoc from a tune_run database record.

    Args:
        tune_run: Dict from backtest_tune_runs table
        tune: Dict from backtest_tunes table (parent)
        workspace_id: Workspace ID

    Returns:
        TrialDoc or None if data is insufficient
    """
    # Skip if not completed
    if tune_run.get("status") != "completed":
        return None

    # Extract metrics
    metrics_is = tune_run.get("metrics_is") or {}
    metrics_oos = tune_run.get("metrics_oos")
    has_oos = metrics_oos is not None and bool(metrics_oos)

    # Extract regime snapshots from metrics
    regime_is_data = metrics_is.get("regime") if metrics_is else None
    regime_oos_data = metrics_oos.get("regime") if metrics_oos else None

    regime_is = RegimeSnapshot.from_dict(regime_is_data) if regime_is_data else None
    regime_oos = RegimeSnapshot.from_dict(regime_oos_data) if regime_oos_data else None

    # Performance metrics
    sharpe_is = metrics_is.get("sharpe")
    sharpe_oos = metrics_oos.get("sharpe") if metrics_oos else None

    return_frac_is = metrics_is.get("return_pct")
    return_frac_oos = metrics_oos.get("return_pct") if metrics_oos else None

    max_dd_frac_is = metrics_is.get("max_drawdown_pct")
    max_dd_frac_oos = metrics_oos.get("max_drawdown_pct") if metrics_oos else None

    n_trades_is = metrics_is.get("trades")
    n_trades_oos = metrics_oos.get("trades") if metrics_oos else None

    # Compute overfit gap
    overfit_gap = compute_overfit_gap(sharpe_is, sharpe_oos)

    # Quality flags
    is_valid = (
        tune_run.get("skip_reason") is None and tune_run.get("failed_reason") is None
    )

    # Compute warnings
    warnings = compute_trial_warnings(
        sharpe_is=sharpe_is,
        sharpe_oos=sharpe_oos,
        overfit_gap=overfit_gap,
        n_trades_oos=n_trades_oos,
        max_dd_frac_oos=max_dd_frac_oos,
        is_valid=is_valid,
        regime_is=regime_is,
        regime_oos=regime_oos,
        has_oos=has_oos,
    )

    # Extract tune metadata
    strategy_name = tune.get("strategy_name", "")
    objective_type = tune.get("objective_type", "sharpe")
    dataset_id = tune.get("dataset_id")
    instrument = tune.get("instrument")
    timeframe = tune.get("timeframe")

    return TrialDoc(
        schema_version=TRIAL_DOC_SCHEMA_VERSION,
        doc_type=KB_TRIALS_DOC_TYPE,
        tune_run_id=tune_run.get("run_id"),
        tune_id=tune_run.get("tune_id"),
        workspace_id=workspace_id,
        dataset_id=dataset_id,
        instrument=instrument,
        timeframe=timeframe,
        strategy_name=strategy_name,
        params=tune_run.get("params", {}),
        sharpe_is=sharpe_is,
        sharpe_oos=sharpe_oos,
        return_frac_is=return_frac_is,
        return_frac_oos=return_frac_oos,
        max_dd_frac_is=max_dd_frac_is,
        max_dd_frac_oos=max_dd_frac_oos,
        n_trades_is=n_trades_is,
        n_trades_oos=n_trades_oos,
        overfit_gap=overfit_gap,
        regime_is=regime_is,
        regime_oos=regime_oos,
        has_oos=has_oos,
        is_valid=is_valid,
        warnings=warnings,
        objective_type=objective_type,
        objective_score=tune_run.get("objective_score"),
        created_at=_format_timestamp(tune_run.get("created_at")),
    )


def _format_timestamp(ts: Any) -> str:
    """Convert timestamp to ISO string."""
    if ts is None:
        return utc_now_iso()
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def build_trial_doc_from_eligible_row(row: dict) -> Optional[TrialDoc]:
    """
    Build a TrialDoc from a kb_eligible_trials view row.

    This factory handles both tune_run and test_variant sources,
    normalizing the data into a consistent TrialDoc format.

    Args:
        row: Dict from kb_eligible_trials view

    Returns:
        TrialDoc or None if data is insufficient
    """
    source_type = row.get("source_type")
    if source_type not in ("tune_run", "test_variant"):
        return None

    # Extract regime snapshots
    regime_is_data = row.get("regime_is")
    regime_oos_data = row.get("regime_oos")

    regime_is = RegimeSnapshot.from_dict(regime_is_data) if regime_is_data else None
    regime_oos = RegimeSnapshot.from_dict(regime_oos_data) if regime_oos_data else None

    # Performance metrics
    sharpe_is = row.get("sharpe_is")
    sharpe_oos = row.get("sharpe_oos")
    return_frac_oos = row.get("return_frac_oos")
    max_dd_frac_oos = row.get("max_dd_frac_oos")
    n_trades_oos = row.get("n_trades_oos")

    # Has OOS if we have OOS metrics
    has_oos = sharpe_oos is not None

    # Compute overfit gap
    overfit_gap = compute_overfit_gap(sharpe_is, sharpe_oos)

    # Quality flags - assume valid if in eligible view
    is_valid = True

    # Compute warnings
    warnings = compute_trial_warnings(
        sharpe_is=sharpe_is,
        sharpe_oos=sharpe_oos,
        overfit_gap=overfit_gap,
        n_trades_oos=n_trades_oos,
        max_dd_frac_oos=max_dd_frac_oos,
        is_valid=is_valid,
        regime_is=regime_is,
        regime_oos=regime_oos,
        has_oos=has_oos,
    )

    # Identity - tune_run_id and tune_id vary by source
    source_id = row.get("source_id")
    group_id = row.get("group_id")

    if source_type == "tune_run":
        tune_run_id = source_id
        tune_id = group_id
    else:
        # test_variant: source_id is backtest_run.id, group_id is run_plan_id
        tune_run_id = source_id
        tune_id = group_id  # Use run_plan_id as group

    return TrialDoc(
        schema_version=TRIAL_DOC_SCHEMA_VERSION,
        doc_type=KB_TRIALS_DOC_TYPE,
        tune_run_id=tune_run_id,
        tune_id=tune_id,
        workspace_id=row.get("workspace_id"),
        dataset_id=None,  # Not in view, could be added
        instrument=None,  # Not in view, could be added
        timeframe=None,  # Not in view, could be added
        strategy_name=row.get("strategy_name", ""),
        params=row.get("params") or {},
        sharpe_is=sharpe_is,
        sharpe_oos=sharpe_oos,
        return_frac_is=None,  # Not tracked for tune_runs in view
        return_frac_oos=return_frac_oos,
        max_dd_frac_is=None,  # Not tracked
        max_dd_frac_oos=max_dd_frac_oos,
        n_trades_is=None,  # Not tracked
        n_trades_oos=n_trades_oos,
        overfit_gap=overfit_gap,
        regime_is=regime_is,
        regime_oos=regime_oos,
        has_oos=has_oos,
        is_valid=is_valid,
        warnings=warnings,
        objective_type=row.get("objective_type", "sharpe"),
        objective_score=row.get("objective_score"),
        created_at=_format_timestamp(row.get("created_at")),
    )


# =============================================================================
# Trial Summary (for responses)
# =============================================================================


def trial_to_summary(t: TrialDoc) -> dict:
    """
    Convert TrialDoc to summary dict for API responses.

    Lighter-weight than full metadata, focused on display.

    Args:
        t: TrialDoc

    Returns:
        Summary dict
    """
    return {
        "tune_run_id": str(t.tune_run_id),
        "tune_id": str(t.tune_id),
        "strategy_name": t.strategy_name,
        "params": t.params,
        "sharpe_oos": t.sharpe_oos,
        "sharpe_is": t.sharpe_is,
        "return_frac_oos": t.return_frac_oos,
        "max_dd_frac_oos": t.max_dd_frac_oos,
        "n_trades_oos": t.n_trades_oos,
        "overfit_gap": t.overfit_gap,
        "regime_tags": t.regime_tags,
        "objective_type": t.objective_type,
        "objective_score": t.objective_score,
        "has_oos": t.has_oos,
        "is_valid": t.is_valid,
        "warnings": t.warnings,
    }
