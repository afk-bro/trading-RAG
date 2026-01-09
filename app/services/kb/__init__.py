"""
Trading Knowledge Base service.

Provides case-based reasoning for strategy parameter recommendations:
- RegimeSnapshot: Market regime feature extraction
- TrialDoc: KB document for backtest trials
- Regime computation: ATR, RSI, BB, efficiency ratio
- Text/metadata conversion for embedding and storage
"""

from app.services.kb.types import (
    RegimeSnapshot,
    TrialDoc,
    compute_trial_warnings,
    compute_overfit_gap,
    utc_now_iso,
)

from app.services.kb.regime import (
    compute_regime_snapshot,
    compute_tags,
    regime_snapshot_to_text,
    compute_atr,
    compute_rsi,
    compute_bollinger_bands,
    compute_bb_width_pct,
    compute_efficiency_ratio,
    compute_zscore,
    compute_trend_strength,
    compute_trend_direction,
)

from app.services.kb.trial_doc import (
    trial_to_text,
    trial_to_metadata,
    trial_to_summary,
    build_trial_doc_from_tune_run,
)

from app.services.kb.parsing import (
    ParsedDataset,
    parse_ohlcv_for_kb,
    compute_fingerprint_from_bytes,
    detect_timeframe_from_filename,
    validate_for_regime,
)

from app.services.kb.constants import (
    # Regime computation
    REGIME_WINDOW_BARS,
    MIN_EFFECTIVE_BARS,
    DEFAULT_FEATURE_PARAMS,
    # Tagging thresholds
    TREND_STRONG_THRESHOLD,
    TREND_WEAK_THRESHOLD,
    VOL_LOW_THRESHOLD,
    VOL_HIGH_THRESHOLD,
    # Retrieval
    MIN_CANDIDATES_THRESHOLD,
    DEFAULT_RETRIEVE_K,
    DEFAULT_RERANK_KEEP,
    DEFAULT_TOP_K,
    # Floors
    DEFAULT_MIN_OOS_SHARPE,
    DEFAULT_MIN_TRADES,
    DEFAULT_MAX_DRAWDOWN_FRAC,
    DEFAULT_MAX_OVERFIT_GAP,
    # Timeouts
    TIMEOUT_EMBED_S,
    TIMEOUT_QDRANT_S,
    TIMEOUT_BACKTEST_S,
    # Weights
    RELAXED_WEIGHT_PENALTY,
    TRADES_REF,
    # Schema
    REGIME_SCHEMA_VERSION,
    TRIAL_DOC_SCHEMA_VERSION,
    KB_TRIALS_COLLECTION_NAME,
    KB_TRIALS_DOC_TYPE,
    # Feature scales
    FEATURE_SCALES,
)

from app.services.kb.embed import (
    KBEmbeddingAdapter,
    EmbeddingResult,
    EmbeddingError,
    EmbeddingTimeoutError,
    EmbeddingBatchError,
    EmbeddingServiceError,
    get_kb_embedder,
)

__all__ = [
    # Types
    "RegimeSnapshot",
    "TrialDoc",
    "compute_trial_warnings",
    "compute_overfit_gap",
    "utc_now_iso",
    # Regime computation
    "compute_regime_snapshot",
    "compute_tags",
    "regime_snapshot_to_text",
    "compute_atr",
    "compute_rsi",
    "compute_bollinger_bands",
    "compute_bb_width_pct",
    "compute_efficiency_ratio",
    "compute_zscore",
    "compute_trend_strength",
    "compute_trend_direction",
    # Trial doc
    "trial_to_text",
    "trial_to_metadata",
    "trial_to_summary",
    "build_trial_doc_from_tune_run",
    # Parsing
    "ParsedDataset",
    "parse_ohlcv_for_kb",
    "compute_fingerprint_from_bytes",
    "detect_timeframe_from_filename",
    "validate_for_regime",
    # Constants
    "REGIME_WINDOW_BARS",
    "MIN_EFFECTIVE_BARS",
    "DEFAULT_FEATURE_PARAMS",
    "TREND_STRONG_THRESHOLD",
    "TREND_WEAK_THRESHOLD",
    "VOL_LOW_THRESHOLD",
    "VOL_HIGH_THRESHOLD",
    "MIN_CANDIDATES_THRESHOLD",
    "DEFAULT_RETRIEVE_K",
    "DEFAULT_RERANK_KEEP",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_OOS_SHARPE",
    "DEFAULT_MIN_TRADES",
    "DEFAULT_MAX_DRAWDOWN_FRAC",
    "DEFAULT_MAX_OVERFIT_GAP",
    "TIMEOUT_EMBED_S",
    "TIMEOUT_QDRANT_S",
    "TIMEOUT_BACKTEST_S",
    "RELAXED_WEIGHT_PENALTY",
    "TRADES_REF",
    "REGIME_SCHEMA_VERSION",
    "TRIAL_DOC_SCHEMA_VERSION",
    "KB_TRIALS_COLLECTION_NAME",
    "KB_TRIALS_DOC_TYPE",
    "FEATURE_SCALES",
    # Embedding
    "KBEmbeddingAdapter",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingTimeoutError",
    "EmbeddingBatchError",
    "EmbeddingServiceError",
    "get_kb_embedder",
]
