"""Sentry initialization and configuration."""

import os
from typing import Any, Optional

import sentry_sdk
import structlog
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app import __version__
from app.config import Settings

logger = structlog.get_logger(__name__)


def _before_send(event: dict, hint: dict) -> Optional[dict]:
    """
    Filter out 4xx client errors from Sentry events.

    We don't want to track user errors (401, 403, 404, 422, 429) as exceptions.
    Only 5xx server errors should be captured.
    """
    # Check if this is an HTTP exception
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        # Filter FastAPI/Starlette HTTP exceptions
        if hasattr(exc_value, "status_code"):
            status_code = exc_value.status_code
            if 400 <= status_code < 500:
                return None  # Drop 4xx errors

    # Check response context for status code
    if "contexts" in event:
        response = event.get("contexts", {}).get("response", {})
        status_code = response.get("status_code", 0)
        if 400 <= status_code < 500:
            return None

    return event


def _create_traces_sampler(settings: Settings) -> Any:
    """Create a route-aware sampling function for Sentry traces."""

    def traces_sampler(sampling_context: dict) -> float:
        """
        Route-aware sampling for KB recommend critical path.

        - 100% for KB recommend (filter by kb_status tag in Sentry dashboards)
        - Inherits parent sampling decision if available
        - Default rate for everything else

        Note: We can't sample by response status (degraded/none) at trace start
        since traces_sampler runs before the request. Instead we sample 100%
        for KB recommend and use kb_status/mode tags to filter in Sentry UI.
        """
        # Check for transaction context
        tx_context = sampling_context.get("transaction_context", {})
        tx_name = tx_context.get("name", "")

        # KB recommend - 100% sampling (filter by kb_status/mode tags in Sentry)
        if "/kb/trials/recommend" in tx_name:
            return 1.0

        # Check parent sampling decision
        parent = sampling_context.get("parent_sampled")
        if parent is not None:
            return float(parent)

        # Default sampling rate
        return settings.sentry_traces_sample_rate

    return traces_sampler


def init_sentry(settings: Settings) -> bool:
    """
    Initialize Sentry if DSN is configured.

    Returns True if Sentry was initialized, False otherwise.
    """
    if not settings.sentry_dsn:
        return False

    # Only send ERROR-level logs as Sentry events
    sentry_logging = LoggingIntegration(
        level=None,  # Keep normal log levels
        event_level="ERROR",  # Only ERROR+ become Sentry events
    )

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=os.environ.get("GIT_SHA", f"trading-rag@{__version__}"),
        integrations=[
            sentry_logging,
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        enable_tracing=True,
        traces_sampler=_create_traces_sampler(settings),
        profiles_sample_rate=settings.sentry_profiles_sample_rate,
        send_default_pii=False,
        attach_stacktrace=True,
        before_send=_before_send,
    )

    # Set global tags for service metadata
    sentry_sdk.set_tag("service", "trading-rag")
    sentry_sdk.set_tag("collection", settings.qdrant_collection_active)
    sentry_sdk.set_tag("embed_model", settings.embed_model)
    # Use 768 as default dimension for nomic-embed-text
    sentry_sdk.set_tag("vector_dim", getattr(settings, "embed_dim", 768))

    logger.info(
        "Sentry initialized",
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )

    return True
