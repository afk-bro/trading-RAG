"""SSE ticket generation and verification.

Provides short-lived signed tickets for SSE connections to avoid
exposing the raw admin token in cookies or headers for long-lived
connections.

Security model:
- Admin loads page → page requests SSE ticket via POST
- Ticket is signed JWT with short expiry (5 min default)
- SSE connection uses ticket cookie for auth
- XSS cannot steal raw admin token (only short-lived ticket)
"""

import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# Default expiry if not configured (5 minutes)
DEFAULT_TICKET_EXPIRY_SECONDS = 300


@dataclass
class SSETicketClaims:
    """Claims contained in an SSE ticket."""

    workspace_id: Optional[UUID]  # None = all workspaces (admin)
    issued_at: float  # Unix timestamp
    expires_at: float  # Unix timestamp
    is_admin: bool = False


def _get_ticket_secret() -> str:
    """
    Get the SSE ticket secret.

    Falls back to ADMIN_TOKEN if SSE_TICKET_SECRET not set,
    or generates a random secret for single-session use.
    """
    from app.config import get_settings

    settings = get_settings()

    if settings.sse_ticket_secret:
        return settings.sse_ticket_secret

    # Fallback to ADMIN_TOKEN
    admin_token = os.environ.get("ADMIN_TOKEN")
    if admin_token:
        # Use derived key to avoid exposing admin token directly
        return f"sse-{admin_token}"

    # Last resort: random per-process secret (tickets invalid after restart)
    logger.warning(
        "No SSE_TICKET_SECRET configured, using ephemeral secret",
        note="SSE tickets will be invalid after server restart",
    )
    return secrets.token_urlsafe(32)


# Cache the secret for the process lifetime
_cached_secret: Optional[str] = None


def _get_secret() -> str:
    """Get cached secret (computed once per process)."""
    global _cached_secret
    if _cached_secret is None:
        _cached_secret = _get_ticket_secret()
    return _cached_secret


def _get_expiry_seconds() -> int:
    """Get ticket expiry from settings."""
    try:
        from app.config import get_settings
        return get_settings().sse_ticket_expiry_seconds
    except Exception:
        return DEFAULT_TICKET_EXPIRY_SECONDS


def create_sse_ticket(
    workspace_id: Optional[UUID] = None,
    is_admin: bool = True,
    expiry_seconds: Optional[int] = None,
) -> str:
    """
    Create a short-lived signed ticket for SSE connections.

    Format: base64url(payload).base64url(signature)
    Payload: workspace_id:issued_at:expires_at:is_admin

    Args:
        workspace_id: Workspace to scope ticket to (None = all)
        is_admin: Whether this is an admin ticket
        expiry_seconds: Custom expiry (defaults to config)

    Returns:
        Signed ticket string
    """
    import base64

    if expiry_seconds is None:
        expiry_seconds = _get_expiry_seconds()

    now = time.time()
    expires_at = now + expiry_seconds

    # Build payload
    ws_str = str(workspace_id) if workspace_id else "*"
    payload = f"{ws_str}:{now:.0f}:{expires_at:.0f}:{1 if is_admin else 0}"
    payload_bytes = payload.encode("utf-8")

    # Sign with HMAC-SHA256
    secret = _get_secret()
    signature = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        digestmod="sha256",
    ).digest()

    # Encode as base64url
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii")
    sig_b64 = base64.urlsafe_b64encode(signature).decode("ascii")

    return f"{payload_b64}.{sig_b64}"


def verify_sse_ticket(ticket: str) -> Optional[SSETicketClaims]:
    """
    Verify and decode an SSE ticket.

    Args:
        ticket: The ticket string to verify

    Returns:
        SSETicketClaims if valid, None if invalid/expired
    """
    import base64

    if not ticket or "." not in ticket:
        logger.debug("sse_ticket_invalid", reason="malformed")
        return None

    try:
        parts = ticket.split(".", 1)
        if len(parts) != 2:
            logger.debug("sse_ticket_invalid", reason="wrong_parts")
            return None

        payload_b64, sig_b64 = parts

        # Decode payload
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        payload = payload_bytes.decode("utf-8")

        # Verify signature (constant-time comparison)
        secret = _get_secret()
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            digestmod="sha256",
        ).digest()
        provided_sig = base64.urlsafe_b64decode(sig_b64)

        if not hmac.compare_digest(expected_sig, provided_sig):
            logger.debug("sse_ticket_invalid", reason="signature_mismatch")
            return None

        # Parse payload: workspace_id:issued_at:expires_at:is_admin
        parts = payload.split(":")
        if len(parts) != 4:
            logger.debug("sse_ticket_invalid", reason="payload_format")
            return None

        ws_str, issued_str, expires_str, admin_str = parts

        # Check expiry
        expires_at = float(expires_str)
        if time.time() > expires_at:
            logger.debug("sse_ticket_invalid", reason="expired")
            return None

        # Parse workspace_id
        workspace_id = None if ws_str == "*" else UUID(ws_str)

        return SSETicketClaims(
            workspace_id=workspace_id,
            issued_at=float(issued_str),
            expires_at=expires_at,
            is_admin=admin_str == "1",
        )

    except Exception as e:
        logger.debug("sse_ticket_invalid", reason="parse_error", error=str(e))
        return None


def get_sse_auth(
    ticket_cookie: Optional[str] = None,
    admin_header: Optional[str] = None,
) -> Optional[SSETicketClaims]:
    """
    Get SSE auth from ticket cookie or admin header.

    Priority:
    1. SSE ticket cookie (preferred, short-lived)
    2. X-Admin-Token header (fallback for curl testing)

    Args:
        ticket_cookie: Value of sse_ticket cookie
        admin_header: Value of X-Admin-Token header

    Returns:
        SSETicketClaims if authenticated, None otherwise
    """
    # 1. Try SSE ticket cookie
    if ticket_cookie:
        claims = verify_sse_ticket(ticket_cookie)
        if claims:
            return claims

    # 2. Fallback to admin token header
    if admin_header:
        from app.deps.security import verify_admin_token
        if verify_admin_token(admin_header):
            return SSETicketClaims(
                workspace_id=None,  # Admin has access to all
                issued_at=time.time(),
                expires_at=time.time() + 3600,  # 1 hour for header auth
                is_admin=True,
            )

    return None
