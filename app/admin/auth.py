"""Admin authentication routes - login, logout, and session management."""

import hmac
import os
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Cookie settings
COOKIE_NAME = "admin_token"
COOKIE_MAX_AGE_REMEMBER = 30 * 24 * 60 * 60  # 30 days
COOKIE_MAX_AGE_SESSION = None  # Session cookie (browser close)


class LoginRequest(BaseModel):
    """Login request body."""

    token: str
    remember: bool = False


def verify_token(token: str) -> bool:
    """Verify admin token using constant-time comparison."""
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not admin_token:
        return False
    return hmac.compare_digest(token.encode(), admin_token.encode())


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Public landing page - no auth required."""
    return templates.TemplateResponse("landing.html", {"request": request})


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, redirect: Optional[str] = None):
    """Login page - no auth required."""
    # Check if already authenticated via cookie
    token = request.cookies.get(COOKIE_NAME)
    if token and verify_token(token):
        # Already logged in, redirect to admin
        return RedirectResponse(url=redirect or "/admin/ingest", status_code=302)

    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/auth/login")
async def login(request: Request, body: LoginRequest):
    """
    Authenticate with admin token and set cookie.

    Returns JSON response with cookie set on success.
    """
    if not verify_token(body.token):
        logger.warning(
            "Failed login attempt",
            client=request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid admin token"},
        )

    logger.info(
        "Admin login successful",
        client=request.client.host if request.client else "unknown",
        remember=body.remember,
    )

    response = JSONResponse(content={"success": True, "message": "Login successful"})

    # Set cookie
    max_age = COOKIE_MAX_AGE_REMEMBER if body.remember else COOKIE_MAX_AGE_SESSION
    response.set_cookie(
        key=COOKIE_NAME,
        value=body.token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production with HTTPS
    )

    return response


@router.post("/auth/logout")
async def logout():
    """Clear auth cookie and redirect to home."""
    response = JSONResponse(content={"success": True, "message": "Logged out"})
    response.delete_cookie(key=COOKIE_NAME)
    return response


@router.get("/auth/logout")
async def logout_get():
    """GET logout for simple links - redirects to home."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key=COOKIE_NAME)
    return response


@router.get("/auth/check")
async def check_auth(request: Request):
    """Check if current session is authenticated."""
    token = request.cookies.get(COOKIE_NAME)
    if token and verify_token(token):
        return {"authenticated": True}
    return {"authenticated": False}
