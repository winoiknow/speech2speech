# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Admin access gate (Phase 5, Workstream A) — ARCHITECTURE.md §11.1.

Two trust planes, deliberately separate:

  * **Human/admin surface** (`/enroll`, the `/v1/speakers*` admin APIs): a browser
    session established by EITHER in-app **OIDC SSO** (authlib) for designated
    admins, OR a **single local-admin** bootstrap token (first-run / SSO-down).
  * **Machine data plane** (`/v1/identify`, `/v1/diarize`): an optional shared
    `SERVICE_API_KEY` (the s2s client already sends it as a bearer) — NOT the human
    session, so s2s keeps working without an admin login.

`/healthz` is always open (liveness).

**Default is open + a loud warning** so existing trusted-network deployments keep
working on upgrade. The gate enforces only once an admin auth is configured
(`OIDC_ENABLED=1` or `LOCAL_ADMIN_TOKEN` set); the data-plane key enforces only
when `SERVICE_API_KEY` is set.

TLS is out of scope here (no sidecar): terminate it at your reverse proxy. OIDC
redirect URIs must be https, and TLS is also what satisfies the browser
secure-context requirement for mic capture on the enroll page.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger("speaker_id.auth")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str) -> set[str]:
    return {v.strip().lower() for v in os.environ.get(name, "").split(",") if v.strip()}


@dataclass
class AuthConfig:
    # OIDC (in-app, authlib)
    oidc_enabled: bool = field(default_factory=lambda: _env_bool("OIDC_ENABLED", False))
    oidc_discovery_url: str = field(default_factory=lambda: os.environ.get("OIDC_DISCOVERY_URL", ""))
    oidc_client_id: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_ID", ""))
    oidc_client_secret: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_SECRET", ""))
    # Authorization allowlist — emails and/or a domain. Empty = ANY IdP-authenticated
    # user is admin (logged as a warning; lock this down in production).
    oidc_allowed_emails: set[str] = field(default_factory=lambda: _csv("OIDC_ALLOWED_EMAILS"))
    oidc_allowed_domain: str = field(default_factory=lambda: os.environ.get("OIDC_ALLOWED_DOMAIN", "").strip().lower())
    # Single local admin (bootstrap token). Compared in constant time.
    local_admin_token: str = field(default_factory=lambda: os.environ.get("LOCAL_ADMIN_TOKEN", ""))
    # Signed-cookie session secret. Auto-generated if unset (sessions won't survive
    # a restart — set it in prod). Only relevant when the admin gate is enforced.
    session_secret: str = field(default_factory=lambda: os.environ.get("SESSION_SECRET", ""))
    # Machine data-plane shared secret for /v1/identify + /v1/diarize.
    service_api_key: str = field(default_factory=lambda: os.environ.get("SERVICE_API_KEY", ""))

    @property
    def admin_enforced(self) -> bool:
        return self.oidc_enabled or bool(self.local_admin_token)


_config = AuthConfig()
_oauth = None  # authlib OAuth registry, built lazily when OIDC is enabled


def get_config() -> AuthConfig:
    return _config


def session_secret() -> str:
    """Secret for SessionMiddleware. Stable if SESSION_SECRET is set, else ephemeral."""
    if _config.session_secret:
        return _config.session_secret
    if _config.admin_enforced:
        logger.warning("SESSION_SECRET unset — generating an ephemeral one; admin sessions "
                       "will not survive a restart. Set SESSION_SECRET in production.")
    return secrets.token_urlsafe(32)


def log_startup_state() -> None:
    if not _config.admin_enforced:
        logger.warning("ADMIN GATE OPEN — /enroll and the speaker admin APIs are UNAUTHENTICATED. "
                       "Set OIDC_ENABLED=1 (+OIDC_*) or LOCAL_ADMIN_TOKEN to lock them down. "
                       "Keep the service on a trusted network until then.")
    else:
        modes = []
        if _config.oidc_enabled:
            modes.append("OIDC")
        if _config.local_admin_token:
            modes.append("local-admin")
        logger.info("Admin gate ENFORCED via %s", "+".join(modes))
        if _config.oidc_enabled and not (_config.oidc_allowed_emails or _config.oidc_allowed_domain):
            logger.warning("OIDC has no allowlist (OIDC_ALLOWED_EMAILS / OIDC_ALLOWED_DOMAIN) — "
                           "ANY user your IdP authenticates becomes an admin.")
    if not _config.service_api_key:
        logger.info("Data plane (/v1/identify, /v1/diarize) is OPEN — set SERVICE_API_KEY to require a key.")


def _ensure_oauth():
    global _oauth
    if _oauth is not None:
        return _oauth
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
    oauth.register(
        name="idp",
        server_metadata_url=_config.oidc_discovery_url,
        client_id=_config.oidc_client_id,
        client_secret=_config.oidc_client_secret,
        client_kwargs={"scope": "openid email profile"},
    )
    _oauth = oauth
    return _oauth


# ── auth checks ───────────────────────────────────────────────────────────────


def _bearer(request: Request) -> Optional[str]:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else None


def admin_session(request: Request) -> Optional[dict]:
    """Non-raising: the current admin identity, or None. Used by the page route."""
    if not _config.admin_enforced:
        return {"via": "open"}
    admin = request.session.get("admin")
    if admin:
        return admin
    # Allow the local-admin token as a bearer too, so scripted/curl admin works.
    tok = _bearer(request)
    if _config.local_admin_token and tok and hmac.compare_digest(tok, _config.local_admin_token):
        return {"via": "local-bearer"}
    return None


def require_admin(request: Request) -> dict:
    """FastAPI dependency for admin APIs — raises 401 when not authenticated."""
    admin = admin_session(request)
    if admin is None:
        raise HTTPException(status_code=401, detail="admin authentication required")
    return admin


def require_service_key(request: Request) -> None:
    """FastAPI dependency for the machine data plane. No-op unless SERVICE_API_KEY set.

    A signed-in admin (session cookie) is also allowed, so the enroll page's
    test-my-voice works without holding the service key.
    """
    if not _config.service_api_key:
        return
    tok = _bearer(request)
    if tok and hmac.compare_digest(tok, _config.service_api_key):
        return
    if request.session.get("admin"):  # an authenticated admin may exercise the data plane
        return
    raise HTTPException(status_code=401, detail="service api key required")


def _authorized_email(email: Optional[str]) -> bool:
    if not _config.oidc_allowed_emails and not _config.oidc_allowed_domain:
        return True  # no allowlist → any authenticated IdP user (warned at startup)
    if not email:
        return False
    email = email.lower()
    if email in _config.oidc_allowed_emails:
        return True
    if _config.oidc_allowed_domain and email.endswith("@" + _config.oidc_allowed_domain):
        return True
    return False


# ── routes ────────────────────────────────────────────────────────────────────

router = APIRouter()


def _login_page() -> str:
    sso = ('<a class="btn" href="/auth/oidc/login">Sign in with SSO</a>'
           if _config.oidc_enabled else "")
    local = ('<form method="post" action="/auth/local">'
             '<input type="password" name="token" placeholder="Local admin token" autofocus required>'
             '<button class="btn" type="submit">Sign in</button></form>'
             if _config.local_admin_token else "")
    if not (sso or local):
        local = "<p>No admin auth configured — the console is open.</p>"
    return f"""<!doctype html><meta charset=utf-8><title>speaker-id admin sign in</title>
<style>body{{font:16px system-ui;max-width:22rem;margin:5rem auto;padding:0 1rem}}
.btn{{display:inline-block;margin:.4rem 0;padding:.6rem 1rem;border:1px solid #888;border-radius:.4rem;
background:#f5f5f5;text-decoration:none;color:#111;cursor:pointer}}
input{{width:100%;padding:.5rem;margin:.3rem 0;box-sizing:border-box}}</style>
<h2>speaker-id admin</h2>{sso}{local}"""


@router.get("/auth/login", include_in_schema=False)
def login_page() -> HTMLResponse:
    return HTMLResponse(_login_page())


@router.get("/auth/oidc/login", include_in_schema=False)
async def oidc_login(request: Request):
    if not _config.oidc_enabled:
        raise HTTPException(status_code=404, detail="OIDC not enabled")
    redirect_uri = request.url_for("oidc_callback")
    # Register THIS EXACT value as the redirect URI in your IdP (strict match,
    # incl. scheme + host + path). Not /enroll — that's the post-login landing page.
    logger.info("OIDC redirect_uri (register this in your IdP): %s", redirect_uri)
    return await _ensure_oauth().idp.authorize_redirect(request, redirect_uri)


async def _exchange_code(app, request: Request) -> dict:
    """Exchange the auth code for tokens WITHOUT verifying the id_token signature.

    authlib's ``authorize_access_token`` auto-parses the id_token, which fetches the
    IdP JWKS and builds a key set — a step that fails with ``KeyError: 'keys'`` / a
    ValueError across authlib versions when the JWKS can't be loaded. We don't need
    the id_token: identity comes from the userinfo endpoint (just a bearer call), so
    we replicate the method minus the parse, keeping the state/CSRF validation.
    """
    params = {"code": request.query_params.get("code"), "state": request.query_params.get("state")}
    state_data = await app.framework.get_state_data(request.session, params["state"])
    await app.framework.clear_state_data(request.session, params["state"])
    params = app._format_state_params(state_data, params)  # raises on state mismatch (CSRF)
    return await app.fetch_access_token(**params)


@router.get("/auth/oidc/callback", name="oidc_callback", include_in_schema=False)
async def oidc_callback(request: Request):
    if not _config.oidc_enabled:
        raise HTTPException(status_code=404, detail="OIDC not enabled")
    error = request.query_params.get("error")
    if error:
        raise HTTPException(status_code=401, detail=f"OIDC error: {error} "
                            f"{request.query_params.get('error_description', '')}".strip())
    oauth = _ensure_oauth()
    try:
        token = await _exchange_code(oauth.idp, request)
        userinfo = dict(await oauth.idp.userinfo(token=token))
    except Exception as e:
        # Surface the real reason (mismatching_state from a lost session cookie,
        # redirect_uri mismatch, unreachable userinfo, …) instead of a generic 401.
        logger.exception("OIDC code exchange / userinfo failed")
        raise HTTPException(status_code=401, detail=f"OIDC authentication failed: {e}")

    email = userinfo.get("email")
    if not _authorized_email(email):
        logger.warning("OIDC user %r not on the admin allowlist → denied", email)
        raise HTTPException(
            status_code=403,
            detail=(f"{email or 'this account'} is not an authorized admin — add it to "
                    "OIDC_ALLOWED_EMAILS or OIDC_ALLOWED_DOMAIN"),
        )
    request.session["admin"] = {"via": "oidc", "email": email, "sub": userinfo.get("sub")}
    logger.info("admin login via OIDC: %s", email)
    return RedirectResponse(url="/enroll", status_code=303)


@router.post("/auth/local", include_in_schema=False)
def local_login(request: Request, token: str = Form(...)):
    if not _config.local_admin_token or not hmac.compare_digest(token, _config.local_admin_token):
        raise HTTPException(status_code=401, detail="invalid local admin token")
    request.session["admin"] = {"via": "local"}
    logger.info("admin login via local-admin token")
    return RedirectResponse(url="/enroll", status_code=303)


@router.get("/auth/logout", include_in_schema=False)
def logout(request: Request):
    request.session.pop("admin", None)
    return RedirectResponse(url="/auth/login", status_code=303)
