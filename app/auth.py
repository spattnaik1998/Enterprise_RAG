"""
Portal Authentication
---------------------
JWT-based auth for the Red Key Sandbox Client Portal.

Each token carries:
  - sub         : username
  - role        : 'msp' | 'client'
  - client_id   : str | None  (None for MSP)
  - client_name : str | None  (None for MSP)
  - exp         : expiry (24 h from issue)

Usage in routes:
    from app.auth import require_msp, require_client

    @app.get("/api/msp/tickets")
    async def msp_tickets(user: dict = Depends(require_msp)):
        ...

    @app.get("/api/portal/tickets")
    async def my_tickets(user: dict = Depends(require_client)):
        client_id = user["client_id"]
        ...
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE_ME_generate_with_secrets_token_hex_32")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRY_HOURS: int = 24

_bearer = HTTPBearer(auto_error=True)


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* value."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def create_token(
    username: str,
    role: str,
    client_id: Optional[str] = None,
    client_name: Optional[str] = None,
) -> str:
    """Sign and return a JWT for the given user."""
    payload = {
        "sub": username,
        "role": role,
        "client_id": client_id,
        "client_name": client_name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises 401 on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Validate the Bearer token and return the decoded payload."""
    return decode_token(credentials.credentials)


def require_msp(user: dict = Depends(get_current_user)) -> dict:
    """Guard: requires role == 'msp'. Raises 403 otherwise."""
    if user.get("role") != "msp":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MSP admin access required.",
        )
    return user


def require_client(user: dict = Depends(get_current_user)) -> dict:
    """Guard: requires role == 'client'. Raises 403 otherwise."""
    if user.get("role") != "client":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client portal access required.",
        )
    return user
