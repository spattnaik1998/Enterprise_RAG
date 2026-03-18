"""
API Rate Limiting
------------------
Per-role request throttling using slowapi.

Limits (per minute):
  admin:  120 req/min on chat, 40 on council
  msp:     60 req/min on chat, 20 on council
  client:  30 req/min on chat
  default: 15 req/min (safety net)
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request
from loguru import logger


def _key_func(request: Request) -> str:
    """Extract rate-limit key from JWT user identity or fall back to IP."""
    # The auth dependency injects user info; we read from request state
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        return user.get("sub", get_remote_address(request))
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)


def get_role_limit(request: Request, base_limit: int = 60) -> str:
    """Return rate limit string based on user role."""
    user = getattr(request.state, "user", None)
    role = user.get("role", "default") if user and isinstance(user, dict) else "default"
    multipliers = {"admin": 2.0, "msp": 1.0, "finance": 1.0, "client": 0.5}
    mult = multipliers.get(role, 0.25)
    limit = max(1, int(base_limit * mult))
    return f"{limit}/minute"
