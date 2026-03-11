"""
ABAC Context
-------------
Lightweight attribute container that travels with every agent call.
Built from HTTP headers by the FastAPI dependency; used by PolicyEngine.

Roles:
    admin       -- Red Key Sandbox senior staff; full access
    msp         -- MSP engineer / support staff; read most data
    finance     -- billing team; read/write billing data
    technician  -- field tech; read PSA + contracts
    client      -- external client; read own tickets only
    readonly    -- audit/reporting; read non-sensitive data
    anonymous   -- unauthenticated; no sensitive access

Classifications:
    public      -- AI research articles, Wikipedia, RSS
    internal    -- PSA tickets, general CRM notes
    sensitive   -- billing invoices, AR data, CRM RED accounts
    restricted  -- contracts, PII fields, communications history
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ABACContext:
    """
    Attribute-Based Access Control context for a single agent call.

    Pass this into PolicyEngine.evaluate() and AgentSecurityGateway.handle().
    """

    user_role: str = "anonymous"
    """Portal role from JWT claim: admin | msp | finance | technician | client | readonly | anonymous"""

    data_classification: str = "internal"
    """Data tier being accessed: public | internal | sensitive | restricted"""

    environment: str = "production"
    """Deployment environment: production | staging | dev"""

    username: str = ""
    """Authenticated username (empty for anonymous calls)."""

    client_id: str | None = None
    """For client-role callers: their client_id (limits scope to own data)."""

    extra_attrs: dict[str, Any] = field(default_factory=dict)
    """Extension point for future attributes (e.g. IP, MFA status, session age)."""

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def to_eval_namespace(self) -> dict[str, Any]:
        """
        Return a flat dict safe for use inside policy rule eval().
        Only contains string/None values — no builtins.
        """
        return {
            "user_role":            self.user_role,
            "data_classification":  self.data_classification,
            "environment":          self.environment,
            "username":             self.username,
            "client_id":            self.client_id,
            **self.extra_attrs,
        }

    @classmethod
    def anonymous(cls) -> "ABACContext":
        """Return a minimal anonymous context (most restrictive)."""
        return cls(user_role="anonymous", data_classification="public")

    @classmethod
    def from_jwt_payload(cls, payload: dict) -> "ABACContext":
        """Build context from a decoded JWT payload (as used by app/auth.py)."""
        return cls(
            user_role=payload.get("role", "anonymous"),
            username=payload.get("sub", ""),
            client_id=payload.get("client_id"),
            data_classification="internal",  # default; overridden per-tool
            environment="production",
        )
