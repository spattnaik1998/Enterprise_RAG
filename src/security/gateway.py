"""
Agent Security Gateway (ASG)
------------------------------
Central enforcement layer between callers and the RAG pipeline / MCP tools.

Pipeline:
    check_prompt_guard -> evaluate_policies -> execute -> audit_log -> return

Usage (server.py):
    from src.security.gateway import AgentSecurityGateway, get_gateway
    gw = get_gateway()
    result = gw.handle(query, abac_ctx, generator=generator)

Usage (MCP tools — decorator):
    @asg_tool(action="read_billing", classification="sensitive")
    async def billing_get_overdue_invoices(...):
        ...
"""
from __future__ import annotations

import hashlib
import functools
import os
from pathlib import Path

from loguru import logger

from src.security.abac import ABACContext
from src.security.audit_logger import AuditEntry, AuditLogger
from src.security.approval_queue import ApprovalQueue
from src.security.policy_engine import PolicyEngine
from src.retrieval.guardrails import PromptGuard


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_gateway: "AgentSecurityGateway | None" = None


def get_gateway() -> "AgentSecurityGateway":
    """Return the module-level gateway singleton (lazy init)."""
    global _gateway
    if _gateway is None:
        _gateway = AgentSecurityGateway()
    return _gateway


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class AgentSecurityGateway:
    """
    Orchestrates security enforcement for every agent call.

    Components wired:
      - PromptGuard (injection detection, loads extra YAML patterns)
      - PolicyEngine (YAML ABAC rule evaluation)
      - AuditLogger (HMAC-signed append-only log)
    """

    def __init__(
        self,
        policy_file: str | Path = "config/policies.yaml",
        audit_log: str | Path = "data/audit/audit.jsonl",
        approval_queue: ApprovalQueue | None = None,
    ) -> None:
        # Compliance: AUDIT_HMAC_KEY must be set in non-dev environments
        environment = os.environ.get("ENVIRONMENT", "dev")
        if environment != "dev" and "AUDIT_HMAC_KEY" not in os.environ:
            raise RuntimeError(
                "AUDIT_HMAC_KEY must be set in non-dev environments. "
                "Refusing to start. Set AUDIT_HMAC_KEY in your environment variables."
            )

        self.guard  = PromptGuard()
        self.policy = PolicyEngine(policy_file)
        self.audit  = AuditLogger(audit_log)
        self.approval_queue = approval_queue or ApprovalQueue()
        logger.info(
            f"[ASG] Initialised | "
            f"policies={self.policy.policy_count} "
            f"actions={self.policy.action_count}"
        )

    # -------------------------------------------------------------------------
    # Main entry — RAG pipeline
    # -------------------------------------------------------------------------

    def handle(
        self,
        query: str,
        ctx: ABACContext,
        generator=None,
        pipeline=None,
        action: str = "rag_query",
    ):
        """
        Run the full security lifecycle for a RAG query, then execute it.

        Returns a QueryResult (same as pipeline.query) or a blocked QueryResult.
        """
        from src.serving.pipeline import QueryResult

        # 1. Prompt-guard check
        guard_result = self.guard.check(query)
        if not guard_result.passed:
            self._log(
                action=action,
                ctx=ctx,
                allowed=False,
                query=query,
                reasons=guard_result.flags,
                policy_ids=[],
            )
            return QueryResult(
                query=query,
                answer=guard_result.blocked_reason,
                citations=[],
                pii_redacted=[],
                blocked=True,
                blocked_reason=guard_result.blocked_reason,
            )

        # 2. Policy evaluation
        decision = self.policy.evaluate(action, ctx)
        self._log(
            action=action,
            ctx=ctx,
            allowed=decision.allowed,
            query=query,
            reasons=decision.reasons,
            policy_ids=decision.policy_ids_evaluated,
        )

        if not decision.allowed:
            reason = "; ".join(decision.reasons) or "Access denied by policy."
            return QueryResult(
                query=query,
                answer=f"Access denied: {reason}",
                citations=[],
                pii_redacted=[],
                blocked=True,
                blocked_reason=reason,
            )

        # 2b. Check if justification required (step-up auth)
        if decision.justification_required:
            logger.info(
                f"[ASG] Justification required for action={action} | "
                f"user={ctx.username} | "
                f"classification={ctx.data_classification}"
            )
            return QueryResult(
                query=query,
                answer="This action requires administrator approval. "
                       "Please provide a business justification and request approval.",
                citations=[],
                pii_redacted=[],
                blocked=True,
                blocked_reason="pending_approval",
            )

        # 3. Execute pipeline
        if pipeline is None:
            raise RuntimeError("[ASG] No pipeline provided to gateway.handle()")
        return pipeline.query(query, generator=generator)

    # -------------------------------------------------------------------------
    # Tool call check (for MCP decorator)
    # -------------------------------------------------------------------------

    def check_tool(
        self,
        action: str,
        ctx: ABACContext,
        tool_name: str = "",
    ) -> tuple[bool, str]:
        """
        Run policy check for an MCP tool call.

        Returns:
            (allowed: bool, reason: str)
        """
        decision = self.policy.evaluate(action, ctx)
        self._log(
            action=action,
            ctx=ctx,
            allowed=decision.allowed,
            query=f"tool:{tool_name}",
            reasons=decision.reasons,
            policy_ids=decision.policy_ids_evaluated,
        )

        if not decision.allowed:
            reason = "; ".join(decision.reasons) or "Access denied by policy."
            return False, reason
        return True, ""

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _log(
        self,
        action: str,
        ctx: ABACContext,
        allowed: bool,
        query: str,
        reasons: list[str],
        policy_ids: list[str],
    ) -> None:
        """Write a signed audit entry."""
        try:
            query_hash = "sha256:" + hashlib.sha256(query.encode()).hexdigest()[:16]
            self.audit.append(
                AuditEntry(
                    action=action,
                    allowed=allowed,
                    user_role=ctx.user_role,
                    data_classification=ctx.data_classification,
                    username=ctx.username,
                    query_hash=query_hash,
                    policy_ids_evaluated=policy_ids,
                    reasons=reasons,
                )
            )
        except Exception as exc:
            # Audit failure is non-fatal (deny-safe: log loss != access grant)
            logger.error(f"[ASG] Audit write failed: {exc}")


# ---------------------------------------------------------------------------
# MCP tool decorator
# ---------------------------------------------------------------------------

def asg_tool(action: str, classification: str = "internal"):
    """
    Decorator for MCP tool functions that enforces ASG policy.

    The decorated async function receives an injected `_abac_ctx` keyword arg
    if callers provide it. If no context is provided, an anonymous context is
    used (most restrictive).

    Example:
        @asg_tool(action="read_billing", classification="sensitive")
        async def billing_get_overdue_invoices(days_overdue: int = 30, **kwargs):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract or create ABAC context
            ctx: ABACContext = kwargs.pop("_abac_ctx", None) or ABACContext.anonymous()
            ctx.data_classification = classification

            gw = get_gateway()
            allowed, reason = gw.check_tool(action, ctx, tool_name=func.__name__)
            if not allowed:
                return [{"error": f"Policy denied: {reason}"}]

            return await func(*args, **kwargs)
        return wrapper
    return decorator
