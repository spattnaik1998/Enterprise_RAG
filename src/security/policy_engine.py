"""
Policy Engine
--------------
Evaluates YAML-defined ABAC policies against an ABACContext.

Policy DSL (config/policies.yaml):
    policies:
      - id: allow_billing_read
        action: read_billing
        abac:
          required_attrs: [user_role, data_classification]
          rules:
            - if: "data_classification == 'sensitive' and user_role not in ['finance','admin']"
              effect: deny
        justification_required: false

Rules are evaluated using a restricted eval() namespace containing only the
ABACContext fields. If ANY deny rule fires, the action is blocked.
If no deny rule fires (or no rules at all), the action is allowed.
"""
from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.security.abac import ABACContext


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    policy_ids_evaluated: list[str] = field(default_factory=list)
    justification_required: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_DEFAULT_POLICY_FILE = Path("config/policies.yaml")

# Minimal safe builtins allowed in rule eval
_SAFE_BUILTINS: dict[str, Any] = {
    "True": True, "False": False, "None": None,
    "int": int, "str": str, "float": float, "bool": bool,
    "len": len, "in": None,  # 'in' is a keyword, not a builtin
}


class PolicyEngine:
    """
    Loads policies from YAML and evaluates them against ABACContext.

    Usage:
        engine = PolicyEngine()
        decision = engine.evaluate("read_billing", ctx)
        if not decision.allowed:
            raise PermissionError(decision.reasons)
    """

    def __init__(self, policy_file: str | Path = _DEFAULT_POLICY_FILE) -> None:
        self._policies: list[dict] = []
        self._action_index: dict[str, list[dict]] = {}
        self.load_policies(policy_file)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load_policies(self, path: str | Path) -> None:
        """Load and parse policies.yaml. Safe to call multiple times (hot-reload)."""
        p = Path(path)
        if not p.exists():
            logger.warning(
                f"[PolicyEngine] Policy file not found: {p}. "
                "All actions will be ALLOWED by default (open mode)."
            )
            self._policies = []
            self._action_index = {}
            return

        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self._policies = raw.get("policies", [])
        self._action_index = {}
        for policy in self._policies:
            action = policy.get("action", "")
            self._action_index.setdefault(action, []).append(policy)

        logger.info(
            f"[PolicyEngine] Loaded {len(self._policies)} policies "
            f"covering {len(self._action_index)} actions from {p}"
        )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    def evaluate(self, action: str, ctx: ABACContext) -> PolicyDecision:
        """
        Evaluate all policies for the given action against ctx.

        Returns PolicyDecision(allowed=False) if any deny rule fires;
        PolicyDecision(allowed=True) otherwise.
        """
        relevant = self._action_index.get(action, [])
        if not relevant:
            # No policy defined for this action -> allow (open default)
            return PolicyDecision(
                allowed=True,
                reasons=["No policy defined; open default allows."],
                policy_ids_evaluated=[],
            )

        namespace = ctx.to_eval_namespace()
        evaluated_ids: list[str] = []
        justification_required = False

        for policy in relevant:
            pid = policy.get("id", "unnamed")
            evaluated_ids.append(pid)

            if policy.get("justification_required", False):
                justification_required = True

            abac_block = policy.get("abac", {})
            rules = abac_block.get("rules", [])

            for rule in rules:
                condition = rule.get("if", "")
                effect    = rule.get("effect", "allow")

                if effect != "deny":
                    continue  # only deny rules can block

                fired = self._eval_condition(condition, namespace, pid)
                if fired:
                    reason = (
                        f"Policy '{pid}' denied action '{action}': "
                        f"rule condition '{condition}' matched."
                    )
                    logger.warning(f"[PolicyEngine] DENY | {reason}")
                    return PolicyDecision(
                        allowed=False,
                        reasons=[reason],
                        policy_ids_evaluated=evaluated_ids,
                        justification_required=justification_required,
                    )

        return PolicyDecision(
            allowed=True,
            reasons=[],
            policy_ids_evaluated=evaluated_ids,
            justification_required=justification_required,
        )

    # -------------------------------------------------------------------------
    # Rule evaluation (restricted eval)
    # -------------------------------------------------------------------------

    def _eval_condition(self, condition: str, namespace: dict, policy_id: str) -> bool:
        """
        Evaluate a policy rule condition string.

        Uses a tightly-restricted namespace with only the ABAC context fields
        and a small allowlist of safe builtins. Never exposes __builtins__.
        """
        if not condition.strip():
            return False

        # Build a clean namespace: ABAC attrs + safe type helpers
        safe_ns: dict[str, Any] = {
            "__builtins__": {},   # block all built-ins
            "not": None,          # not is a keyword, not a name
        }
        safe_ns.update(namespace)

        # Allow 'in' via list membership using a helper
        # (actual 'in' operator works natively in eval)
        try:
            return bool(eval(condition, safe_ns))  # noqa: S307
        except NameError as exc:
            logger.warning(
                f"[PolicyEngine] Rule eval NameError in policy '{policy_id}': {exc} "
                f"| condition={condition!r}"
            )
            return False
        except Exception as exc:
            logger.error(
                f"[PolicyEngine] Rule eval error in policy '{policy_id}': {exc} "
                f"| condition={condition!r}"
            )
            return False

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @property
    def policy_count(self) -> int:
        return len(self._policies)

    @property
    def action_count(self) -> int:
        return len(self._action_index)
