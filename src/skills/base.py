"""
Skill Base Classes
------------------
Abstract base for all skills and supporting dataclasses.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SkillContext:
    """Context passed to skill execution."""

    query: str
    user_id: Optional[str] = None
    user_role: str = "msp"
    metadata: dict = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result returned from skill execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    skill_name: str = ""


class Skill(ABC):
    """
    Abstract base class for all MSP skills.

    Subclasses must implement:
      - name: str
      - description: str
      - required_sources: list[str] (e.g., ["billing", "crm"])
      - execute(context: SkillContext) -> SkillResult
    """

    name: str = "unknown"
    description: str = ""
    required_sources: list[str] = []
    version: str = "1.0.0"

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill.

        Args:
            context: SkillContext with query, user info, metadata

        Returns:
            SkillResult with success, data, error, latency_ms
        """
        pass

    def matches_query(self, query: str) -> float:
        """
        Score how well this skill matches a query (0.0 to 1.0).

        Default: simple keyword matching against description.

        Override in subclasses for better matching.

        Args:
            query: User query string

        Returns:
            Match score 0.0 (no match) to 1.0 (perfect match)
        """
        query_lower = query.lower()
        matches = sum(1 for word in self.name.lower().split() if word in query_lower)
        return min(1.0, matches / max(1, len(self.name.split())))
