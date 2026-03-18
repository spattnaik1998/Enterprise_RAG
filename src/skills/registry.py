"""
Skill Registry
--------------
Central registry for discovering and accessing available skills.
"""
from __future__ import annotations

from loguru import logger
from typing import Optional

from src.skills.base import Skill


class SkillRegistry:
    """Singleton registry for all available skills."""

    _instance: Optional[SkillRegistry] = None
    _skills: dict[str, Skill] = {}

    def __new__(cls) -> SkillRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Lazy-load skills on first access."""
        if not self._skills:
            self._load_skills()

    def _load_skills(self) -> None:
        """Load all available skills."""
        try:
            from src.skills.ar_risk_report import ARRiskReportSkill
            from src.skills.ticket_triage import TicketTriageSkill
            from src.skills.contract_renewal import ContractRenewalSkill
            from src.skills.client_health import ClientHealthSkill
            from src.skills.escalation_brief import EscalationBriefSkill

            skills = [
                ARRiskReportSkill(),
                TicketTriageSkill(),
                ContractRenewalSkill(),
                ClientHealthSkill(),
                EscalationBriefSkill(),
            ]

            for skill in skills:
                self._skills[skill.name] = skill
                logger.info(f"[SkillRegistry] Loaded skill: {skill.name}")

        except Exception as e:
            logger.error(f"[SkillRegistry] Failed to load skills: {e}")

    def register(self, skill: Skill) -> None:
        """Register a new skill."""
        self._skills[skill.name] = skill
        logger.info(f"[SkillRegistry] Registered skill: {skill.name}")

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        self._initialize()
        return self._skills.get(name)

    def list_skills(self) -> list[dict]:
        """List all available skills with metadata."""
        self._initialize()
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
                "required_sources": skill.required_sources,
            }
            for skill in self._skills.values()
        ]

    def match(self, query: str) -> Optional[Skill]:
        """
        Find best-matching skill for a query.

        Returns the skill with highest match score, or None.
        """
        self._initialize()
        if not self._skills:
            return None

        best_match = None
        best_score = 0.0

        for skill in self._skills.values():
            score = skill.matches_query(query)
            if score > best_score:
                best_score = score
                best_match = skill

        return best_match if best_score > 0.3 else None

    def clear(self) -> None:
        """Clear all skills (mainly for testing)."""
        self._skills.clear()

    @property
    def skills(self) -> dict[str, Skill]:
        """Access all skills directly."""
        self._initialize()
        return self._skills
