"""Multi-agent orchestration package: BaseAgent, CouncilOrchestrator, DeadlockDetector."""
from src.agents.base import BaseAgent, AgentProposal
from src.agents.council import CouncilOrchestrator, CouncilVerdict
from src.agents.deadlock import DeadlockDetector

__all__ = [
    "BaseAgent", "AgentProposal",
    "CouncilOrchestrator", "CouncilVerdict",
    "DeadlockDetector",
]
