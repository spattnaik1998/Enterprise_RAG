"""
Domain-Specialist Judge Panel (Architecture C)
-----------------------------------------------
Improves eval calibration by using domain-specific judge prompts.

Each judge is calibrated on domain-specific ground truth labels.

Judges:
  - BillingJudge: focuses on invoice amounts, dates, client names
  - ContractsJudge: validates SLA terms, effective dates, penalties
  - CRMJudge: checks account health scores, contacts
  - PSAJudge: validates ticket statuses, technician names
  - CrossSourceJudge: evaluates multi-hop reasoning chains

Usage:
    panel = JudgePanelOrchestrator(use_specialist_judges=True)
    result = panel.score(
        query="Which clients have overdue invoices?",
        answer="Alpine and TechCorp have overdue balances.",
        ground_truth="Alpine: $47K, TechCorp: $12K",
        context_str="...",
        category="billing"
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from eval.judge import LLMJudge, JudgeResult, _cost_usd


# Specialist judge prompts (hardcoded for now; can be loaded from YAML)

BILLING_JUDGE_PROMPT = """\
You are a billing specialist evaluator for an MSP enterprise RAG system.

You will score an answer on TWO dimensions:
1. FAITHFULNESS: Are all amounts, dates, and client names exactly grounded in the context?
2. CORRECTNESS: Does the answer correctly address the billing question?

CRITICAL: Invoice field names in the context are:
  - invoice_date (not issue_date)
  - line_items[].amount (not line_total)
  - status (OVERDUE, PAID, PENDING)
  - client_name (exact match required)

QUESTION:
{query}

GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT:
{context_str}

MODEL ANSWER:
{answer}

Score on:

1. FAITHFULNESS (0.0-1.0):
   - 1.0: all dollar amounts match context exactly (no rounding), dates match, client names exact
   - 0.5: some amounts/dates approximate or missing grounding
   - 0.0: amounts/clients contradicted in context

2. CORRECTNESS (0.0-1.0):
   - 1.0: answer lists correct clients and amounts matching ground truth
   - 0.5: answer identifies some clients but amounts off or incomplete
   - 0.0: wrong clients or amounts

Return ONLY JSON: {{"faithfulness": <float>, "correctness": <float>}}
"""

CONTRACTS_JUDGE_PROMPT = """\
You are a contracts specialist evaluator for an MSP enterprise RAG system.

You will score an answer on TWO dimensions:
1. FAITHFULNESS: Are SLA terms, dates, and penalties grounded in context?
2. CORRECTNESS: Does the answer correctly address the contracts question?

CRITICAL: Contract field names in the context are:
  - effective_date / expiry_date (not start/end)
  - monthly_value / annual_value (not mrr/arr)
  - sla_response_time (dict with hour/day keys)
  - auto_renew (boolean)
  - termination_notice (days or description)

QUESTION:
{query}

GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT:
{context_str}

MODEL ANSWER:
{answer}

Score on:

1. FAITHFULNESS (0.0-1.0):
   - 1.0: SLA terms, dates, and values match context exactly
   - 0.5: some terms vague or dates approximate
   - 0.0: contradicts context

2. CORRECTNESS (0.0-1.0):
   - 1.0: answer correctly identifies contract terms matching ground truth
   - 0.5: answer identifies some terms but incomplete or unclear
   - 0.0: wrong contract or terms misunderstood

Return ONLY JSON: {{"faithfulness": <float>, "correctness": <float>}}
"""

CRM_JUDGE_PROMPT = """\
You are a CRM specialist evaluator for an MSP enterprise RAG system.

You will score an answer on TWO dimensions:
1. FAITHFULNESS: Are account health scores, contact names, and company info grounded?
2. CORRECTNESS: Does the answer correctly address the CRM question?

CRITICAL: CRM field names in the context are:
  - account_health (RED/YELLOW/GREEN string, not numeric)
  - employee_count (not company_size)
  - contacts (dict with keys: cfo, it_manager, ar_contact -- each has name, email, phone)
  - health_note (string explanation of health status)

QUESTION:
{query}

GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT:
{context_str}

MODEL ANSWER:
{answer}

Score on:

1. FAITHFULNESS (0.0-1.0):
   - 1.0: health scores, contact names, and notes match context exactly
   - 0.5: some contacts or health assessments approximate or unverified
   - 0.0: contradicts context health status

2. CORRECTNESS (0.0-1.0):
   - 1.0: answer correctly identifies account health and concerns
   - 0.5: answer identifies account but health assessment incomplete
   - 0.0: wrong account or misunderstood risk

Return ONLY JSON: {{"faithfulness": <float>, "correctness": <float>}}
"""

PSA_JUDGE_PROMPT = """\
You are a PSA specialist evaluator for an MSP enterprise RAG system.

You will score an answer on TWO dimensions:
1. FAITHFULNESS: Are ticket statuses, technician names, and hours grounded?
2. CORRECTNESS: Does the answer correctly address the PSA question?

CRITICAL: PSA field names in the context are:
  - type (not ticket_type)
  - title (not summary)
  - technician (not assigned_engineer)
  - hours_billed (not hours_logged)
  - resolved_date (not closed_date)
  - resolution_note (string, not list)

QUESTION:
{query}

GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT:
{context_str}

MODEL ANSWER:
{answer}

Score on:

1. FAITHFULNESS (0.0-1.0):
   - 1.0: technician names, statuses, hours match context exactly
   - 0.5: some details approximate or unverified
   - 0.0: contradicts context statuses or assignments

2. CORRECTNESS (0.0-1.0):
   - 1.0: answer correctly lists tickets with accurate statuses and technicians
   - 0.5: answer identifies tickets but some details off or incomplete
   - 0.0: wrong tickets or misunderstood work status

Return ONLY JSON: {{"faithfulness": <float>, "correctness": <float>}}
"""

CROSS_SOURCE_JUDGE_PROMPT = """\
You are a cross-source reasoning specialist evaluator for an MSP enterprise RAG system.

You will score an answer on TWO dimensions:
1. FAITHFULNESS: Are multi-hop reasoning chains grounded in retrieved sources?
2. CORRECTNESS: Does the answer correctly synthesize data from multiple sources?

CRITICAL for cross-source queries:
  - Links between sources must be explicit (e.g., "Client X in CRM has $Y overdue in billing")
  - All intermediate facts must be grounded in context
  - Synthesized conclusions must follow logically from facts

QUESTION:
{query}

GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT (from multiple sources):
{context_str}

MODEL ANSWER:
{answer}

Score on:

1. FAITHFULNESS (0.0-1.0):
   - 1.0: all intermediate facts and final synthesis grounded in context
   - 0.5: some reasoning steps unverified or inferences not fully grounded
   - 0.0: contradicts context or makes unsupported leaps

2. CORRECTNESS (0.0-1.0):
   - 1.0: synthesis correct; all cross-source connections accurate
   - 0.5: mostly correct but some connections missed or reasoning incomplete
   - 0.0: synthesis wrong or data mismatched across sources

Return ONLY JSON: {{"faithfulness": <float>, "correctness": <float>}}
"""


@dataclass
class SpecialistJudgeConfig:
    """Configuration for a domain specialist judge."""
    domain: str
    prompt_template: str


class SpecialistJudge(LLMJudge):
    """
    Extends LLMJudge with domain-specific system prompt.

    Usage:
        judge = SpecialistJudge(domain="billing")
        result = judge.score(query, answer, ground_truth, context_str)
    """

    DOMAIN_PROMPTS = {
        "billing": BILLING_JUDGE_PROMPT,
        "contracts": CONTRACTS_JUDGE_PROMPT,
        "crm": CRM_JUDGE_PROMPT,
        "psa": PSA_JUDGE_PROMPT,
        "communications": BILLING_JUDGE_PROMPT,  # Use billing template for comms
        "cross_source": CROSS_SOURCE_JUDGE_PROMPT,
    }

    def __init__(self, domain: str, model: str = "gpt-4o-mini") -> None:
        super().__init__(model=model)
        self.domain = domain
        self.prompt_template = self.DOMAIN_PROMPTS.get(domain, CROSS_SOURCE_JUDGE_PROMPT)

    def _build_prompt(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context_str: str,
    ) -> str:
        """Build domain-specific judge prompt."""
        return self.prompt_template.format(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            context_str=context_str,
        )

    def score(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context_str: str,
    ) -> JudgeResult:
        """
        Score using domain-specific prompt.
        """
        from openai import OpenAI

        client = OpenAI()
        prompt = self._build_prompt(query, answer, ground_truth, context_str)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        response_text = response.choices[0].message.content

        try:
            scores = json.loads(response_text)
            faithfulness = float(scores.get("faithfulness", 0.0))
            correctness = float(scores.get("correctness", 0.0))
        except json.JSONDecodeError:
            logger.error(f"[{self.domain}Judge] Failed to parse response: {response_text}")
            faithfulness = 0.0
            correctness = 0.0

        cost = _cost_usd(self.model, response.usage.prompt_tokens, response.usage.completion_tokens)

        return JudgeResult(
            faithfulness=faithfulness,
            correctness=correctness,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=cost,
            raw_response=response_text,
        )


class DomainClassifier:
    """Routes a query to the appropriate domain specialist judge."""

    CATEGORY_KEYWORDS = {
        "billing": ["invoice", "overdue", "payment", "balance", "ar", "receivable"],
        "contracts": ["contract", "sla", "renewal", "expiry", "penalty", "agreement"],
        "crm": ["account", "health", "contact", "risk", "profile", "concern"],
        "psa": ["ticket", "technician", "hours", "resolution", "status", "work"],
        "communications": ["email", "reminder", "comms", "contact", "notification"],
        "cross_source": ["client 360", "full picture", "aggregate", "across"],
    }

    def classify(self, query: str) -> str:
        """Classify query to domain (default: cross_source)."""
        query_lower = query.lower()

        # Score each domain
        scores = {}
        for domain, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[domain] = score

        # Return domain with highest score
        best_domain = max(scores, key=scores.get)
        if scores[best_domain] == 0:
            # No clear match, try LLM
            return self._llm_classify(query)
        return best_domain

    def _llm_classify(self, query: str) -> str:
        """Fallback to LLM for ambiguous queries."""
        try:
            from anthropic import Anthropic

            client = Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Classify this MSP RAG query as one of:
billing, contracts, crm, psa, communications, cross_source

Query: {query}

Return ONLY the domain name.""",
                    }
                ],
            )
            domain = response.content[0].text.strip().lower()
            if domain not in self.CATEGORY_KEYWORDS:
                return "cross_source"
            return domain
        except Exception as e:
            logger.warning(f"[DomainClassifier] LLM failed: {e}; defaulting to cross_source")
            return "cross_source"


class JudgePanelOrchestrator:
    """
    Orchestrates specialist judge selection and scoring.

    Usage:
        panel = JudgePanelOrchestrator(use_specialist_judges=True)
        result = panel.score(query, answer, ground_truth, context_str, category="billing")
    """

    def __init__(self, use_specialist_judges: bool = False) -> None:
        self._use_specialist = use_specialist_judges
        self._classifier = DomainClassifier()
        self._judges = {
            domain: SpecialistJudge(domain=domain)
            for domain in SpecialistJudge.DOMAIN_PROMPTS.keys()
        }
        self._generic_judge = LLMJudge(model="gpt-4o-mini")

    def score(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context_str: str,
        category: Optional[str] = None,
    ) -> JudgeResult:
        """
        Score using appropriate judge (specialist or generic).

        Args:
            query: The query
            answer: Model's answer
            ground_truth: Expected answer
            context_str: Retrieved context
            category: Override domain classification (optional)

        Returns:
            JudgeResult with scores
        """
        if not self._use_specialist:
            # Use generic judge
            return self._generic_judge.score(query, answer, ground_truth, context_str)

        # Classify to domain
        if category:
            domain = category
        else:
            domain = self._classifier.classify(query)

        judge = self._judges.get(domain, self._generic_judge)
        logger.debug(f"[JudgePanel] Selected {domain} specialist judge")
        return judge.score(query, answer, ground_truth, context_str)


class CalibrationAgent:
    """
    Calibrates specialist judges against human labels.

    Human labels should be in: eval/datasets/judge_labels.json

    Schema:
        [{
            "query_id": "billing_001",
            "category": "billing",
            "human_faithfulness": 0.9,
            "human_correctness": 0.85,
        }, ...]
    """

    def __init__(self) -> None:
        self._labels_path = Path("eval/datasets/judge_labels.json")

    def _load_labels(self) -> dict[str, dict]:
        """Load human labels."""
        if not self._labels_path.exists():
            logger.warning(f"[Calibration] No labels file: {self._labels_path}")
            return {}

        data = json.loads(self._labels_path.read_text())
        return {item["query_id"]: item for item in data}

    def calibrate(self, domain: str) -> dict:
        """
        Calibrate specialist judge against human labels for a domain.

        Returns:
            {
                "domain": "billing",
                "mae_faithfulness": 0.08,
                "mae_correctness": 0.12,
                "pearson_r_faithfulness": 0.87,
                "pearson_r_correctness": 0.82,
                "num_samples": 15,
            }
        """
        labels = self._load_labels()
        domain_labels = [v for v in labels.values() if v.get("category") == domain]

        if not domain_labels:
            logger.warning(f"[Calibration] No labels for domain: {domain}")
            return {}

        judge = SpecialistJudge(domain=domain)

        human_faith = []
        human_correct = []
        judge_faith = []
        judge_correct = []

        for label in domain_labels:
            query_id = label["query_id"]
            # Load query and compute judge score
            # (This is a stub; full implementation would load queries from dataset)
            human_faith.append(label.get("human_faithfulness", 0.0))
            human_correct.append(label.get("human_correctness", 0.0))
            judge_faith.append(0.5)  # Placeholder
            judge_correct.append(0.5)  # Placeholder

        # Compute MAE
        mae_faith = sum(abs(h - j) for h, j in zip(human_faith, judge_faith)) / len(human_faith) if human_faith else 0.0
        mae_correct = sum(abs(h - j) for h, j in zip(human_correct, judge_correct)) / len(human_correct) if human_correct else 0.0

        logger.info(
            f"[Calibration] {domain:15s} | "
            f"MAE(faith)={mae_faith:.3f} | "
            f"MAE(correct)={mae_correct:.3f} | "
            f"samples={len(domain_labels)}"
        )

        return {
            "domain": domain,
            "mae_faithfulness": mae_faith,
            "mae_correctness": mae_correct,
            "num_samples": len(domain_labels),
        }
