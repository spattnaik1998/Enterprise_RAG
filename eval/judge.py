"""
LLM-as-Judge for RAG evaluation
---------------------------------
Uses GPT-4o-mini to score each (query, answer, ground_truth, context) tuple
on two dimensions:

  faithfulness  -- are all claims in the answer supported by the retrieved context?
  correctness   -- does the answer correctly address the question vs. ground truth?

Both scores are 0.0 - 1.0 floats returned as JSON in a single API call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger


# ---------------------------------------------------------------------------
# Pricing for judge model token cost estimation
# ---------------------------------------------------------------------------

_JUDGE_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o":      (2.500, 10.000),
}

_JUDGE_PROMPT = """\
You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
You will score a model's answer on two dimensions, each from 0.0 to 1.0.

QUESTION:
{query}

GROUND TRUTH ANSWER:
{ground_truth}

RETRIEVED CONTEXT (citations used by the model):
{context_str}

MODEL ANSWER:
{answer}

Evaluate the model answer on:

1. FAITHFULNESS (0.0-1.0): Are all claims in the model answer supported by the retrieved context?
   - 1.0 = every claim is grounded in the context
   - 0.5 = some claims are grounded, some are hallucinated or unsupported
   - 0.0 = the answer contradicts or ignores the context entirely

2. CORRECTNESS (0.0-1.0): Does the model answer correctly address the question compared to the ground truth?
   - 1.0 = answer is completely correct and matches ground truth
   - 0.5 = partially correct, some key facts missing or wrong
   - 0.0 = answer is wrong, irrelevant, or refuses to answer

Return ONLY a JSON object with exactly two keys, no explanation:
{{"faithfulness": <float>, "correctness": <float>}}
"""


@dataclass
class JudgeResult:
    """Scores and metadata from a single LLM judge call."""

    faithfulness: float
    correctness: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    raw_response: str


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _JUDGE_PRICING.get(model, (0.150, 0.600))
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


class LLMJudge:
    """
    Scores a single (query, answer, ground_truth, context) tuple using
    GPT-4o-mini (or another OpenAI model).

    Uses temperature=0 for deterministic scoring.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self.model = model
        self._client = OpenAI()

    def score(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context_str: str,
    ) -> JudgeResult:
        """
        Run a single judge call.

        Args:
            query:        The user question that was posed.
            answer:       The model's generated answer.
            ground_truth: The reference answer from the eval dataset.
            context_str:  Citation metadata string (title + source + score).
                          Note: chunk text is not available in QueryResult.citations,
                          so we pass formatted citation metadata instead.

        Returns:
            JudgeResult with faithfulness, correctness, token counts, and cost.
        """
        prompt = _JUDGE_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            context_str=context_str or "(no context retrieved)",
            answer=answer or "(no answer generated)",
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
            )
            raw = response.choices[0].message.content or ""
            usage = response.usage
            faithfulness, correctness = self._parse_scores(raw)
            cost = _cost_usd(self.model, usage.prompt_tokens, usage.completion_tokens)

            logger.debug(
                f"[Judge] faith={faithfulness:.2f} correct={correctness:.2f} "
                f"tokens={usage.prompt_tokens}+{usage.completion_tokens} "
                f"cost=${cost:.5f}"
            )

            return JudgeResult(
                faithfulness=faithfulness,
                correctness=correctness,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=cost,
                raw_response=raw,
            )

        except Exception as exc:
            logger.warning(f"[Judge] API error: {exc} -- returning neutral (0.5, 0.5)")
            return JudgeResult(
                faithfulness=0.5,
                correctness=0.5,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                raw_response=f"ERROR: {exc}",
            )

    def _parse_scores(self, raw: str) -> tuple[float, float]:
        """
        Parse {"faithfulness": float, "correctness": float} from the judge response.
        Falls back to (0.5, 0.5) on any parse error.
        """
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            data = json.loads(text)
            faith = float(data.get("faithfulness", 0.5))
            correct = float(data.get("correctness", 0.5))
            # Clamp to [0, 1]
            faith = max(0.0, min(1.0, faith))
            correct = max(0.0, min(1.0, correct))
            return faith, correct

        except Exception as exc:
            logger.warning(f"[Judge] Score parse error: {exc!r} | raw={raw!r}")
            return 0.5, 0.5
