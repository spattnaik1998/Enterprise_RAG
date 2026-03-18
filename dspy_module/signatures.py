"""
DSPy Typed Signatures for RAG Pipeline Stages
-----------------------------------------------
Each signature defines input/output fields for a specific pipeline component.
These are optimizable by DSPy's BootstrapFewShot and MIPRO optimizers.
"""

import dspy


class RAGSignature(dspy.Signature):
    """
    Answer an enterprise MSP query using retrieved context passages.

    The answer must cite source numbers from the context and end with a
    Sources section.
    """
    query: str = dspy.InputField(
        desc="The user's question about MSP operations, billing, contracts, or CRM"
    )
    context: str = dspy.InputField(
        desc="Numbered retrieved passages [1]..[N] from the enterprise knowledge base"
    )
    answer: str = dspy.OutputField(
        desc="Grounded, factual answer citing source numbers [1]..[N]. "
             "Ends with a Sources section listing which passages were cited."
    )


class CreativeProposalSignature(dspy.Signature):
    """
    Generate a creative, high-confidence answer (FastCreativeAgent).

    Used in the Council Orchestrator. Prioritizes breadth and speculation
    within the bounds of the retrieved context.
    """
    query: str = dspy.InputField(
        desc="The enterprise RAG question"
    )
    context: str = dspy.InputField(
        desc="Numbered retrieved + reranked context passages"
    )
    creative_answer: str = dspy.OutputField(
        desc="Creative, forward-thinking answer with confidence and context citations. "
             "Ends with a Sources section."
    )


class ConservativeProposalSignature(dspy.Signature):
    """
    Generate a conservative, risk-averse answer (ConservativeCheckerAgent).

    Used in the Council Orchestrator. Emphasizes precision, cites high-confidence
    sources, and explicitly flags uncertainties.
    """
    query: str = dspy.InputField(
        desc="The enterprise RAG question"
    )
    context: str = dspy.InputField(
        desc="Numbered retrieved + reranked context passages"
    )
    conservative_answer: str = dspy.OutputField(
        desc="Conservative, precision-focused answer with explicit confidence statements. "
             "Only cites high-confidence sources. Flags uncertainties clearly. "
             "Ends with a Sources section."
    )


class PolicyVerdictSignature(dspy.Signature):
    """
    PolicyVerifier evaluates both creative and conservative proposals.

    Returns a JSON verdict deciding which proposal to accept, or escalate if both fail.
    """
    query: str = dspy.InputField(
        desc="The original query"
    )
    creative_proposal: str = dspy.InputField(
        desc="FastCreativeAgent's answer"
    )
    conservative_proposal: str = dspy.InputField(
        desc="ConservativeCheckerAgent's answer"
    )
    context: str = dspy.InputField(
        desc="Numbered context passages used by both agents"
    )
    verdict_json: str = dspy.OutputField(
        desc='JSON object with keys: "decision" ("accept_creative"|"accept_conservative"|"escalate"), '
             '"winning_agent", "dissent_summary", "policy_reasons" (list of strings). '
             'Return only valid JSON, no markdown fence.'
    )


class RerankerSignature(dspy.Signature):
    """
    Score candidate chunks by relevance to the query.

    Returns a JSON object mapping chunk IDs to scores (0-10).
    """
    query: str = dspy.InputField(
        desc="The user's question"
    )
    candidates: str = dspy.InputField(
        desc="Newline-separated candidate chunks with IDs (format: [ID] content...)"
    )
    scores_json: str = dspy.OutputField(
        desc='JSON object mapping chunk_id (string) to score (0-10 integer). '
             'Example: {"chunk_1": 9, "chunk_2": 4}. '
             'Return only valid JSON, no markdown fence.'
    )
