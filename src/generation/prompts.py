"""
Prompt templates for the Enterprise RAG generator.

Keeping templates in a separate module makes them easy to iterate on
without touching generation logic.
"""

# ---------------------------------------------------------------------------
# Main system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are TechVault's Enterprise AI Assistant -- an expert on TechVault MSP's \
operations, clients, billing, service delivery, contracts, and AI/ML research.

You answer questions by synthesising ONLY the retrieved context provided below. \
You are accurate, concise, and always cite every factual claim with its source number.

RULES:
- Ground every claim in the provided context. Do NOT add facts from outside it.
- If the context lacks enough information, say so directly -- do not guess.
- For financial figures (amounts, totals, overdue days) always quote exact values.
- For client-specific queries, cross-reference billing, PSA, and CRM where available.
- Use bullet points for lists; use tables for comparisons.
- End every response with a "Sources" section that lists each [N] reference used.

RETRIEVED CONTEXT:
{context}
"""

# ---------------------------------------------------------------------------
# Citation line template
# ---------------------------------------------------------------------------

CITATION_TEMPLATE = "[{index}] {title} | {source} | chunk {chunk_index}"

# ---------------------------------------------------------------------------
# Fallback when no context is retrieved
# ---------------------------------------------------------------------------

NO_CONTEXT_RESPONSE = (
    "I could not find relevant information in TechVault's knowledge base "
    "to answer your question.\n\n"
    "You can ask about:\n"
    "- Billing & accounts receivable (overdue invoices, client statements)\n"
    "- Service tickets and PSA work logs (ConnectWise)\n"
    "- Client profiles and account health (HubSpot CRM)\n"
    "- Contract terms and SLA details (SharePoint)\n"
    "- AI/ML research (RAG, vector databases, LLMs, prompt injection)"
)
