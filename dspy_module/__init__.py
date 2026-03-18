"""
DSPy Modular Training Framework for Enterprise RAG
----------------------------------------------------
Provides automated prompt optimization and few-shot demonstration compilation
for the 6-stage RAG pipeline using the 80-query evaluation dataset.

Exports:
  - Signatures: RAGSignature, CreativeProposalSignature, ConservativeProposalSignature, etc.
  - Modules: DSPyRetrieverAdapter, DSPyRAGModule, DSPyRerankerModule, DSPyCouncilModule
  - Dataset: MSPDataset (train/dev/gold split)
  - Metrics: keyword_recall_metric, source_type_metric, faithfulness_metric, correctness_metric, rag_composite_metric
  - CLI: typer-based trainer with optimize, evaluate, compare commands
"""

from dspy_module.signatures import (
    RAGSignature,
    CreativeProposalSignature,
    ConservativeProposalSignature,
    PolicyVerdictSignature,
    RerankerSignature,
)
from dspy_module.modules import (
    DSPyRetrieverAdapter,
    DSPyRAGModule,
    DSPyRerankerModule,
    DSPyCouncilModule,
)
from dspy_module.dataset import MSPDataset
from dspy_module.metrics import (
    keyword_recall_metric,
    source_type_metric,
    faithfulness_metric,
    correctness_metric,
    rag_composite_metric,
    cheap_metric,
)

__all__ = [
    "RAGSignature",
    "CreativeProposalSignature",
    "ConservativeProposalSignature",
    "PolicyVerdictSignature",
    "RerankerSignature",
    "DSPyRetrieverAdapter",
    "DSPyRAGModule",
    "DSPyRerankerModule",
    "DSPyCouncilModule",
    "MSPDataset",
    "keyword_recall_metric",
    "source_type_metric",
    "faithfulness_metric",
    "correctness_metric",
    "rag_composite_metric",
    "cheap_metric",
]
