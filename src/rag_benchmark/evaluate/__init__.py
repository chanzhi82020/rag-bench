"""RAG Benchmark Evaluate Module

This module provides functionality for evaluating RAG systems using RAGAS metrics.
It supports end-to-end evaluation as well as stage-specific evaluation (retrieval/generation).

Key Components:
- evaluate: Core evaluation function
- EvaluationResult: Structured evaluation results
- MetricResult: Individual metric results
- Evaluation configurations and utilities

Example:
    >>> from ragas.dataset_schema import EvaluationDataset
    >>> from rag_benchmark.evaluate import evaluate
    >>> from ragas.metrics import faithfulness, answer_relevancy
    >>>
    >>> # Evaluate RAG system
    >>> result = evaluate(
    ...     dataset=evaluation_dataset,
    ...     metrics=[faithfulness, answer_relevancy],
    ...     name="my_rag_system"
    ... )
    >>>
    >>> # View results
    >>> print(result.summary())
    >>> result.save("results/evaluation.json")
"""

from .evaluator import evaluate, evaluate_e2e, evaluate_generation, evaluate_retrieval
from .results import EvaluationResult, MetricResult

__version__ = "0.1.0"
__all__ = [
    # Core Functions
    "evaluate",
    "evaluate_e2e",
    "evaluate_retrieval",
    "evaluate_generation",
    # Result Classes
    "EvaluationResult",
    "MetricResult",
]
