"""RAG Benchmark Evaluate Module

This module provides functionality for evaluating RAG systems using RAGAS metrics.
It supports end-to-end evaluation as well as stage-specific evaluation (retrieval/generation).

Key Components:
- evaluate: Core evaluation function
- EvaluationResult: Structured evaluation results
- MetricResult: Individual metric results
- Evaluation configurations and utilities
"""

from .evaluator import aevaluate, evaluate_e2e, evaluate_generation, evaluate_retrieval
from . import metrics_retrieval

# Import RAGAS-compatible metric classes
from .metrics_retrieval import (
    RecallAtK,
    PrecisionAtK,
    F1AtK,
    NDCGAtK,
    MRRMetric,
    MAPMetric,
)

__version__ = "0.2.0"
__all__ = [
    # Core Functions
    "aevaluate",
    "evaluate_e2e",
    "evaluate_retrieval",
    "evaluate_generation",
    # RAGAS-compatible Metric Classes
    "RecallAtK",
    "PrecisionAtK",
    "F1AtK",
    "NDCGAtK",
    "MRRMetric",
    "MAPMetric",
    # Pre-defined metric instances
]
