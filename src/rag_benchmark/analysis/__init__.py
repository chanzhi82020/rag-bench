"""RAG Benchmark Analysis Module

This module provides functionality for analyzing and comparing RAG evaluation results.

Key Components:
- compare: Compare multiple evaluation results
- visualize: Generate visualizations for evaluation results (requires matplotlib)
- ResultComparison: Structured comparison results

Example:
    >>> from rag_benchmark.analysis import compare_results
    >>> 
    >>> # Compare two evaluation results
    >>> comparison = compare_results([result1, result2], names=["Model A", "Model B"])
    >>> print(comparison.summary())
    >>> 
    >>> # Visualize metrics (requires matplotlib)
    >>> from rag_benchmark.analysis import plot_metrics
    >>> plot_metrics(comparison, metrics=["faithfulness", "answer_relevancy"])
"""

from .compare import compare_results, ResultComparison

__version__ = "0.2.0"
__all__ = [
    "compare_results",
    "ResultComparison",
]

# Optional visualization imports
try:
    from .visualize import plot_metrics, plot_comparison, plot_distribution
    __all__.extend(["plot_metrics", "plot_comparison", "plot_distribution"])
except ImportError:
    # matplotlib not installed
    pass
