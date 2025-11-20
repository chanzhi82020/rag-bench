"""RAG Benchmark Prepare Module

This module provides functionality for preparing experiment datasets from golden datasets
by filling in retrieved contexts and generated responses using RAG systems.

This module directly uses RAGAS data structures:
- SingleTurnSample: Individual experiment record
- EvaluationDataset: Collection of experiment records

Key Components:
- RAGInterface: Abstract interface for RAG systems
- prepare_experiment_dataset: Core function to prepare experiment data
- DummyRAG/SimpleRAG: Example RAG implementations
"""

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from .baseline_rag import BaselineRAG
from .dummy_rag import DummyRAG, SimpleRAG
from .prepare import prepare_experiment_dataset
from .rag_interface import (
    RAGConfig,
    RAGInterface,
    RetrievalResult,
    GenerationResult,
)

__version__ = "0.1.0"
__all__ = [
    # RAGAS types (re-exported for convenience)
    "SingleTurnSample",
    "EvaluationDataset",
    # RAG Interface
    "RAGInterface",
    "RAGConfig",
    "RetrievalResult",
    "GenerationResult",
    # Core Function
    "prepare_experiment_dataset",
    # Example RAG Implementations
    "DummyRAG",
    "SimpleRAG",
    "BaselineRAG",
]
