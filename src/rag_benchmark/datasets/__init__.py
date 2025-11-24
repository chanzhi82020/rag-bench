"""RAG Benchmark Datasets Module

This module provides interfaces for accessing and manipulating golden datasets
for RAG (Retrieval-Augmented Generation) benchmarking.

Key Components:
- GoldenDataset: High-level interface for dataset operations with filtering and transformations
- BaseLoader: Abstract base class for data loaders
- DatasetRegistry: Registry for managing available datasets
"""

# Core interfaces
from .golden import GoldenDataset, DatasetView, load_dataset

# Schemas
from .schemas.golden import (
    GoldenRecord,
    CorpusRecord,
    DatasetMetadata,
    GoldenRecordModel,
    CorpusRecordModel,
    DatasetMetadataModel,
    parse_golden_record,
    parse_corpus_record
)

# Loaders
from .loaders.base import BaseLoader, ValidationResult
from .loaders.jsonl import JSONLLoader

# Registry
from .registry import (
    DatasetRegistry,
    DATASET_REGISTRY,
    register_dataset,
    list_datasets,
    get_dataset_info,
    load_golden_dataset,
    load_corpus_dataset,
    list_golden_datasets,
    get_dataset_metadata,
    validate_dataset,
    get_dataset_sample,
    count_dataset_records,
    create_custom_loader
)

# Validators
from .validators.format import FormatValidator
from .validators.quality import QualityValidator

__version__ = "0.1.0"
__all__ = [
    # Core interfaces
    "GoldenDataset",
    "DatasetView",
    "load_dataset",

    # Schemas
    "GoldenRecord",
    "CorpusRecord",
    "DatasetMetadata",
    "GoldenRecordModel",
    "CorpusRecordModel",
    "DatasetMetadataModel",
    "parse_golden_record",
    "parse_corpus_record",

    # Loaders
    "BaseLoader",
    "JSONLLoader",
    "ValidationResult",
    # Registry
    "DatasetRegistry",
    "DATASET_REGISTRY",
    "register_dataset",
    "list_datasets",
    "get_dataset_info",
    "load_golden_dataset",
    "load_corpus_dataset",
    "list_golden_datasets",
    "get_dataset_metadata",
    "validate_dataset",
    "get_dataset_sample",
    "count_dataset_records",
    "create_custom_loader",

    # Validators
    "FormatValidator",
    "QualityValidator",
]
