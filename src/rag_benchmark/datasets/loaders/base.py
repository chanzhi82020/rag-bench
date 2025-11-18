"""Base loader interface for dataset loading"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Dict, Any, Optional, Union
from pathlib import Path
import json

from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord, DatasetMetadata


class ValidationResult:
    """Result of dataset validation"""
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.statistics: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        """Add an error message"""
        self.is_valid = False
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)
    
    def add_statistic(self, key: str, value: Any):
        """Add a statistic"""
        self.statistics[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics
        }


class BaseLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, dataset_path: Union[str, Path]):
        self.dataset_path = Path(dataset_path)
        self._metadata: Optional[DatasetMetadata] = None
    
    @abstractmethod
    def load_golden_records(self) -> Iterator[GoldenRecord]:
        """Load golden records from the dataset
        
        Returns:
            Iterator of GoldenRecord objects
        """
        pass
    
    @abstractmethod
    def load_corpus_records(self) -> Iterator[CorpusRecord]:
        """Load corpus records from the dataset
        
        Returns:
            Iterator of CorpusRecord objects
        """
        pass
    
    def get_metadata(self) -> Optional[DatasetMetadata]:
        """Get dataset metadata"""
        return self._metadata
    
    def count_records(self) -> int:
        """Count the number of golden records in the dataset
        
        Returns:
            Number of records
        """
        count = 0
        for _ in self.load_golden_records():
            count += 1
        return count
    
    def validate(self) -> ValidationResult:
        """Validate the dataset
        
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            result.add_error(f"Dataset path does not exist: {self.dataset_path}")
            return result
        
        # Validate golden records
        golden_count = 0
        ctx_count_total = 0
        ctx_count_min = float('inf')
        ctx_count_max = 0
        
        for record in self.load_golden_records():
            golden_count += 1
            ctx_count = len(record.reference_contexts)
            ctx_count_total += ctx_count
            ctx_count_min = min(ctx_count_min, ctx_count)
            ctx_count_max = max(ctx_count_max, ctx_count)
        
        if golden_count == 0:
            result.add_error("No golden records found")
        else:
            result.add_statistic("golden_record_count", golden_count)
            result.add_statistic("avg_contexts_per_record", ctx_count_total / golden_count)
            result.add_statistic("min_contexts_per_record", ctx_count_min)
            result.add_statistic("max_contexts_per_record", ctx_count_max)
        
        # Validate corpus records
        corpus_count = 0
        corpus_ids = set()
        
        for record in self.load_corpus_records():
            corpus_count += 1
            if record.reference_context_id in corpus_ids:
                result.add_error(f"Duplicate corpus ID: {record.reference_context_id}")
            corpus_ids.add(record.reference_context_id)
        
        if corpus_count == 0:
            result.add_error("No corpus records found")
        else:
            result.add_statistic("corpus_record_count", corpus_count)
            result.add_statistic("unique_corpus_ids", len(corpus_ids))
        
        return result
    
    def get_sample(self, n: int = 5) -> List[GoldenRecord]:
        """Get a sample of n golden records
        
        Args:
            n: Number of samples to get
            
        Returns:
            List of n GoldenRecord objects
        """
        samples = []
        for i, record in enumerate(self.load_golden_records()):
            if i >= n:
                break
            samples.append(record)
        return samples