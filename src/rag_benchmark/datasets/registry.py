"""Dataset registry for managing available datasets"""

import logging
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Union
from typing import Type

from rag_benchmark.datasets.loaders.base import BaseLoader, ValidationResult
from rag_benchmark.datasets.loaders.jsonl import JSONLLoader
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord
from rag_benchmark.datasets.validators.quality import QualityValidator

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for managing available datasets"""

    def __init__(self):
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._loaders: Dict[str, Type[BaseLoader]] = {
            "jsonl": JSONLLoader
        }
        self._base_path = Path(__file__).parent.parent.parent.parent / "scripts" / "datasets"

        # Register built-in datasets
        self._register_builtin_datasets()

    def _register_builtin_datasets(self):
        """Register built-in datasets"""
        builtin_datasets = {
            "hotpotqa": {
                "name": "hotpotqa",
                "display_name": "HotpotQA",
                "description": "Multi-hop question answering dataset",
                "source": "https://hotpotqa.github.io/",
                "version": "1.0",
                "domain": "general",
                "language": "en",
                "loader_type": "jsonl",
                "path": self._base_path / "hotpotqa",
                "license": "MIT",
                "tags": ["multi-hop", "reasoning", "qa"],
                "default_subset": "distractor",
                "available_subsets": ["distractor", "fullwiki"]
            },
            "natural_questions": {
                "name": "natural_questions",
                "display_name": "Natural Questions",
                "description": "Real user questions from Google Search",
                "source": "https://ai.google.com/research/pubs/pub47761",
                "version": "1.0",
                "domain": "general",
                "language": "en",
                "loader_type": "jsonl",
                "path": self._base_path / "nq",
                "license": "CC BY-SA 3.0",
                "tags": ["search", "real-questions", "qa"]
            },
            "customer_service": {
                "name": "customer_service",
                "display_name": "Customer Service",
                "description": "Customer service Q&A dataset",
                "source": "private",
                "version": "1.0",
                "domain": "customer_service",
                "language": "zh",
                "loader_type": "jsonl",
                "path": self._base_path / "customer_service",
                "license": "proprietary",
                "tags": ["customer-service", "zh", "private"]
            },
            "xquad": {
                "name": "xquad",
                "display_name": "XQuAD",
                "description": "Cross-lingual Question Answering Dataset",
                "source": "https://github.com/google-deepmind/xquad",
                "version": "1.0",
                "domain": "question_answering",
                "language": "multilingual",
                "loader_type": "jsonl",
                "path": self._base_path / "xquad",
                "license": "Apache 2.0",
                "tags": ["xquad", "question-answering", "multilingual"],
                "default_subset": "zh",
                "available_subsets": ["zh", "en"]
            }
        }

        for name, info in builtin_datasets.items():
            self._datasets[name] = info

    def register_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """Register a new dataset
        
        Args:
            dataset_info: Dataset information dictionary
        """
        required_fields = ["name", "display_name", "description", "loader_type", "path"]
        for field in required_fields:
            if field not in dataset_info:
                raise ValueError(f"Missing required field: {field}")

        if dataset_info["loader_type"] not in self._loaders:
            raise ValueError(f"Unknown loader type: {dataset_info['loader_type']}")

        self._datasets[dataset_info["name"]] = dataset_info
        logger.info(f"Registered dataset: {dataset_info['name']}")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all registered datasets
        
        Returns:
            List of dataset information dictionaries
        """
        return list(self._datasets.values())

    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset information dictionary or None if not found
        """
        return self._datasets.get(name)

    def get_loader(self, name: str, subset: Optional[str] = None, **kwargs) -> Optional[BaseLoader]:
        """Get a loader for a specific dataset
        
        Args:
            name: Dataset name
            subset: Optional subset name (for datasets with multiple variants)
            **kwargs: Additional arguments for loader
            
        Returns:
            BaseLoader instance or None if not found
        """
        dataset_info = self.get_dataset_info(name)
        if not dataset_info:
            logger.error(f"Dataset not found: {name}")
            return None

        loader_type = dataset_info["loader_type"]
        loader_class = self._loaders.get(loader_type)
        if not loader_class:
            logger.error(f"Loader type not found: {loader_type}")
            return None

        dataset_path = Path(dataset_info["path"])

        # Handle subset by looking into subdirectories
        # If no subset specified, try to use default subset
        if not subset and "default_subset" in dataset_info:
            subset = dataset_info["default_subset"]
            logger.debug(f"Using default subset '{subset}' for dataset '{name}'")

        if subset:
            subset_path = dataset_path / subset
            if subset_path.exists():
                dataset_path = subset_path
                logger.info(f"Using subset '{subset}' at path: {dataset_path}")
            else:
                logger.warning(f"Subset path not found: {subset_path}, using base path")

        return loader_class(dataset_path, **kwargs)

    def is_dataset_available(self, name: str, subset: Optional[str] = None) -> bool:
        """Check if a dataset is available (path exists)
        
        Args:
            name: Dataset name
            subset: Optional subset name
            
        Returns:
            True if dataset is available, False otherwise
        """
        dataset_info = self.get_dataset_info(name)
        if not dataset_info:
            return False

        dataset_path = Path(dataset_info["path"])

        # If no subset specified, try to use default subset
        if not subset and "default_subset" in dataset_info:
            subset = dataset_info["default_subset"]

        # If subset is specified (or default exists), check subset path
        if subset:
            subset_path = dataset_path / subset
            if subset_path.exists():
                return True
            # If subset was specified but doesn't exist, check if dataset has other subsets
            if "available_subsets" in dataset_info:
                logger.info(f"Dataset '{name}' has available subsets: {dataset_info['available_subsets']}")

        # Otherwise, check if base path exists
        if dataset_path.exists():
            return True

        # Check if there are any subdirectories that might be subsets
        if dataset_path.parent.exists():
            return any(dataset_path.iterdir())

        return False

    def validate_dataset(self, name: str) -> Dict[str, Any]:
        """Validate a dataset
        
        Args:
            name: Dataset name
            
        Returns:
            Validation result dictionary
        """
        loader = self.get_loader(name)
        if not loader:
            return {
                "is_valid": False,
                "error": f"Dataset not found or cannot be loaded: {name}"
            }

        # Format validation
        format_result = loader.validate()

        # Quality validation
        golden_records = list(loader.load_golden_records())
        corpus_records = list(loader.load_corpus_records())
        quality_result = QualityValidator.validate_dataset_quality(
            golden_records, corpus_records
        )

        return {
            "dataset_name": name,
            "is_valid": format_result.is_valid and quality_result.is_valid,
            "format_validation": format_result.to_dict(),
            "quality_validation": quality_result.to_dict(),
            "record_count": len(golden_records),
            "corpus_count": len(corpus_records)
        }


# Global registry instance
DATASET_REGISTRY = DatasetRegistry()


def register_dataset(dataset_info: Dict[str, Any]) -> None:
    """Register a new dataset globally
    
    Args:
        dataset_info: Dataset information dictionary
    """
    DATASET_REGISTRY.register_dataset(dataset_info)


def list_datasets() -> List[Dict[str, Any]]:
    """List all registered datasets
    
    Returns:
        List of dataset information dictionaries
    """
    return DATASET_REGISTRY.list_datasets()


def get_dataset_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific dataset
    
    Args:
        name: Dataset name
        
    Returns:
        Dataset information dictionary or None if not found
    """
    return DATASET_REGISTRY.get_dataset_info(name)


def load_golden_dataset(
        name: str,
        subset: Optional[str] = None,
        streaming: bool = True
) -> Iterator[GoldenRecord]:
    """Load golden records from a dataset

    Args:
        name: Dataset name
        subset: Optional subset name (for datasets with multiple versions)
        streaming: Whether to stream records (True) or load all at once (False)

    Returns:
        Iterator of GoldenRecord objects
    """
    loader = DATASET_REGISTRY.get_loader(name, subset=subset)
    if not loader:
        raise ValueError(f"Dataset not found: {name}")

    if not DATASET_REGISTRY.is_dataset_available(name, subset=subset):
        # Try to find available subsets
        dataset_info = DATASET_REGISTRY.get_dataset_info(name)
        if dataset_info:
            base_path = Path(dataset_info["path"])
            if base_path.exists():
                subsets = [d.name for d in base_path.iterdir() if d.is_dir()]
                if subsets:
                    raise ValueError(
                        f"Dataset '{name}' has subsets available: {subsets}. Please specify a subset parameter.")
        raise ValueError(f"Dataset not available: {name} (path does not exist)")

    records = loader.load_golden_records()

    if streaming:
        return records
    else:
        return list(records)


def load_corpus_dataset(
        name: str,
        subset: Optional[str] = None,
        streaming: bool = True
) -> Iterator[CorpusRecord]:
    """Load corpus records from a dataset

    Args:
        name: Dataset name
        subset: Optional subset name (for datasets with multiple versions)
        streaming: Whether to stream records (True) or load all at once (False)

    Returns:
        Iterator of CorpusRecord objects
    """
    loader = DATASET_REGISTRY.get_loader(name, subset=subset)
    if not loader:
        raise ValueError(f"Dataset not found: {name}")

    if not DATASET_REGISTRY.is_dataset_available(name, subset=subset):
        raise ValueError(f"Dataset not available: {name} (path does not exist)")

    records = loader.load_corpus_records()

    if streaming:
        return records
    else:
        return list(records)


def list_golden_datasets(
        available_only: bool = False
) -> List[Dict[str, Any]]:
    """List all golden datasets

    Args:
        available_only: Whether to only show datasets that are available (path exists)

    Returns:
        List of dataset information dictionaries
    """
    datasets = DATASET_REGISTRY.list_datasets()

    if available_only:
        datasets = [
            d for d in datasets
            if DATASET_REGISTRY.is_dataset_available(d["name"])
        ]

    return datasets


def get_dataset_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a dataset

    Args:
        name: Dataset name

    Returns:
        Dataset metadata dictionary or None if not found
    """
    return DATASET_REGISTRY.get_dataset_info(name)


def validate_dataset(name: str) -> ValidationResult:
    """Validate a dataset

    Args:
        name: Dataset name

    Returns:
        ValidationResult with validation details
    """
    loader = DATASET_REGISTRY.get_loader(name)
    if not loader:
        result = ValidationResult()
        result.add_error(f"Dataset not found: {name}")
        return result

    return loader.validate()


def get_dataset_sample(name: str, n: int = 5) -> List[GoldenRecord]:
    """Get a sample of records from a dataset

    Args:
        name: Dataset name
        n: Number of samples to get

    Returns:
        List of n GoldenRecord objects
    """
    loader = DATASET_REGISTRY.get_loader(name)
    if not loader:
        raise ValueError(f"Dataset not found: {name}")

    return loader.get_sample(n)


def count_dataset_records(name: str) -> int:
    """Count the number of records in a dataset

    Args:
        name: Dataset name

    Returns:
        Number of golden records in the dataset
    """
    loader = DATASET_REGISTRY.get_loader(name)
    if not loader:
        raise ValueError(f"Dataset not found: {name}")

    return loader.count_records()


def create_custom_loader(
        dataset_path: Union[str, Path],
        loader_type: str = "jsonl",
        **kwargs
) -> BaseLoader:
    """Create a loader for a custom dataset

    Args:
        dataset_path: Path to dataset directory
        loader_type: Type of loader to use ("jsonl")
        **kwargs: Additional arguments for loader

    Returns:
        BaseLoader instance
    """
    if loader_type == "jsonl":
        return JSONLLoader(dataset_path, **kwargs)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
