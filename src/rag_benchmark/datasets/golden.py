"""Golden Dataset interface for unified dataset access and manipulation"""

from typing import Iterator, List, Dict, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import logging

from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord
from rag_benchmark.datasets.loaders.base import BaseLoader
from rag_benchmark.datasets.registry import DATASET_REGISTRY

logger = logging.getLogger(__name__)


class SelectedSamplesLoader(BaseLoader):
    """Loader for a subset of golden records selected by IDs"""
    
    def __init__(self, golden_records: List[GoldenRecord], parent_loader: BaseLoader):
        self.golden_records = golden_records
        self.parent_loader = parent_loader
        # Extract corpus IDs from golden records
        self.corpus_ids = set()
        for record in golden_records:
            if record.reference_context_ids:
                self.corpus_ids.update(record.reference_context_ids)
    
    def load_golden_records(self) -> Iterator[GoldenRecord]:
        return iter(self.golden_records)
    
    def load_corpus_records(self) -> Iterator[CorpusRecord]:
        # Only return corpus records referenced by golden records
        for corpus_record in self.parent_loader.load_corpus_records():
            if corpus_record.reference_context_id in self.corpus_ids:
                yield corpus_record
    
    def count_records(self) -> int:
        return len(self.golden_records)


class SubsetLoader(BaseLoader):
    """Loader for a random subset of golden records"""
    
    def __init__(self, golden_records: List[GoldenRecord], parent_loader: BaseLoader):
        self.golden_records = golden_records
        self.parent_loader = parent_loader
        # Extract corpus IDs from golden records
        self.corpus_ids = set()
        for record in golden_records:
            if record.reference_context_ids:
                self.corpus_ids.update(record.reference_context_ids)
    
    def load_golden_records(self) -> Iterator[GoldenRecord]:
        return iter(self.golden_records)
    
    def load_corpus_records(self) -> Iterator[CorpusRecord]:
        # Only return corpus records referenced by golden records
        for corpus_record in self.parent_loader.load_corpus_records():
            if corpus_record.reference_context_id in self.corpus_ids:
                yield corpus_record
    
    def count_records(self) -> int:
        return len(self.golden_records)


class DatasetView:
    """A view of the dataset with applied filters and transformations"""
    
    def __init__(self, 
                 dataset: 'GoldenDataset',
                 filters: Optional[List[Callable[[GoldenRecord], bool]]] = None,
                 transform: Optional[Callable[[GoldenRecord], GoldenRecord]] = None):
        self.dataset = dataset
        self.filters = filters or []
        self.transform = transform
        
    def __iter__(self) -> Iterator[GoldenRecord]:
        for record in self.dataset.loader.load_golden_records():
            # Apply filters
            if all(filter_fn(record) for filter_fn in self.filters):
                # Apply transformation
                if self.transform:
                    record = self.transform(record)
                yield record
    
    def filter(self, predicate: Callable[[GoldenRecord], bool]) -> 'DatasetView':
        """Apply a filter to this view"""
        return DatasetView(
            dataset=self.dataset,
            filters=self.filters + [predicate],
            transform=self.transform
        )
    
    def map(self, func: Callable[[GoldenRecord], GoldenRecord]) -> 'DatasetView':
        """Apply a transformation to records in this view"""
        if self.transform:
            # Chain transformations
            old_transform = self.transform
            def combined_transform(record):
                return func(old_transform(record))
            transform = combined_transform
        else:
            transform = func
            
        return DatasetView(
            dataset=self.dataset,
            filters=self.filters,
            transform=transform
        )
    
    def collect(self, limit: Optional[int] = None) -> List[GoldenRecord]:
        """Collect records from this view into a list"""
        records = []
        for i, record in enumerate(self):
            if limit and i >= limit:
                break
            records.append(record)
        return records
    
    def first(self) -> Optional[GoldenRecord]:
        """Get the first record in this view"""
        for record in self:
            return record
        return None
    
    def count(self) -> int:
        """Count records in this view"""
        return sum(1 for _ in self)


class GoldenDataset:
    """Unified interface for golden datasets with high-level operations"""
    
    def __init__(self, 
                 name: str,
                 subset: Optional[str] = None,
                 loader: Optional[BaseLoader] = None):
        """Initialize golden dataset
        
        Args:
            name: Dataset name in registry
            subset: Optional subset name
            loader: Optional pre-configured loader (for testing)
        """
        self.name = name
        self.subset = subset
        self._loader = loader
        
    @property
    def loader(self) -> BaseLoader:
        """Get the underlying loader"""
        if self._loader is None:
            self._loader = DATASET_REGISTRY.get_loader(self.name, self.subset)
            if self._loader is None:
                raise ValueError(f"Dataset not found: {self.name}")
        return self._loader
    
    def __iter__(self) -> Iterator[GoldenRecord]:
        """Iterate over golden records"""
        return self.loader.load_golden_records()
    
    def iter_corpus(self) -> Iterator[CorpusRecord]:
        """Iterate over corpus records"""
        return self.loader.load_corpus_records()
    
    # High-level operations
    
    def view(self) -> DatasetView:
        """Create a new view of this dataset"""
        return DatasetView(dataset=self)
    
    def filter(self, predicate: Callable[[GoldenRecord], bool]) -> DatasetView:
        """Filter records using a predicate function"""
        return self.view().filter(predicate)
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[GoldenRecord]:
        """Sample n records from the dataset"""
        import random
        
        if seed is not None:
            random.seed(seed)
        
        records = list(self)
        
        if n >= len(records):
            return records
        
        return random.sample(records, n)
    
    def get_record_ids(self) -> List[str]:
        """Get all record IDs in the dataset
        
        Returns:
            List of all record IDs
        """
        return [record.id for record in self]
    
    def select_by_ids(self, record_ids: List[str]) -> 'GoldenDataset':
        """Select specific records by their IDs
        
        This method creates a new GoldenDataset with only the selected records,
        and filters corpus records to only include those referenced by the selected records.
        
        Args:
            record_ids: List of record IDs
            
        Returns:
            New GoldenDataset with selected records
            
        Raises:
            ValueError: If any record ID is invalid
        """
        # Load all records and create ID -> record mapping
        all_records = list(self)
        id_to_record = {record.id: record for record in all_records}
        
        # Validate IDs
        invalid_ids = [rid for rid in record_ids if rid not in id_to_record]
        if invalid_ids:
            raise ValueError(f"Invalid record IDs: {invalid_ids}")
        
        # Select records (preserving order and allowing duplicates)
        selected_records = [id_to_record[rid] for rid in record_ids]
        
        # Create new dataset with selected samples
        selected_loader = SelectedSamplesLoader(selected_records, self.loader)
        subset_name = f"{self.name}_selected_{len(record_ids)}"
        if self.subset:
            subset_name = f"{self.name}_{self.subset}_selected_{len(record_ids)}"
        
        return GoldenDataset(
            name=subset_name,
            subset=None,
            loader=selected_loader
        )
    
    def select_random(self, n: int, seed: Optional[int] = None) -> 'GoldenDataset':
        """Create a subset dataset with n records
        
        This returns a new GoldenDataset object with a custom loader,
        making it compatible with all dataset operations.
        
        Args:
            n: Number of records to include
            seed: Optional random seed for reproducibility
            
        Returns:
            New GoldenDataset with subset of records
        """
        # Get subset records
        records = self.sample(n, seed)
        
        # Create new dataset with subset loader
        subset_loader = SubsetLoader(records, self.loader)
        subset_name = f"{self.name}_subset_{n}"
        if self.subset:
            subset_name = f"{self.name}_{self.subset}_subset_{n}"
        
        return GoldenDataset(
            name=subset_name,
            subset=None,
            loader=subset_loader
        )
    
    def slice(self, start: int, end: Optional[int] = None, step: int = 1) -> List[GoldenRecord]:
        """Slice the dataset"""
        return list(self)[start:end:step]
    
    def head(self, n: int = 5) -> List[GoldenRecord]:
        """Get the first n records"""
        return self.slice(0, n)
    
    def tail(self, n: int = 5) -> List[GoldenRecord]:
        """Get the last n records"""
        records = list(self)
        return records[-n:] if n < len(records) else records
    
    def count(self) -> int:
        """Count the number of records"""
        return self.loader.count_records()
    
    def paginate(self, page: int, page_size: int) -> Tuple[List[GoldenRecord], int]:
        """Get paginated records and total count
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of records per page
            
        Returns:
            Tuple of (paginated records, total count)
            
        Raises:
            ValueError: If page < 1 or page_size < 1
        """
        if page < 1:
            raise ValueError("Page number must be >= 1")
        if page_size < 1:
            raise ValueError("Page size must be >= 1")
        
        # Get all records and total count
        all_records = list(self)
        total_count = len(all_records)
        
        # Calculate start and end indices
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Return paginated records
        paginated_records = all_records[start_idx:end_idx]
        
        return paginated_records, total_count
    
    def search(self, query: str, case_sensitive: bool = False) -> 'GoldenDataset':
        """Search samples by user_input text
        
        Args:
            query: Search query string
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            New GoldenDataset with filtered records
        """
        # Prepare query for comparison
        search_query = query if case_sensitive else query.lower()
        
        # Filter records
        filtered_records = []
        for record in self:
            user_input = record.user_input if case_sensitive else record.user_input.lower()
            if search_query in user_input:
                filtered_records.append(record)
        
        # Create new dataset with filtered records
        filtered_loader = SubsetLoader(filtered_records, self.loader)
        subset_name = f"{self.name}_search"
        if self.subset:
            subset_name = f"{self.name}_{self.subset}_search"
        
        return GoldenDataset(
            name=subset_name,
            subset=None,
            loader=filtered_loader
        )
    
    def get_corpus_by_ids(self, doc_ids: List[str]) -> List[CorpusRecord]:
        """Get specific corpus documents by IDs
        
        Args:
            doc_ids: List of document IDs to retrieve
            
        Returns:
            List of CorpusRecord objects matching the IDs
        """
        # Convert to set for faster lookup
        doc_ids_set = set(doc_ids)
        
        # Filter corpus records
        matching_records = []
        for corpus_record in self.iter_corpus():
            if corpus_record.reference_context_id in doc_ids_set:
                matching_records.append(corpus_record)
        
        return matching_records
    
    def stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            "dataset_name": self.name,
            "subset": self.subset,
            "record_count": self.count(),
        }
        
        # Analyze text lengths
        input_lengths = []
        reference_lengths = []
        context_counts = []
        
        for record in self:
            input_lengths.append(len(record.user_input))
            reference_lengths.append(len(record.reference))
            context_counts.append(len(record.reference_contexts))
        
        if input_lengths:
            stats.update({
                "avg_input_length": sum(input_lengths) / len(input_lengths),
                "min_input_length": min(input_lengths),
                "max_input_length": max(input_lengths),
                "avg_reference_length": sum(reference_lengths) / len(reference_lengths),
                "min_reference_length": min(reference_lengths),
                "max_reference_length": max(reference_lengths),
                "avg_contexts_per_record": sum(context_counts) / len(context_counts),
                "min_contexts_per_record": min(context_counts),
                "max_contexts_per_record": max(context_counts),
            })
        
        # Add corpus statistics
        corpus_records = list(self.iter_corpus())
        stats["corpus_count"] = len(corpus_records)
        
        if corpus_records:
            corpus_lengths = [len(record.reference_context) for record in corpus_records]
            stats.update({
                "avg_corpus_length": sum(corpus_lengths) / len(corpus_lengths),
                "min_corpus_length": min(corpus_lengths),
                "max_corpus_length": max(corpus_lengths),
            })
        
        return stats
    
    def validate(self, 
                 validate_format: bool = True,
                 validate_quality: bool = True) -> Dict[str, Any]:
        """Validate the dataset"""
        from rag_benchmark.datasets.validators.quality import QualityValidator
        
        result = {
            "dataset_name": self.name,
            "subset": self.subset,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Format validation using loader
        if validate_format:
            format_result = self.loader.validate()
            result["format_validation"] = format_result.to_dict()
            if not format_result.is_valid:
                result["is_valid"] = False
                result["errors"].extend(format_result.errors)
            result["warnings"].extend(format_result.warnings)
            result["statistics"].update(format_result.statistics)
        
        # Quality validation
        if validate_quality:
            golden_records = list(self)
            corpus_records = list(self.iter_corpus())
            quality_result = QualityValidator.validate_dataset_quality(
                golden_records, corpus_records
            )
            result["quality_validation"] = quality_result.to_dict()
            if not quality_result.is_valid:
                result["is_valid"] = False
            result["errors"].extend(quality_result.errors)
            result["warnings"].extend(quality_result.warnings)
            result["statistics"].update(quality_result.statistics)
        
        return result
    
    def export(self, 
               output_path: Union[str, Path],
               format_type: str = "jsonl",
               include_corpus: bool = True,
               limit: Optional[int] = None) -> None:
        """Export dataset to file"""
        from dataclasses import asdict
        import json
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format_type == "jsonl":
            # Export golden records
            with open(output_path / "qac.jsonl", 'w', encoding='utf-8') as f:
                for i, record in enumerate(self):
                    if limit and i >= limit:
                        break
                    data = asdict(record)
                    # Remove empty optional fields
                    if not data.get("reference_context_ids"):
                        data.pop("reference_context_ids", None)
                    if not data.get("metadata"):
                        data.pop("metadata", None)
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # Export corpus records
            if include_corpus:
                corpus_ids = set()
                for record in self:
                    if record.reference_context_ids:
                        corpus_ids.update(record.reference_context_ids)
                
                with open(output_path / "corpus.jsonl", 'w', encoding='utf-8') as f:
                    for corpus_record in self.iter_corpus():
                        if corpus_record.reference_context_id in corpus_ids:
                            data = asdict(corpus_record)
                            if not data.get("metadata"):
                                data.pop("metadata", None)
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def __len__(self) -> int:
        return self.count()
    
    def __getitem__(self, key) -> Union[GoldenRecord, List[GoldenRecord]]:
        """Support indexing and slicing"""
        if isinstance(key, int):
            return self.slice(key, key + 1)[0]
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.count()
            step = key.step if key.step is not None else 1
            return self.slice(start, stop, step)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")
    
    def __repr__(self) -> str:
        return f"GoldenDataset(name='{self.name}', subset='{self.subset}', records={self.count()})"


def load_dataset(name: str, subset: Optional[str] = None) -> GoldenDataset:
    """Load a dataset by name from registry
    
    Args:
        name: Dataset name
        subset: Optional subset name
        
    Returns:
        GoldenDataset instance
    """
    return GoldenDataset(name=name, subset=subset)