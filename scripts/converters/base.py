"""Base converter for converting datasets to Golden Dataset format"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple, Union
from tqdm import tqdm
logger = logging.getLogger(__name__)

from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord, DatasetMetadata


class ConversionResult:
    """Result of dataset conversion"""
    def __init__(self):
        self.success = True
        self.converted_records = 0
        self.failed_records = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.statistics: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        """Add an error message"""
        self.success = False
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
            "success": self.success,
            "converted_records": self.converted_records,
            "failed_records": self.failed_records,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics
        }


class BaseConverter(ABC):
    """Abstract base class for dataset converters"""
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 batch_size: int = 1000):
        """Initialize converter
        
        Args:
            output_dir: Output directory for converted dataset
            batch_size: Number of records to process in each batch
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # Output file paths
        self.qac_path = self.output_dir / "qac.jsonl"
        self.corpus_path = self.output_dir / "corpus.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"
        
        # Track conversion progress
        self.converted_count = 0
        self.failed_count = 0
        self.corpus_map: Dict[str, CorpusRecord] = {}
    
    def _generate_id(self) -> str:
        """Generate a unique ID using UUID v4
        
        Returns:
            String representation of a UUID v4
        """
        return str(uuid.uuid4())
    
    @abstractmethod
    def load_source_data(self, source_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Load source data for conversion
        
        Args:
            source_path: Path to source data
            
        Returns:
            Iterator of source data records
        """
        pass
    
    @abstractmethod
    def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
        """Convert a single source record to GoldenRecord and CorpusRecords
        
        Subclasses MUST generate unique IDs for all records:
        - Generate a unique ID for each GoldenRecord using self._generate_id()
        - Generate a unique ID for each CorpusRecord if not already present
        
        Example:
            golden_id = self._generate_id()
            golden_record = GoldenRecord(
                id=golden_id,
                user_input=source_record['question'],
                reference=source_record['answer'],
                reference_contexts=[context],
                reference_context_ids=[corpus_id]
            )
            
            corpus_id = source_record.get('doc_id') or self._generate_id()
            corpus_record = CorpusRecord(
                reference_context_id=corpus_id,
                reference_context=context,
                title=title
            )
        
        Args:
            source_record: Source data record
            
        Returns:
            List of tuples (GoldenRecord, List[CorpusRecord])
        """
        pass
    
    @abstractmethod
    def create_metadata(self, source_path: Union[str, Path], num_records: int) -> DatasetMetadata:
        """Create metadata for the converted dataset
        
        Args:
            source_path: Path to source data
            num_records: Number of converted records
            
        Returns:
            DatasetMetadata object
        """
        pass
    
    def convert(self, source_path: Union[str, Path]) -> ConversionResult:
        """Convert entire dataset
        
        Args:
            source_path: Path to source dataset
            
        Returns:
            ConversionResult with conversion details
        """
        result = ConversionResult()
        
        logger.info(f"Starting conversion from {source_path} to {self.output_dir}")
        
        # Load source data
        try:
            source_iter = self.load_source_data(source_path)
        except Exception as e:
            result.add_error(f"Failed to load source data: {e}")
            return result
        
        # Process records in batches
        golden_batch = []
        corpus_batch = []

        for source_record in tqdm(source_iter, desc="Converting records", unit="records"):
            try:
                converted = self.convert_record(source_record)

                for golden_record, corpus_records in converted:
                    golden_batch.append(golden_record)
                    corpus_batch.extend(corpus_records)
                    self.converted_count += 1

                    # Write batch if it reaches batch size
                    if len(golden_batch) >= self.batch_size:
                        self._write_batch(golden_batch, corpus_batch)
                        golden_batch = []
                        corpus_batch = []

            except Exception as e:
                self.failed_count += 1
                logger.error(f"Failed to convert record: {e}")
                result.add_error(f"Record conversion failed: {e}")

        # Write remaining records
        if golden_batch:
            self._write_batch(golden_batch, corpus_batch)

        # Save metadata
        try:
            metadata = self.create_metadata(source_path, self.converted_count)
            self._save_metadata(metadata)
        except Exception as e:
            result.add_error(f"Failed to save metadata: {e}")
        
        # Update result
        result.converted_records = self.converted_count
        result.failed_records = self.failed_count
        result.add_statistic("total_source_records", self.converted_count + self.failed_count)
        result.add_statistic("success_rate", self.converted_count / (self.converted_count + self.failed_count) if (self.converted_count + self.failed_count) > 0 else 0)
        result.add_statistic("corpus_documents", len(self.corpus_map))
        
        logger.info(f"Conversion complete. Converted: {self.converted_count}, Failed: {self.failed_count}")
        
        return result

    def _write_batch(self, golden_records: List[GoldenRecord], corpus_records: List[CorpusRecord]):
        """Write a batch of records to files
        
        Args:
            golden_records: List of GoldenRecord objects
            corpus_records: List of CorpusRecord objects
        """
        # Write golden records
        with open(self.qac_path, 'a', encoding='utf-8') as f:
            for record in golden_records:
                # Ensure id is written as first field in JSON
                data = {
                    "id": record.id,
                    "user_input": record.user_input,
                    "reference": record.reference,
                    "reference_contexts": record.reference_contexts
                }
                if record.reference_context_ids:
                    data["reference_context_ids"] = record.reference_context_ids
                if record.metadata:
                    data["metadata"] = record.metadata
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # Write unique corpus records (check against global corpus_map)
        unique_corpus = {}
        for record in corpus_records:
            if record.reference_context_id not in self.corpus_map:
                unique_corpus[record.reference_context_id] = record
                self.corpus_map[record.reference_context_id] = record
        
        with open(self.corpus_path, 'a', encoding='utf-8') as f:
            for record in unique_corpus.values():
                data = {
                    "reference_context": record.reference_context,
                    "reference_context_id": record.reference_context_id,
                    "title": record.title
                }
                if record.metadata:
                    data["metadata"] = record.metadata
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def _save_metadata(self, metadata: DatasetMetadata):
        """Save metadata to file
        
        Args:
            metadata: DatasetMetadata object
        """
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.__dict__, f, indent=2, ensure_ascii=False, default=str)