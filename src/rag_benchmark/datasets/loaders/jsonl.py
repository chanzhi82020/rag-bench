"""JSONL format loader for datasets"""

import json
import logging
import uuid
from typing import Iterator, List, Dict, Any, Optional, Union
from pathlib import Path

from rag_benchmark.datasets.loaders.base import BaseLoader
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord, DatasetMetadata, parse_golden_record, parse_corpus_record

logger = logging.getLogger(__name__)


class JSONLLoader(BaseLoader):
    """Loader for JSONL format datasets"""
    
    def __init__(self, dataset_path: Union[str, Path], 
                 qac_file: str = "qac.jsonl",
                 corpus_file: str = "corpus.jsonl",
                 metadata_file: str = "metadata.json"):
        """Initialize JSONL loader
        
        Args:
            dataset_path: Path to dataset directory
            qac_file: Name of QAC file
            corpus_file: Name of corpus file
            metadata_file: Name of metadata file
        """
        super().__init__(dataset_path)
        self.qac_path = self.dataset_path / qac_file
        self.corpus_path = self.dataset_path / corpus_file
        self.metadata_path = self.dataset_path / metadata_file
        
        # Load metadata if exists
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = DatasetMetadata(**json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
    
    def load_golden_records(self) -> Iterator[GoldenRecord]:
        """Load golden records from QAC JSONL file
        
        Reads the 'id' field from each record. If the 'id' field is missing,
        a UUID will be generated automatically with a warning logged.
        """
        if not self.qac_path.exists():
            logger.error(f"QAC file not found: {self.qac_path}")
            return
        
        with open(self.qac_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Check if id field is missing and log warning
                    if 'id' not in data:
                        generated_id = str(uuid.uuid4())
                        data['id'] = generated_id
                        logger.warning(
                            f"Generated ID '{generated_id}' for record at line {line_num} "
                            f"in {self.qac_path} (id field was missing)"
                        )
                    # Validate that id is non-empty string
                    elif not isinstance(data['id'], str) or not data['id'].strip():
                        logger.error(
                            f"Invalid id field at line {line_num} in {self.qac_path}: "
                            f"id must be a non-empty string"
                        )
                        continue
                    
                    yield parse_golden_record(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {self.qac_path} line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error parsing record in {self.qac_path} line {line_num}: {e}")
                    continue
    
    def load_corpus_records(self) -> Iterator[CorpusRecord]:
        """Load corpus records from corpus JSONL file"""
        if not self.corpus_path.exists():
            logger.error(f"Corpus file not found: {self.corpus_path}")
            return
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    yield parse_corpus_record(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {self.corpus_path} line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error parsing corpus record in {self.corpus_path} line {line_num}: {e}")
                    continue
    
    def save_golden_records(self, records: Iterator[GoldenRecord], 
                           file_path: Optional[Union[str, Path]] = None) -> Path:
        """Save golden records to JSONL file
        
        The 'id' field is written as the first field in each JSON record.
        
        Args:
            records: Iterator of GoldenRecord objects
            file_path: Optional custom file path
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = self.qac_path
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                # Write id as first field
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
        
        return file_path
    
    def save_corpus_records(self, records: Iterator[CorpusRecord],
                           file_path: Optional[Union[str, Path]] = None) -> Path:
        """Save corpus records to JSONL file
        
        Args:
            records: Iterator of CorpusRecord objects
            file_path: Optional custom file path
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = self.corpus_path
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                data = {
                    "reference_context": record.reference_context,
                    "reference_context_id": record.reference_context_id,
                    "title": record.title
                }
                if record.metadata:
                    data["metadata"] = record.metadata
                
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        return file_path
    
    def save_metadata(self, metadata: DatasetMetadata,
                     file_path: Optional[Union[str, Path]] = None) -> Path:
        """Save metadata to JSON file
        
        Args:
            metadata: DatasetMetadata object
            file_path: Optional custom file path
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = self.metadata_path
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.__dict__, f, indent=2, ensure_ascii=False, default=str)
        
        return file_path