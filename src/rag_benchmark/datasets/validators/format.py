"""Format validation utilities for datasets"""

import re
from typing import List, Dict, Any, Set
from pathlib import Path

from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord
from rag_benchmark.datasets.loaders.base import ValidationResult


class FormatValidator:
    """Validates dataset format and structure"""
    
    @staticmethod
    def validate_golden_record(record: GoldenRecord, record_id: str = None) -> List[str]:
        """Validate a single golden record format
        
        Args:
            record: GoldenRecord to validate
            record_id: Optional record identifier for error messages
            
        Returns:
            List of error messages
        """
        errors = []
        prefix = f"Record {record_id}" if record_id else "Record"
        
        # Validate user_input
        if not record.user_input or not record.user_input.strip():
            errors.append(f"{prefix}: user_input is empty")
        elif len(record.user_input) > 5000:
            errors.append(f"{prefix}: user_input too long ({len(record.user_input)} > 5000)")
        
        # Validate reference
        if not record.reference or not record.reference.strip():
            errors.append(f"{prefix}: reference is empty")
        elif len(record.reference) > 10000:
            errors.append(f"{prefix}: reference too long ({len(record.reference)} > 10000)")
        
        # Validate reference_contexts
        if not record.reference_contexts:
            errors.append(f"{prefix}: reference_contexts is empty")
        else:
            for i, ctx in enumerate(record.reference_contexts):
                if not ctx or not ctx.strip():
                    errors.append(f"{prefix}: reference_contexts[{i}] is empty")
                elif len(ctx) > 10000:
                    errors.append(f"{prefix}: reference_contexts[{i}] too long ({len(ctx)} > 10000)")
        
        # Validate reference_context_ids if present
        if record.reference_context_ids is not None:
            if len(record.reference_context_ids) != len(record.reference_contexts):
                errors.append(
                    f"{prefix}: mismatch between reference_context_ids "
                    f"({len(record.reference_context_ids)}) and reference_contexts "
                    f"({len(record.reference_contexts)})"
                )
            else:
                # Check for duplicates
                id_set = set(record.reference_context_ids)
                if len(id_set) != len(record.reference_context_ids):
                    duplicates = [x for x in record.reference_context_ids if record.reference_context_ids.count(x) > 1]
                    errors.append(f"{prefix}: duplicate reference_context_ids: {duplicates}")
        
        return errors
    
    @staticmethod
    def validate_corpus_record(record: CorpusRecord, record_id: str = None) -> List[str]:
        """Validate a single corpus record format
        
        Args:
            record: CorpusRecord to validate
            record_id: Optional record identifier for error messages
            
        Returns:
            List of error messages
        """
        errors = []
        prefix = f"Corpus {record_id}" if record_id else "Corpus"
        
        # Validate reference_context
        if not record.reference_context or not record.reference_context.strip():
            errors.append(f"{prefix}: reference_context is empty")
        elif len(record.reference_context) > 50000:
            errors.append(f"{prefix}: reference_context too long ({len(record.reference_context)} > 50000)")
        
        # Validate reference_context_id
        if not record.reference_context_id or not record.reference_context_id.strip():
            errors.append(f"{prefix}: reference_context_id is empty")
        else:
            # Check for invalid characters
            if not re.match(r'^[a-zA-Z0-9_-]+$', record.reference_context_id):
                errors.append(f"{prefix}: reference_context_id contains invalid characters")
        
        # Validate title
        if not record.title or not record.title.strip():
            errors.append(f"{prefix}: title is empty")
        elif len(record.title) > 1000:
            errors.append(f"{prefix}: title too long ({len(record.title)} > 1000)")
        
        return errors
    
    @staticmethod
    def validate_dataset_consistency(
        golden_records: List[GoldenRecord],
        corpus_records: List[CorpusRecord]
    ) -> List[str]:
        """Validate consistency between golden and corpus records
        
        Args:
            golden_records: List of GoldenRecord objects
            corpus_records: List of CorpusRecord objects
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Create corpus ID lookup
        corpus_id_set = {record.reference_context_id for record in corpus_records}
        corpus_id_to_record = {record.reference_context_id: record for record in corpus_records}
        
        # Check all referenced context IDs exist in corpus
        for i, golden in enumerate(golden_records):
            if golden.reference_context_ids:
                for ctx_id in golden.reference_context_ids:
                    if ctx_id not in corpus_id_set:
                        errors.append(
                            f"Golden record {i}: reference_context_id '{ctx_id}' not found in corpus"
                        )
        
        # Check for orphan corpus records (not referenced by any golden record)
        referenced_ids = set()
        for golden in golden_records:
            if golden.reference_context_ids:
                referenced_ids.update(golden.reference_context_ids)
        
        orphan_ids = corpus_id_set - referenced_ids
        if orphan_ids:
            errors.append(f"Found {len(orphan_ids)} orphan corpus records not referenced: {list(orphan_ids)[:5]}...")
        
        return errors
    
    @staticmethod
    def validate_file_structure(dataset_path: Path) -> List[str]:
        """Validate dataset file structure
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check directory exists
        if not dataset_path.exists():
            errors.append(f"Dataset directory does not exist: {dataset_path}")
            return errors
        
        if not dataset_path.is_dir():
            errors.append(f"Dataset path is not a directory: {dataset_path}")
            return errors
        
        # Check required files
        required_files = ["qac.jsonl", "corpus.jsonl"]
        for file_name in required_files:
            file_path = dataset_path / file_name
            if not file_path.exists():
                errors.append(f"Required file missing: {file_path}")
            elif file_path.stat().st_size == 0:
                errors.append(f"Required file is empty: {file_path}")
        
        # Check optional files
        optional_files = ["metadata.json"]
        for file_name in optional_files:
            file_path = dataset_path / file_name
            if file_path.exists() and file_path.stat().st_size == 0:
                errors.append(f"Optional file is empty: {file_path}")
        
        return errors