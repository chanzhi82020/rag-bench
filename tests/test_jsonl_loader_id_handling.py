"""Tests for JSONLLoader ID handling"""

import json
import tempfile
import uuid
from pathlib import Path
import pytest

from rag_benchmark.datasets.loaders.jsonl import JSONLLoader
from rag_benchmark.datasets.schemas.golden import GoldenRecord


def test_load_golden_records_with_id():
    """Test loading records that have id field"""
    with tempfile.TemporaryDirectory() as tmpdir:
        qac_path = Path(tmpdir) / "qac.jsonl"
        
        # Create test data with id
        test_id = str(uuid.uuid4())
        test_data = {
            "id": test_id,
            "user_input": "What is RAG?",
            "reference": "RAG stands for Retrieval Augmented Generation",
            "reference_contexts": ["Context 1", "Context 2"]
        }
        
        with open(qac_path, 'w') as f:
            f.write(json.dumps(test_data) + '\n')
        
        # Load records
        loader = JSONLLoader(tmpdir)
        records = list(loader.load_golden_records())
        
        assert len(records) == 1
        assert records[0].id == test_id
        assert records[0].user_input == "What is RAG?"


def test_load_golden_records_without_id_generates_uuid(caplog):
    """Test loading records without id field generates UUID and logs warning"""
    with tempfile.TemporaryDirectory() as tmpdir:
        qac_path = Path(tmpdir) / "qac.jsonl"
        
        # Create test data without id
        test_data = {
            "user_input": "What is RAG?",
            "reference": "RAG stands for Retrieval Augmented Generation",
            "reference_contexts": ["Context 1", "Context 2"]
        }
        
        with open(qac_path, 'w') as f:
            f.write(json.dumps(test_data) + '\n')
        
        # Load records
        loader = JSONLLoader(tmpdir)
        records = list(loader.load_golden_records())
        
        assert len(records) == 1
        assert records[0].id is not None
        assert len(records[0].id) == 36  # UUID format
        
        # Check warning was logged
        assert "Generated ID" in caplog.text
        assert "id field was missing" in caplog.text


def test_load_golden_records_with_empty_id_skips_record(caplog):
    """Test loading records with empty id field skips the record"""
    with tempfile.TemporaryDirectory() as tmpdir:
        qac_path = Path(tmpdir) / "qac.jsonl"
        
        # Create test data with empty id
        test_data = {
            "id": "",
            "user_input": "What is RAG?",
            "reference": "RAG stands for Retrieval Augmented Generation",
            "reference_contexts": ["Context 1", "Context 2"]
        }
        
        with open(qac_path, 'w') as f:
            f.write(json.dumps(test_data) + '\n')
        
        # Load records
        loader = JSONLLoader(tmpdir)
        records = list(loader.load_golden_records())
        
        assert len(records) == 0
        assert "Invalid id field" in caplog.text
        assert "id must be a non-empty string" in caplog.text


def test_save_golden_records_writes_id_first():
    """Test saving records writes id as first field"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_id = str(uuid.uuid4())
        record = GoldenRecord(
            id=test_id,
            user_input="What is RAG?",
            reference="RAG stands for Retrieval Augmented Generation",
            reference_contexts=["Context 1", "Context 2"]
        )
        
        # Save records
        loader = JSONLLoader(tmpdir)
        saved_path = loader.save_golden_records(iter([record]))
        
        # Read back and check id is first field
        with open(saved_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            
            # Check id exists and matches
            assert "id" in data
            assert data["id"] == test_id
            
            # Check id is first field (Python 3.7+ dicts maintain insertion order)
            keys = list(data.keys())
            assert keys[0] == "id"


def test_save_and_load_roundtrip_preserves_id():
    """Test that saving and loading preserves the id field"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_id = str(uuid.uuid4())
        original_record = GoldenRecord(
            id=test_id,
            user_input="What is RAG?",
            reference="RAG stands for Retrieval Augmented Generation",
            reference_contexts=["Context 1", "Context 2"],
            reference_context_ids=["ctx-1", "ctx-2"],
            metadata={"source": "test"}
        )
        
        # Save records
        loader = JSONLLoader(tmpdir)
        loader.save_golden_records(iter([original_record]))
        
        # Load records back
        loaded_records = list(loader.load_golden_records())
        
        assert len(loaded_records) == 1
        loaded_record = loaded_records[0]
        
        # Verify all fields match
        assert loaded_record.id == original_record.id
        assert loaded_record.user_input == original_record.user_input
        assert loaded_record.reference == original_record.reference
        assert loaded_record.reference_contexts == original_record.reference_contexts
        assert loaded_record.reference_context_ids == original_record.reference_context_ids
        assert loaded_record.metadata == original_record.metadata
