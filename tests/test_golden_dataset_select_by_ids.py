"""Tests for GoldenDataset.select_by_ids() method"""

import pytest
from typing import Iterator

from rag_benchmark.datasets.golden import GoldenDataset
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord
from rag_benchmark.datasets.loaders.base import BaseLoader


class MockLoader(BaseLoader):
    """Mock loader for testing"""
    
    def __init__(self, num_records: int = 10):
        self.num_records = num_records
        self.records = [
            GoldenRecord(
                id=str(i),
                user_input=f"Question {i}",
                reference=f"Answer {i}",
                reference_contexts=[f"Context {i}"],
                reference_context_ids=[f"ctx_{i}"],
                metadata={"index": i}
            )
            for i in range(num_records)
        ]
        self.corpus = [
            CorpusRecord(
                reference_context=f"Context {i}",
                reference_context_id=f"ctx_{i}",
                title=f"Document {i}",
                metadata={}
            )
            for i in range(num_records)
        ]
    
    def load_golden_records(self) -> Iterator[GoldenRecord]:
        return iter(self.records)
    
    def load_corpus_records(self) -> Iterator[CorpusRecord]:
        return iter(self.corpus)
    
    def count_records(self) -> int:
        return self.num_records


def test_select_by_ids_basic():
    """Test basic select_by_ids functionality"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "2", "5"]
    
    result = dataset.select_by_ids(sample_ids)
    
    assert result.count() == 3
    records = list(result)
    assert records[0].user_input == "Question 0"
    assert records[1].user_input == "Question 2"
    assert records[2].user_input == "Question 5"


def test_select_by_ids_returns_new_dataset():
    """Test that select_by_ids returns a new GoldenDataset instance"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "1"]
    
    result = dataset.select_by_ids(sample_ids)
    
    assert isinstance(result, GoldenDataset)
    assert result is not dataset
    assert result.count() == 2
    assert dataset.count() == 10  # Original unchanged


def test_select_by_ids_filters_corpus():
    """Test that corpus records are filtered appropriately"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["1", "3", "7"]
    
    result = dataset.select_by_ids(sample_ids)
    
    # Check corpus records
    corpus_records = list(result.iter_corpus())
    corpus_ids = {r.reference_context_id for r in corpus_records}
    
    # Should only have corpus records for selected samples
    assert "ctx_1" in corpus_ids
    assert "ctx_3" in corpus_ids
    assert "ctx_7" in corpus_ids
    assert len(corpus_ids) == 3
    
    # Should not have corpus records for non-selected samples
    assert "ctx_0" not in corpus_ids
    assert "ctx_2" not in corpus_ids


def test_select_by_ids_invalid_id_format():
    """Test that invalid ID format raises ValueError"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "abc", "2"]
    
    with pytest.raises(ValueError, match="Invalid record IDs"):
        dataset.select_by_ids(sample_ids)


def test_select_by_ids_out_of_range():
    """Test that out-of-range IDs raise ValueError"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "15", "2"]
    
    with pytest.raises(ValueError, match="Invalid record IDs"):
        dataset.select_by_ids(sample_ids)


def test_select_by_ids_negative_index():
    """Test that negative indices raise ValueError"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["-1", "2"]
    
    with pytest.raises(ValueError, match="Invalid record IDs"):
        dataset.select_by_ids(sample_ids)


def test_select_by_ids_empty_list():
    """Test selecting with empty list"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = []
    
    result = dataset.select_by_ids(sample_ids)
    
    assert result.count() == 0
    assert list(result) == []


def test_select_by_ids_single_sample():
    """Test selecting a single sample"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["5"]
    
    result = dataset.select_by_ids(sample_ids)
    
    assert result.count() == 1
    records = list(result)
    assert records[0].user_input == "Question 5"


def test_select_by_ids_duplicate_ids():
    """Test selecting with duplicate IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["2", "2", "5"]
    
    result = dataset.select_by_ids(sample_ids)
    
    # Should include duplicates
    assert result.count() == 3
    records = list(result)
    assert records[0].user_input == "Question 2"
    assert records[1].user_input == "Question 2"
    assert records[2].user_input == "Question 5"


def test_select_by_ids_preserves_order():
    """Test that selected records preserve the order of sample_ids"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["7", "2", "9", "1"]
    
    result = dataset.select_by_ids(sample_ids)
    
    records = list(result)
    assert records[0].user_input == "Question 7"
    assert records[1].user_input == "Question 2"
    assert records[2].user_input == "Question 9"
    assert records[3].user_input == "Question 1"
