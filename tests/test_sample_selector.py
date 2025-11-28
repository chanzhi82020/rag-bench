"""Tests for SampleSelector module"""

import pytest
from typing import Iterator, List

from rag_benchmark.api.sample_selector import SampleSelector
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


def test_select_by_ids_valid():
    """Test selecting samples by valid IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "2", "5"]
    
    result = SampleSelector.select_by_ids(dataset, sample_ids)
    
    assert result.count() == 3
    records = list(result)
    assert records[0].user_input == "Question 0"
    assert records[1].user_input == "Question 2"
    assert records[2].user_input == "Question 5"


def test_select_by_ids_invalid():
    """Test selecting samples with invalid IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "15", "abc"]
    
    with pytest.raises(ValueError, match="Invalid record IDs"):
        SampleSelector.select_by_ids(dataset, sample_ids)


def test_select_random():
    """Test random sample selection"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = SampleSelector.select_random(dataset, n=5, seed=42)
    
    assert result.count() == 5
    
    # Test reproducibility with same seed
    result2 = SampleSelector.select_random(dataset, n=5, seed=42)
    records1 = list(result)
    records2 = list(result2)
    
    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.user_input == r2.user_input


def test_select_random_invalid_size():
    """Test random selection with invalid size"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    with pytest.raises(ValueError, match="Sample size must be positive"):
        SampleSelector.select_random(dataset, n=0)
    
    with pytest.raises(ValueError, match="Sample size must be positive"):
        SampleSelector.select_random(dataset, n=-5)


def test_select_all():
    """Test selecting all samples"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = SampleSelector.select_all(dataset)
    
    assert result is dataset
    assert result.count() == 10


def test_validate_sample_ids_valid():
    """Test validation with valid sample IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "5", "9"]
    
    invalid = SampleSelector.validate_sample_ids(dataset, sample_ids)
    
    assert invalid == []


def test_validate_sample_ids_out_of_range():
    """Test validation with out-of-range IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "10", "15"]
    
    invalid = SampleSelector.validate_sample_ids(dataset, sample_ids)
    
    assert "10" in invalid
    assert "15" in invalid
    assert "0" not in invalid


def test_validate_sample_ids_invalid_format():
    """Test validation with invalid format IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "abc", "1.5", ""]
    
    invalid = SampleSelector.validate_sample_ids(dataset, sample_ids)
    
    assert "abc" in invalid
    assert "1.5" in invalid
    assert "" in invalid
    assert "0" not in invalid


def test_validate_sample_ids_negative():
    """Test validation with negative IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["-1", "5"]
    
    invalid = SampleSelector.validate_sample_ids(dataset, sample_ids)
    
    assert "-1" in invalid
    assert "5" not in invalid


def test_select_by_ids_preserves_corpus():
    """Test that selecting by IDs preserves referenced corpus records"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    sample_ids = ["0", "2"]
    
    result = SampleSelector.select_by_ids(dataset, sample_ids)
    
    corpus_records = list(result.iter_corpus())
    corpus_ids = {r.reference_context_id for r in corpus_records}
    
    # Should only have corpus records for selected samples
    assert "ctx_0" in corpus_ids
    assert "ctx_2" in corpus_ids
    assert len(corpus_ids) == 2
