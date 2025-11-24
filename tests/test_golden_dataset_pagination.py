"""Tests for GoldenDataset pagination, search, and corpus retrieval methods"""

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
                user_input=f"Question {i} about testing",
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


# Tests for paginate() method

def test_paginate_first_page():
    """Test pagination on first page"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    records, total = dataset.paginate(page=1, page_size=3)
    
    assert total == 10
    assert len(records) == 3
    assert records[0].user_input == "Question 0 about testing"
    assert records[1].user_input == "Question 1 about testing"
    assert records[2].user_input == "Question 2 about testing"


def test_paginate_middle_page():
    """Test pagination on middle page"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    records, total = dataset.paginate(page=2, page_size=3)
    
    assert total == 10
    assert len(records) == 3
    assert records[0].user_input == "Question 3 about testing"
    assert records[1].user_input == "Question 4 about testing"
    assert records[2].user_input == "Question 5 about testing"


def test_paginate_last_page_partial():
    """Test pagination on last page with partial results"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    records, total = dataset.paginate(page=4, page_size=3)
    
    assert total == 10
    assert len(records) == 1
    assert records[0].user_input == "Question 9 about testing"


def test_paginate_beyond_last_page():
    """Test pagination beyond last page returns empty list"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    records, total = dataset.paginate(page=5, page_size=3)
    
    assert total == 10
    assert len(records) == 0


def test_paginate_invalid_page_number():
    """Test that page < 1 raises ValueError"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    with pytest.raises(ValueError, match="Page number must be >= 1"):
        dataset.paginate(page=0, page_size=3)


def test_paginate_invalid_page_size():
    """Test that page_size < 1 raises ValueError"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    with pytest.raises(ValueError, match="Page size must be >= 1"):
        dataset.paginate(page=1, page_size=0)


def test_paginate_large_page_size():
    """Test pagination with page size larger than dataset"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    records, total = dataset.paginate(page=1, page_size=20)
    
    assert total == 10
    assert len(records) == 10


# Tests for search() method

def test_search_case_insensitive():
    """Test case-insensitive search"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = dataset.search("QUESTION")
    
    assert result.count() == 10  # All records contain "question"
    records = list(result)
    assert all("Question" in r.user_input for r in records)


def test_search_case_sensitive():
    """Test case-sensitive search"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = dataset.search("QUESTION", case_sensitive=True)
    
    assert result.count() == 0  # No records contain uppercase "QUESTION"


def test_search_partial_match():
    """Test search with partial match"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = dataset.search("testing")
    
    assert result.count() == 10  # All records contain "testing"


def test_search_specific_number():
    """Test search for specific number"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = dataset.search("Question 5")
    
    assert result.count() == 1
    records = list(result)
    assert records[0].user_input == "Question 5 about testing"


def test_search_no_results():
    """Test search with no matching results"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = dataset.search("nonexistent")
    
    assert result.count() == 0
    assert list(result) == []


def test_search_returns_new_dataset():
    """Test that search returns a new GoldenDataset instance"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    result = dataset.search("Question")
    
    assert isinstance(result, GoldenDataset)
    assert result is not dataset
    assert dataset.count() == 10  # Original unchanged


# Tests for get_corpus_by_ids() method

def test_get_corpus_by_ids_single():
    """Test retrieving a single corpus document"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    corpus_docs = dataset.get_corpus_by_ids(["ctx_5"])
    
    assert len(corpus_docs) == 1
    assert corpus_docs[0].reference_context_id == "ctx_5"
    assert corpus_docs[0].reference_context == "Context 5"


def test_get_corpus_by_ids_multiple():
    """Test retrieving multiple corpus documents"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    corpus_docs = dataset.get_corpus_by_ids(["ctx_1", "ctx_3", "ctx_7"])
    
    assert len(corpus_docs) == 3
    corpus_ids = {doc.reference_context_id for doc in corpus_docs}
    assert "ctx_1" in corpus_ids
    assert "ctx_3" in corpus_ids
    assert "ctx_7" in corpus_ids


def test_get_corpus_by_ids_nonexistent():
    """Test retrieving nonexistent corpus documents"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    corpus_docs = dataset.get_corpus_by_ids(["ctx_999"])
    
    assert len(corpus_docs) == 0


def test_get_corpus_by_ids_mixed():
    """Test retrieving mix of existing and nonexistent documents"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    corpus_docs = dataset.get_corpus_by_ids(["ctx_1", "ctx_999", "ctx_5"])
    
    assert len(corpus_docs) == 2
    corpus_ids = {doc.reference_context_id for doc in corpus_docs}
    assert "ctx_1" in corpus_ids
    assert "ctx_5" in corpus_ids
    assert "ctx_999" not in corpus_ids


def test_get_corpus_by_ids_empty_list():
    """Test retrieving with empty ID list"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    corpus_docs = dataset.get_corpus_by_ids([])
    
    assert len(corpus_docs) == 0


def test_get_corpus_by_ids_duplicate_ids():
    """Test retrieving with duplicate IDs"""
    dataset = GoldenDataset(name="test", loader=MockLoader(10))
    
    corpus_docs = dataset.get_corpus_by_ids(["ctx_2", "ctx_2", "ctx_5"])
    
    # Should return unique documents (no duplicates)
    assert len(corpus_docs) == 2
    corpus_ids = {doc.reference_context_id for doc in corpus_docs}
    assert "ctx_2" in corpus_ids
    assert "ctx_5" in corpus_ids
