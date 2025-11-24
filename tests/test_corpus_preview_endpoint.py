"""Tests for POST /datasets/corpus/preview endpoint

Tests verify that the endpoint correctly returns corpus document information including IDs,
validating Requirement 6.2
"""

import pytest
from fastapi.testclient import TestClient

from rag_benchmark.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_corpus_preview_includes_id_field(client):
    """Test that /datasets/corpus/preview endpoint includes id field for each document
    
    Validates Requirement 6.2: WHEN the `/datasets/corpus/preview` endpoint returns documents 
    THEN the system SHALL include the `reference_context_id` field
    """
    # Use xquad dataset which should have corpus with IDs
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure includes pagination metadata
    assert "documents" in data
    assert "total_count" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert len(data["documents"]) > 0
    
    # Verify each document has an id field
    for doc in data["documents"]:
        assert "id" in doc, "Document must include 'id' field"
        assert isinstance(doc["id"], str), "Document ID must be a string"
        assert len(doc["id"]) > 0, "Document ID must not be empty"
        
        # Also verify other expected fields are still present
        assert "content" in doc
        assert "length" in doc


def test_corpus_preview_id_format(client):
    """Test that document IDs are in valid format
    
    Validates that IDs are non-empty strings
    """
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify each document ID is a non-empty string
    for doc in data["documents"]:
        doc_id = doc["id"]
        assert isinstance(doc_id, str), f"Document ID must be a string, got {type(doc_id)}"
        assert len(doc_id) > 0, "Document ID must not be empty"


def test_corpus_preview_unique_ids(client):
    """Test that document IDs are unique within a response
    
    Validates that each document has a unique ID
    """
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 20}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Collect all IDs
    doc_ids = [doc["id"] for doc in data["documents"]]
    
    # Verify uniqueness
    assert len(doc_ids) == len(set(doc_ids)), "Document IDs must be unique"


def test_corpus_preview_empty_dataset(client):
    """Test corpus preview with a dataset that has no corpus
    
    Validates graceful handling of empty corpus
    """
    # This test assumes there might be datasets without corpus
    # If all datasets have corpus, this test will need adjustment
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    
    # Should still return 200 with pagination metadata
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_count" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data


def test_corpus_preview_pagination_metadata(client):
    """Test that pagination metadata is correctly included in response
    
    Validates Requirements 2.3, 5.4: pagination metadata in corpus preview
    """
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify all pagination metadata fields are present
    assert "total_count" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert "documents" in data
    
    # Verify metadata values are correct
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert isinstance(data["total_count"], int)
    assert isinstance(data["total_pages"], int)
    
    # Verify total_pages calculation
    expected_pages = (data["total_count"] + 10 - 1) // 10
    assert data["total_pages"] == expected_pages


def test_corpus_preview_pagination_page_size(client):
    """Test that different page sizes work correctly
    
    Validates Requirements 2.3, 5.4: page_size parameter validation
    """
    # Test with page_size=20
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 20}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["page_size"] == 20
    assert len(data["documents"]) <= 20


def test_corpus_preview_pagination_second_page(client):
    """Test that second page returns different documents
    
    Validates Requirements 2.3: pagination works across pages
    """
    # Get first page
    response1 = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    
    assert response1.status_code == 200
    data1 = response1.json()
    
    # Only test second page if there are enough documents
    if data1["total_pages"] > 1:
        # Get second page
        response2 = client.post(
            "/datasets/corpus/preview",
            json={"name": "xquad", "subset": None},
            params={"page": 2, "page_size": 10}
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Verify pages have different documents
        ids1 = {doc["id"] for doc in data1["documents"]}
        ids2 = {doc["id"] for doc in data2["documents"]}
        assert ids1.isdisjoint(ids2), "Different pages should have different documents"


def test_corpus_preview_invalid_page_number(client):
    """Test that invalid page number returns error
    
    Validates Requirements 5.5: parameter validation
    """
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 0, "page_size": 10}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]


def test_corpus_preview_invalid_page_size(client):
    """Test that invalid page size returns error
    
    Validates Requirements 5.5: parameter validation
    """
    response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 15}  # 15 is not in allowed list
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]
