"""Tests for POST /datasets/corpus/by-id endpoint

Tests verify that the endpoint correctly retrieves specific corpus documents by their IDs.
"""

import pytest
from fastapi.testclient import TestClient
from rag_benchmark.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_corpus_by_id_basic(client):
    """Test basic corpus document retrieval by IDs
    
    Validates Requirement 3.2: WHEN a user clicks on a reference_context_id 
    THEN the system SHALL navigate to or highlight the corresponding corpus document
    """
    # First get a valid document ID from the corpus
    preview_response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": "en"},
        params={"page": 1, "page_size": 100}
    )
    assert preview_response.status_code == 200
    preview_data = preview_response.json()
    
    # Skip test if no corpus documents available
    if len(preview_data["documents"]) == 0:
        pytest.skip("No corpus documents available for testing")
    
    valid_doc_id = preview_data["documents"][0]["id"]
    
    # Now test the by-id endpoint
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "xquad",
                "subset": "en"
            },
            "document_ids": [valid_doc_id]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "dataset_name" in data
    assert "subset" in data
    assert "documents" in data
    assert "requested_count" in data
    assert "found_count" in data
    
    # Verify dataset info
    assert data["dataset_name"] == "xquad"
    assert data["subset"] == "en"
    
    # Verify document structure
    assert len(data["documents"]) > 0
    doc = data["documents"][0]
    assert "id" in doc
    assert "content" in doc
    assert "length" in doc
    
    # Verify counts
    assert data["requested_count"] == 1
    assert data["found_count"] == len(data["documents"])


def test_corpus_by_id_multiple_documents(client):
    """Test retrieving multiple corpus documents
    
    Validates Requirement 3.2: System can retrieve multiple corpus documents
    """
    # Get two valid document IDs
    preview_response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": "en"},
        params={"page": 1, "page_size": 10}
    )
    assert preview_response.status_code == 200
    preview_data = preview_response.json()
    
    # Skip test if not enough corpus documents
    if len(preview_data["documents"]) < 2:
        pytest.skip("Not enough corpus documents for testing")
    
    doc_ids = [doc["id"] for doc in preview_data["documents"][:2]]
    
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "xquad",
                "subset": "en"
            },
            "document_ids": doc_ids
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return documents (may be less than requested if some IDs don't exist)
    assert data["requested_count"] == 2
    assert data["found_count"] <= 2
    assert len(data["documents"]) == data["found_count"]


def test_corpus_by_id_empty_list(client):
    """Test that empty document_ids list returns error"""
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "xquad",
                "subset": "en"
            },
            "document_ids": []
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]


def test_corpus_by_id_nonexistent_ids(client):
    """Test retrieving nonexistent corpus documents
    
    Should return empty documents list and report missing IDs
    """
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "xquad",
                "subset": "en"
            },
            "document_ids": ["nonexistent_id_12345"]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return empty documents but valid response
    assert data["requested_count"] == 1
    assert data["found_count"] == 0
    assert len(data["documents"]) == 0
    assert data["missing_ids"] is not None
    assert "nonexistent_id_12345" in data["missing_ids"]


def test_corpus_by_id_mixed_valid_invalid(client):
    """Test retrieving mix of valid and invalid document IDs
    
    Should return only valid documents and report missing ones
    """
    # Get one valid document ID
    preview_response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": "en"},
        params={"page": 1, "page_size": 10}
    )
    assert preview_response.status_code == 200
    preview_data = preview_response.json()
    
    # Skip test if no corpus documents available
    if len(preview_data["documents"]) == 0:
        pytest.skip("No corpus documents available for testing")
    
    valid_doc_id = preview_data["documents"][0]["id"]
    
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "xquad",
                "subset": "en"
            },
            "document_ids": [valid_doc_id, "invalid_id_999"]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return only valid documents
    assert data["requested_count"] == 2
    assert data["found_count"] < data["requested_count"]
    assert len(data["documents"]) == data["found_count"]
    
    # Should report missing IDs
    if data["missing_ids"]:
        assert "invalid_id_999" in data["missing_ids"]


def test_corpus_by_id_invalid_dataset(client):
    """Test with invalid dataset name"""
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "nonexistent_dataset",
                "subset": "en"
            },
            "document_ids": ["some_id"]
        }
    )
    
    assert response.status_code == 400


def test_corpus_by_id_document_content_structure(client):
    """Test that returned documents have correct structure and content
    
    Validates Requirement 2.2: WHEN viewing corpus THEN the system SHALL 
    display document ID, content preview, and metadata
    """
    # Get a valid document ID
    preview_response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": "en"},
        params={"page": 1, "page_size": 10}
    )
    assert preview_response.status_code == 200
    preview_data = preview_response.json()
    
    # Skip test if no corpus documents available
    if len(preview_data["documents"]) == 0:
        pytest.skip("No corpus documents available for testing")
    
    valid_doc_id = preview_data["documents"][0]["id"]
    
    response = client.post(
        "/datasets/corpus/by-id",
        json={
            "dataset_info": {
                "name": "xquad",
                "subset": "en"
            },
            "document_ids": [valid_doc_id]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    if data["found_count"] > 0:
        doc = data["documents"][0]
        
        # Verify document structure
        assert isinstance(doc["id"], str)
        assert isinstance(doc["content"], str)
        assert isinstance(doc["length"], int)
        
        # Verify content is not empty
        assert len(doc["content"]) > 0
        assert doc["length"] == len(doc["content"])
