"""Tests for POST /rag/index endpoint validation

Tests verify that the endpoint correctly validates document IDs and provides clear error messages,
validating Requirements 6.3, 8.2, 8.3, 8.4, 8.5
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from rag_benchmark.api.main import app, rag_instances
from rag_benchmark.prepare import BaselineRAG, RAGConfig


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_rag_instance():
    """Create a mock RAG instance for testing"""
    # Create a mock RAG
    mock_rag = Mock(spec=BaselineRAG)
    mock_rag.index_documents = Mock()
    mock_rag.save_to_disk = Mock()
    
    # Add to rag_instances
    rag_name = "test_rag_for_indexing"
    rag_instances[rag_name] = {
        "rag": mock_rag,
        "rag_type": "baseline",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "rag_config": {
            "top_k": 5,
            "temperature": 0.7,
            "max_length": 512
        }
    }
    
    yield rag_name
    
    # Cleanup
    if rag_name in rag_instances:
        del rag_instances[rag_name]


def test_rag_index_with_invalid_document_ids(client, mock_rag_instance):
    """Test that invalid document IDs are rejected with clear error message
    
    Validates Requirements 8.4, 8.5: THE system SHALL validate that provided document IDs exist in the corpus
    and THE system SHALL return an error if any document ID is invalid
    """
    request_data = {
        "rag_name": mock_rag_instance,
        "dataset_name": "xquad",
        "subset": None,
        "document_ids": ["invalid-doc-id-1", "invalid-doc-id-2", "nonexistent-id"]
    }
    
    response = client.post("/rag/index", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    
    # Verify error structure
    assert "error" in error_detail
    assert "details" in error_detail
    assert "action" in error_detail
    
    # Verify error message mentions invalid document IDs
    assert "Invalid document IDs" in error_detail["error"]
    assert "invalid-doc-id-1" in str(error_detail["details"]) or "do not exist" in error_detail["details"]
    
    # Verify action suggests using /datasets/corpus/preview
    assert "/datasets/corpus/preview" in error_detail["action"]


def test_rag_index_with_mixed_valid_invalid_document_ids(client, mock_rag_instance):
    """Test that validation catches invalid document IDs even when some are valid
    
    Validates that ALL document IDs must be valid
    """
    # First get some valid document IDs
    corpus_response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    assert corpus_response.status_code == 200
    valid_doc_ids = [doc["id"] for doc in corpus_response.json()["documents"]][:2]
    
    # Mix valid and invalid IDs
    mixed_ids = valid_doc_ids + ["invalid-doc-id-1", "invalid-doc-id-2"]
    
    request_data = {
        "rag_name": mock_rag_instance,
        "dataset_name": "xquad",
        "subset": None,
        "document_ids": mixed_ids
    }
    
    response = client.post("/rag/index", json=request_data)
    
    # Should reject because some IDs are invalid
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "Invalid document IDs" in error_detail["error"]
    
    # Verify the invalid IDs are mentioned
    assert "invalid-doc-id-1" in str(error_detail["details"]) or "do not exist" in error_detail["details"]


def test_rag_index_with_valid_document_ids(client, mock_rag_instance):
    """Test that valid document IDs are accepted and indexing proceeds
    
    Validates Requirements 6.3, 8.2, 8.3: WHEN selecting documents for indexing THEN the system SHALL accept document IDs
    and WHEN indexing selected documents THEN the system SHALL use the provided document IDs to filter the corpus
    """
    # Get some valid document IDs
    corpus_response = client.post(
        "/datasets/corpus/preview",
        json={"name": "xquad", "subset": None},
        params={"page": 1, "page_size": 10}
    )
    assert corpus_response.status_code == 200
    valid_doc_ids = [doc["id"] for doc in corpus_response.json()["documents"]][:3]
    
    request_data = {
        "rag_name": mock_rag_instance,
        "dataset_name": "xquad",
        "subset": None,
        "document_ids": valid_doc_ids
    }
    
    response = client.post("/rag/index", json=request_data)
    
    # Should succeed
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "message" in data
    assert "document_count" in data
    assert "total_corpus_count" in data
    
    # Verify only selected documents were indexed
    assert data["document_count"] == len(valid_doc_ids)
    assert data["document_count"] <= data["total_corpus_count"]


def test_rag_index_without_document_ids_indexes_all(client, mock_rag_instance):
    """Test that omitting document_ids indexes all documents
    
    Validates backward compatibility - no document_ids means index all
    """
    request_data = {
        "rag_name": mock_rag_instance,
        "dataset_name": "xquad",
        "subset": None
        # No document_ids specified
    }
    
    response = client.post("/rag/index", json=request_data)
    
    # Should succeed and index all documents
    assert response.status_code == 200
    data = response.json()
    
    # Verify all documents were indexed
    assert data["document_count"] == data["total_corpus_count"]


def test_rag_index_with_empty_document_ids_list(client, mock_rag_instance):
    """Test that empty document_ids list is handled gracefully
    
    Validates edge case handling
    """
    request_data = {
        "rag_name": mock_rag_instance,
        "dataset_name": "xquad",
        "subset": None,
        "document_ids": []  # Empty list
    }
    
    response = client.post("/rag/index", json=request_data)
    
    # Should succeed but index 0 documents
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 0


def test_rag_index_nonexistent_rag(client):
    """Test that indexing with nonexistent RAG returns 404"""
    request_data = {
        "rag_name": "nonexistent_rag",
        "dataset_name": "xquad",
        "subset": None
    }
    
    response = client.post("/rag/index", json=request_data)
    
    assert response.status_code == 404
    assert "不存在" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
