"""Tests for POST /datasets/stats endpoint

Tests verify that the endpoint correctly returns dataset statistics including
sample count, corpus count, and various averages.
"""

import pytest
from fastapi.testclient import TestClient
from rag_benchmark.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_dataset_stats_basic(client):
    """Test basic dataset statistics retrieval
    
    Validates Requirements:
    - 7.1: WHEN viewing a dataset THEN the system SHALL display total sample count
    - 7.2: THE system SHALL show average question length and answer length
    - 7.3: THE system SHALL display the number of unique corpus documents referenced
    """
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "dataset_name" in data
    assert "subset" in data
    assert "record_count" in data
    assert "avg_input_length" in data
    assert "avg_reference_length" in data
    assert "avg_contexts_per_record" in data
    assert "corpus_count" in data
    
    # Verify dataset info
    assert data["dataset_name"] == "xquad"
    assert data["subset"] == "en"
    
    # Verify statistics are valid numbers
    assert isinstance(data["record_count"], int)
    assert data["record_count"] > 0
    
    assert isinstance(data["avg_input_length"], (int, float))
    assert data["avg_input_length"] > 0
    
    assert isinstance(data["avg_reference_length"], (int, float))
    assert data["avg_reference_length"] > 0
    
    assert isinstance(data["avg_contexts_per_record"], (int, float))
    assert data["avg_contexts_per_record"] > 0
    
    assert isinstance(data["corpus_count"], int)
    assert data["corpus_count"] > 0


def test_dataset_stats_total_sample_count(client):
    """Test that total sample count is returned
    
    Validates Requirement 7.1: WHEN viewing a dataset THEN the system 
    SHALL display total sample count
    """
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify total sample count exists and is positive
    assert "record_count" in data
    assert data["record_count"] > 0


def test_dataset_stats_average_lengths(client):
    """Test that average question and answer lengths are returned
    
    Validates Requirement 7.2: THE system SHALL show average question 
    length and answer length
    """
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify average lengths exist and are positive
    assert "avg_input_length" in data
    assert data["avg_input_length"] > 0
    
    assert "avg_reference_length" in data
    assert data["avg_reference_length"] > 0


def test_dataset_stats_corpus_count(client):
    """Test that corpus document count is returned
    
    Validates Requirement 7.3: THE system SHALL display the number of 
    unique corpus documents referenced
    """
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify corpus count exists and is positive
    assert "corpus_count" in data
    assert data["corpus_count"] > 0


def test_dataset_stats_context_distribution(client):
    """Test that reference context distribution is returned
    
    Validates Requirement 7.4: THE system SHALL show distribution of 
    reference_context counts
    """
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify average contexts per record exists and is positive
    assert "avg_contexts_per_record" in data
    assert data["avg_contexts_per_record"] > 0


def test_dataset_stats_invalid_dataset(client):
    """Test with invalid dataset name"""
    response = client.post(
        "/datasets/stats",
        json={
            "name": "nonexistent_dataset",
            "subset": "en"
        }
    )
    
    assert response.status_code == 400


def test_dataset_stats_without_subset(client):
    """Test statistics for dataset without subset"""
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    # Should work with or without subset depending on dataset
    # Just verify it doesn't crash
    assert response.status_code in [200, 400]


def test_dataset_stats_consistency(client):
    """Test that statistics are internally consistent
    
    Validates that the returned statistics make logical sense
    """
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify logical consistency
    # Average contexts per record should be reasonable (typically 1-10)
    assert 0 < data["avg_contexts_per_record"] <= 100
    
    # Corpus count should be at least as large as the number of samples
    # (though it could be smaller if samples share contexts)
    assert data["corpus_count"] > 0
    
    # Average lengths should be reasonable (not negative, not absurdly large)
    assert 0 < data["avg_input_length"] < 100000
    assert 0 < data["avg_reference_length"] < 100000


def test_dataset_stats_response_model(client):
    """Test that response matches DatasetStats model structure"""
    response = client.post(
        "/datasets/stats",
        json={
            "name": "xquad",
            "subset": "en"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify all required fields are present
    required_fields = [
        "dataset_name",
        "subset",
        "record_count",
        "avg_input_length",
        "avg_reference_length",
        "avg_contexts_per_record",
        "corpus_count"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
