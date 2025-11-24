"""Tests for POST /datasets/sample endpoint

Tests verify that the endpoint correctly returns sample information including IDs,
validating Requirement 6.1
"""

import pytest
from fastapi.testclient import TestClient

from rag_benchmark.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_sample_endpoint_includes_id_field(client):
    """Test that /datasets/sample endpoint includes id field for each sample
    
    Validates Requirement 6.1: WHEN the `/datasets/sample` endpoint returns samples 
    THEN the system SHALL include the `id` field for each sample
    """
    # Use xquad dataset which should have IDs after Phase 1 implementation
    response = client.post(
        "/datasets/sample",
        json={"name": "xquad", "subset": None},
        params={"n": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "samples" in data
    assert len(data["samples"]) > 0
    
    # Verify each sample has an id field
    for sample in data["samples"]:
        assert "id" in sample, "Sample must include 'id' field"
        assert isinstance(sample["id"], str), "Sample ID must be a string"
        assert len(sample["id"]) > 0, "Sample ID must not be empty"
        
        # Also verify other expected fields are still present
        assert "user_input" in sample
        assert "reference" in sample
        assert "reference_contexts" in sample


def test_sample_endpoint_id_format(client):
    """Test that sample IDs are in valid format
    
    Validates that IDs are non-empty strings (UUID format will be enforced after dataset regeneration)
    """
    response = client.post(
        "/datasets/sample",
        json={"name": "xquad", "subset": None},
        params={"n": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify each sample ID is a non-empty string
    for sample in data["samples"]:
        sample_id = sample["id"]
        assert isinstance(sample_id, str), f"Sample ID must be a string, got {type(sample_id)}"
        assert len(sample_id) > 0, "Sample ID must not be empty"


def test_sample_endpoint_unique_ids(client):
    """Test that sample IDs are unique within a response
    
    Validates that each sample has a unique ID
    """
    response = client.post(
        "/datasets/sample",
        json={"name": "xquad", "subset": None},
        params={"n": 10}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Collect all IDs
    sample_ids = [sample["id"] for sample in data["samples"]]
    
    # Verify uniqueness
    assert len(sample_ids) == len(set(sample_ids)), "Sample IDs must be unique"
