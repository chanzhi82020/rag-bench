"""Test validation in POST /evaluate/start endpoint"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

from rag_benchmark.api.main import app, TASKS_DIR


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def temp_tasks_dir():
    """Create temporary tasks directory for testing"""
    temp_dir = tempfile.mkdtemp()
    original_tasks_dir = TASKS_DIR
    
    # Temporarily replace TASKS_DIR
    import rag_benchmark.api.main as main_module
    main_module.TASKS_DIR = Path(temp_dir)
    
    yield Path(temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    main_module.TASKS_DIR = original_tasks_dir


def test_evaluate_start_with_invalid_strategy(client):
    """Test that invalid strategy is rejected"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "invalid_strategy"
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "Invalid sample selection strategy" in error_detail["error"]
    assert "invalid_strategy" in error_detail["details"]


def test_evaluate_start_specific_ids_without_sample_ids(client):
    """Test that specific_ids strategy requires sample_ids"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "specific_ids"
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "sample_ids is required" in error_detail["details"]


def test_evaluate_start_specific_ids_with_empty_list(client):
    """Test that specific_ids strategy rejects empty sample_ids list"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "specific_ids",
            "sample_ids": []
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "sample_ids is required" in error_detail["details"]
    assert "must not be empty" in error_detail["details"]


def test_evaluate_start_random_without_sample_size(client):
    """Test that random strategy requires sample_size"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "random"
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "sample_size is required" in error_detail["details"]


def test_evaluate_start_random_with_zero_sample_size(client):
    """Test that random strategy rejects zero sample_size"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "random",
            "sample_size": 0
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    # Pydantic validation catches this (422) before our custom validation
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    # Pydantic error format is different - it's a list of errors
    assert isinstance(error_detail, list)
    assert any("sample_size" in str(err) for err in error_detail)


def test_evaluate_start_random_with_negative_sample_size(client):
    """Test that random strategy rejects negative sample_size"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "random",
            "sample_size": -5
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    # Pydantic validation catches this (422) before our custom validation
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    # Pydantic error format is different - it's a list of errors
    assert isinstance(error_detail, list)
    assert any("sample_size" in str(err) for err in error_detail)


def test_evaluate_start_all_strategy_accepts_no_params(client, temp_tasks_dir):
    """Test that 'all' strategy doesn't require additional parameters"""
    # Note: This test will fail if dataset doesn't exist or RAG doesn't exist
    # But it should pass validation and fail later in execution
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "all"
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    # Should pass validation (status 200) even if task fails later
    # Or fail with dataset/RAG not found, not validation error
    if response.status_code != 200:
        # If it fails, it should not be a validation error
        error_detail = response.json().get("detail", {})
        if isinstance(error_detail, dict):
            assert "Invalid sample selection" not in error_detail.get("error", "")


def test_evaluate_start_without_sample_selection(client, temp_tasks_dir):
    """Test backward compatibility - no sample_selection means evaluate all"""
    request_data = {
        "dataset_name": "hotpotqa",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    # Should pass validation (status 200) even if task fails later
    # Or fail with dataset/RAG not found, not validation error
    if response.status_code != 200:
        # If it fails, it should not be a validation error
        error_detail = response.json().get("detail", {})
        if isinstance(error_detail, dict):
            assert "Invalid sample selection" not in error_detail.get("error", "")


def test_evaluate_start_with_invalid_sample_ids(client):
    """Test that invalid sample IDs are rejected with clear error message
    
    Validates Requirements 6.4, 6.5: THE API SHALL validate that provided IDs exist in the dataset
    and THE API SHALL return clear error messages for invalid IDs
    """
    request_data = {
        "dataset_name": "xquad",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "specific_ids",
            "sample_ids": ["invalid-id-1", "invalid-id-2", "nonexistent-uuid"]
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    
    # Verify error structure
    assert "error" in error_detail
    assert "details" in error_detail
    assert "action" in error_detail
    
    # Verify error message mentions invalid IDs
    assert "Invalid sample IDs" in error_detail["error"] or "invalid" in error_detail["details"].lower()
    
    # Verify action suggests using /datasets/sample
    assert "/datasets/sample" in error_detail["action"]


def test_evaluate_start_with_mixed_valid_invalid_ids(client):
    """Test that validation catches invalid IDs even when some are valid
    
    Validates that ALL sample IDs must be valid
    """
    # First get some valid IDs
    sample_response = client.post(
        "/datasets/sample",
        json={"name": "xquad", "subset": None},
        params={"n": 2}
    )
    assert sample_response.status_code == 200
    valid_ids = [s["id"] for s in sample_response.json()["samples"]]
    
    # Mix valid and invalid IDs
    mixed_ids = valid_ids + ["invalid-id-1", "invalid-id-2"]
    
    request_data = {
        "dataset_name": "xquad",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "specific_ids",
            "sample_ids": mixed_ids
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    # Should reject because some IDs are invalid
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "Invalid sample IDs" in error_detail["error"] or "invalid" in error_detail["details"].lower()


def test_evaluate_start_random_exceeds_dataset_size(client):
    """Test that random strategy validates sample_size doesn't exceed dataset size
    
    Validates that sample_size is reasonable for the dataset
    """
    request_data = {
        "dataset_name": "xquad",
        "rag_name": "test_rag",
        "eval_type": "e2e",
        "model_info": {
            "llm_model_id": "gpt-4",
            "embedding_model_id": "text-embedding-3-small"
        },
        "sample_selection": {
            "strategy": "random",
            "sample_size": 999999  # Unreasonably large
        }
    }
    
    response = client.post("/evaluate/start", json=request_data)
    
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "sample_size" in error_detail["details"].lower()
    assert "exceeds" in error_detail["details"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
