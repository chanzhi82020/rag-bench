"""Tests for GET /evaluate/{task_id}/samples endpoint

Tests verify that the endpoint correctly returns sample information and detailed results
for evaluation tasks, validating Requirements 4.1, 4.2, 4.3, 4.4, 4.5
"""

import pytest
from fastapi.testclient import TestClient
import json
import shutil
from datetime import datetime

from rag_benchmark.api.main import app, TASKS_DIR, tasks_status


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def cleanup_tasks():
    """Clean up test tasks after each test"""
    yield
    # Clean up any test tasks
    for task_id in list(tasks_status.keys()):
        if task_id.startswith("test-"):
            tasks_status.pop(task_id, None)
            task_dir = TASKS_DIR / task_id
            if task_dir.exists():
                shutil.rmtree(task_dir)


def create_test_task(task_id: str, status: str = "completed", include_sample_info: bool = True, include_results: bool = True):
    """Helper to create a test task with sample info and results"""
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample_info
    sample_info = None
    if include_sample_info:
        sample_info = {
            "selection_strategy": "specific_ids",
            "total_samples": 100,
            "selected_samples": 10,
            "sample_ids": [f"sample_{i}" for i in range(10)],
            "completed_samples": 8,
            "failed_samples": 2
        }
    
    # Create detailed results
    detailed_results = []
    if include_results and status == "completed":
        for i in range(10):
            result = {
                "user_input": f"Question {i}",
                "response": f"Answer {i}",
                "faithfulness": 0.9 if i < 8 else None,  # First 8 succeeded, last 2 failed
                "answer_relevancy": 0.85 if i < 8 else None
            }
            detailed_results.append(result)
    
    # Create task status
    task_data = {
        "task_id": task_id,
        "status": status,
        "progress": 1.0 if status == "completed" else 0.5,
        "current_stage": "完成" if status == "completed" else "运行中",
        "result": {
            "metrics": {"faithfulness": 0.9, "answer_relevancy": 0.85},
            "detailed_results": detailed_results,
            "sample_count": 10,
            "eval_type": "e2e"
        } if include_results and status == "completed" else None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sample_info": sample_info
    }
    
    # Save to disk
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    # Also add to memory
    tasks_status[task_id] = task_data
    
    return task_data


def test_get_samples_for_completed_task(client, cleanup_tasks):
    """Test getting samples for a completed task with results
    
    Validates Requirements 4.1, 4.2, 4.3, 4.4, 4.5
    """
    task_id = "test-completed-task"
    create_test_task(task_id, status="completed")
    
    response = client.get(f"/evaluate/{task_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert data["task_id"] == task_id
    assert data["task_status"] == "completed"
    assert data["sample_info"] is not None
    assert data["detailed_results"] is not None
    assert data["result_count"] == 10
    
    # Verify sample_info (Requirements 4.1, 4.2, 4.3)
    sample_info = data["sample_info"]
    assert sample_info["selection_strategy"] == "specific_ids"
    assert sample_info["total_samples"] == 100
    assert sample_info["selected_samples"] == 10
    assert len(sample_info["sample_ids"]) == 10
    
    # Verify completed/failed counts (Requirement 4.4)
    assert sample_info["completed_samples"] == 8
    assert sample_info["failed_samples"] == 2
    
    # Verify detailed results (Requirement 4.5)
    assert len(data["detailed_results"]) == 10
    # Check that first 8 have metrics (succeeded)
    for i in range(8):
        assert data["detailed_results"][i]["faithfulness"] is not None
        assert data["detailed_results"][i]["answer_relevancy"] is not None
    # Check that last 2 don't have metrics (failed)
    for i in range(8, 10):
        assert data["detailed_results"][i]["faithfulness"] is None
        assert data["detailed_results"][i]["answer_relevancy"] is None


def test_get_samples_for_running_task(client, cleanup_tasks):
    """Test getting samples for a running task (no results yet)"""
    task_id = "test-running-task"
    create_test_task(task_id, status="running", include_results=False)
    
    response = client.get(f"/evaluate/{task_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["task_id"] == task_id
    assert data["task_status"] == "running"
    assert data["sample_info"] is not None
    assert data["detailed_results"] == []
    assert data["result_count"] == 0


def test_get_samples_for_task_without_sample_info(client, cleanup_tasks):
    """Test getting samples for a task without sample_info (backward compatibility)"""
    task_id = "test-no-sample-info"
    create_test_task(task_id, status="completed", include_sample_info=False)
    
    response = client.get(f"/evaluate/{task_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["task_id"] == task_id
    assert data["task_status"] == "completed"
    assert data["sample_info"] is None
    assert data["detailed_results"] is not None


def test_get_samples_for_nonexistent_task(client, cleanup_tasks):
    """Test getting samples for a task that doesn't exist"""
    response = client.get("/evaluate/nonexistent-task-id/samples")
    
    assert response.status_code == 404
    data = response.json()
    assert "error" in data["detail"]
    assert "not found" in data["detail"]["error"].lower()


def test_get_samples_loads_from_disk(client, cleanup_tasks):
    """Test that endpoint loads task from disk if not in memory"""
    task_id = "test-disk-task"
    create_test_task(task_id, status="completed")
    
    # Remove from memory
    tasks_status.pop(task_id, None)
    
    # Should still work by loading from disk
    response = client.get(f"/evaluate/{task_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["sample_info"] is not None


def test_get_samples_with_random_selection(client, cleanup_tasks):
    """Test getting samples for a task with random selection strategy"""
    task_id = "test-random-selection"
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    sample_info = {
        "selection_strategy": "random",
        "total_samples": 1000,
        "selected_samples": 50,
        "sample_ids": None,  # Random selection doesn't track specific IDs
        "completed_samples": 50,
        "failed_samples": 0
    }
    
    task_data = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0,
        "current_stage": "完成",
        "result": {
            "metrics": {"faithfulness": 0.9},
            "detailed_results": [{"user_input": f"Q{i}", "response": f"A{i}"} for i in range(50)],
            "sample_count": 50,
            "eval_type": "e2e"
        },
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sample_info": sample_info
    }
    
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    tasks_status[task_id] = task_data
    
    response = client.get(f"/evaluate/{task_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["sample_info"]["selection_strategy"] == "random"
    assert data["sample_info"]["sample_ids"] is None
    assert data["sample_info"]["selected_samples"] == 50
    assert data["result_count"] == 50


def test_get_samples_with_all_selection(client, cleanup_tasks):
    """Test getting samples for a task with 'all' selection strategy"""
    task_id = "test-all-selection"
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    sample_info = {
        "selection_strategy": "all",
        "total_samples": 200,
        "selected_samples": 200,
        "sample_ids": None,
        "completed_samples": 200,
        "failed_samples": 0
    }
    
    task_data = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0,
        "current_stage": "完成",
        "result": {
            "metrics": {"faithfulness": 0.9},
            "detailed_results": [],  # Empty for brevity
            "sample_count": 200,
            "eval_type": "e2e"
        },
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sample_info": sample_info
    }
    
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    tasks_status[task_id] = task_data
    
    response = client.get(f"/evaluate/{task_id}/samples")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["sample_info"]["selection_strategy"] == "all"
    assert data["sample_info"]["total_samples"] == 200
    assert data["sample_info"]["selected_samples"] == 200
