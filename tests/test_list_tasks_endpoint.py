"""Tests for GET /evaluate/tasks endpoint with status filtering"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import shutil
from datetime import datetime

from rag_benchmark.api.main import app, TASKS_DIR


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def setup_test_tasks():
    """Create test tasks with different statuses"""
    # Create test task directories with different statuses
    test_tasks = []
    
    statuses = ["pending", "running", "completed", "failed"]
    
    for i, status in enumerate(statuses):
        task_id = f"test-task-{status}-{i}"
        task_dir = TASKS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create status.json with sample_info
        task_status = {
            "task_id": task_id,
            "status": status,
            "progress": 0.5 if status == "running" else (1.0 if status == "completed" else 0.0),
            "current_stage": f"Stage for {status}",
            "result": {"metrics": {"test": 0.9}} if status == "completed" else None,
            "error": "Test error" if status == "failed" else None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "sample_info": {
                "selection_strategy": "all",
                "total_samples": 100,
                "selected_samples": 100,
                "sample_ids": None,
                "completed_samples": 100 if status == "completed" else 0,
                "failed_samples": 0
            }
        }
        
        status_file = task_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump(task_status, f, indent=2)
        
        test_tasks.append(task_id)
    
    yield test_tasks
    
    # Cleanup
    for task_id in test_tasks:
        task_dir = TASKS_DIR / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)


def test_list_all_tasks(client, setup_test_tasks):
    """Test listing all tasks without filter"""
    response = client.get("/evaluate/tasks")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "tasks" in data
    assert "count" in data
    assert data["count"] >= 4  # At least our 4 test tasks
    
    # Verify all tasks have sample_info
    for task in data["tasks"]:
        if task["task_id"].startswith("test-task-"):
            assert "sample_info" in task
            assert task["sample_info"] is not None


def test_list_tasks_filter_pending(client, setup_test_tasks):
    """Test listing tasks with pending status filter"""
    response = client.get("/evaluate/tasks?status=pending")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "tasks" in data
    assert "count" in data
    assert "filter" in data
    assert "total_count" in data
    assert data["filter"] == "pending"
    
    # All returned tasks should have pending status
    for task in data["tasks"]:
        assert task["status"] == "pending"
    
    # Should have at least 1 pending task
    assert data["count"] >= 1


def test_list_tasks_filter_running(client, setup_test_tasks):
    """Test listing tasks with running status filter"""
    response = client.get("/evaluate/tasks?status=running")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["filter"] == "running"
    
    # All returned tasks should have running status
    for task in data["tasks"]:
        assert task["status"] == "running"
    
    # Should have at least 1 running task
    assert data["count"] >= 1


def test_list_tasks_filter_completed(client, setup_test_tasks):
    """Test listing tasks with completed status filter"""
    response = client.get("/evaluate/tasks?status=completed")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["filter"] == "completed"
    
    # All returned tasks should have completed status
    for task in data["tasks"]:
        assert task["status"] == "completed"
        # Completed tasks should have sample_info with completed_samples
        if task["task_id"].startswith("test-task-"):
            assert task["sample_info"]["completed_samples"] > 0
    
    # Should have at least 1 completed task
    assert data["count"] >= 1


def test_list_tasks_filter_failed(client, setup_test_tasks):
    """Test listing tasks with failed status filter"""
    response = client.get("/evaluate/tasks?status=failed")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["filter"] == "failed"
    
    # All returned tasks should have failed status
    for task in data["tasks"]:
        assert task["status"] == "failed"
    
    # Should have at least 1 failed task
    assert data["count"] >= 1


def test_list_tasks_invalid_status_filter(client):
    """Test listing tasks with invalid status filter"""
    response = client.get("/evaluate/tasks?status=invalid_status")
    
    assert response.status_code == 400
    data = response.json()
    
    assert "detail" in data
    assert "error" in data["detail"]
    assert "Invalid status filter" in data["detail"]["error"]


def test_list_tasks_includes_sample_info(client, setup_test_tasks):
    """Test that listed tasks include sample_info"""
    response = client.get("/evaluate/tasks")
    
    assert response.status_code == 200
    data = response.json()
    
    # Find our test tasks
    test_tasks = [task for task in data["tasks"] if task["task_id"].startswith("test-task-")]
    
    assert len(test_tasks) >= 4
    
    for task in test_tasks:
        assert "sample_info" in task
        sample_info = task["sample_info"]
        
        assert "selection_strategy" in sample_info
        assert "total_samples" in sample_info
        assert "selected_samples" in sample_info
        assert "completed_samples" in sample_info
        assert "failed_samples" in sample_info


def test_list_tasks_loads_from_directory_structure(client, setup_test_tasks):
    """Test that tasks are loaded from new directory structure"""
    response = client.get("/evaluate/tasks")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify that our test tasks created in directory structure are loaded
    task_ids = [task["task_id"] for task in data["tasks"]]
    
    for test_task_id in setup_test_tasks:
        assert test_task_id in task_ids


def test_list_tasks_filter_returns_subset(client, setup_test_tasks):
    """Test that filtering returns a subset of all tasks"""
    # Get all tasks
    response_all = client.get("/evaluate/tasks")
    all_data = response_all.json()
    
    # Get filtered tasks
    response_filtered = client.get("/evaluate/tasks?status=completed")
    filtered_data = response_filtered.json()
    
    # Filtered count should be less than or equal to total count
    assert filtered_data["count"] <= filtered_data["total_count"]
    assert filtered_data["total_count"] == all_data["count"]
