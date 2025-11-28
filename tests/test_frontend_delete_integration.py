"""Integration test for frontend delete task functionality"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import shutil


@pytest.fixture
def client():
    """Create a test client"""
    from rag_benchmark.api.main import app
    return TestClient(app)


def test_delete_task_api_response_format(client, monkeypatch, tmp_path):
    """Test that delete endpoint returns expected response format for frontend"""
    # Setup: Create a temporary tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    
    # Monkeypatch the TASKS_DIR
    from rag_benchmark.api import main
    monkeypatch.setattr(main, "TASKS_DIR", tasks_dir)
    
    # Create a test task
    task_id = "test-task-123"
    task_dir = tasks_dir / task_id
    task_dir.mkdir()
    
    # Create task files
    status_file = task_dir / "status.json"
    status_file.write_text(json.dumps({
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0,
        "created_at": "2025-11-24T10:00:00",
        "updated_at": "2025-11-24T10:10:00"
    }))
    
    config_file = task_dir / "config.json"
    config_file.write_text(json.dumps({
        "dataset_name": "test_dataset",
        "rag_name": "test_rag"
    }))
    
    # Add task to in-memory status
    main.tasks_status[task_id] = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0
    }
    
    # Call delete endpoint
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    
    # Check response format expected by frontend
    assert "message" in data
    assert "task_id" in data
    assert data["task_id"] == task_id
    assert "deleted_from_memory" in data
    assert "deleted_from_disk" in data
    assert data["deleted_from_memory"] is True
    assert data["deleted_from_disk"] is True
    
    # Verify task was removed from memory
    assert task_id not in main.tasks_status
    
    # Verify task directory was removed
    assert not task_dir.exists()


def test_delete_nonexistent_task_error_format(client, monkeypatch, tmp_path):
    """Test that delete endpoint returns proper error format for frontend"""
    # Setup: Create a temporary tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    
    # Monkeypatch the TASKS_DIR
    from rag_benchmark.api import main
    monkeypatch.setattr(main, "TASKS_DIR", tasks_dir)
    
    # Try to delete non-existent task
    task_id = "nonexistent-task"
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify error response
    assert response.status_code == 404
    data = response.json()
    
    # Check error format expected by frontend
    assert "detail" in data
    detail = data["detail"]
    assert isinstance(detail, dict)
    assert "error" in detail
    assert "details" in detail
    assert "action" in detail
    assert task_id in detail["details"]


def test_delete_task_with_checkpoint_files(client, monkeypatch, tmp_path):
    """Test that delete removes all task files including checkpoints"""
    # Setup: Create a temporary tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    
    # Monkeypatch the TASKS_DIR
    from rag_benchmark.api import main
    monkeypatch.setattr(main, "TASKS_DIR", tasks_dir)
    
    # Create a test task with multiple files
    task_id = "test-task-with-checkpoints"
    task_dir = tasks_dir / task_id
    task_dir.mkdir()
    
    # Create various task files
    (task_dir / "status.json").write_text("{}")
    (task_dir / "config.json").write_text("{}")
    (task_dir / "checkpoint.json").write_text("{}")
    (task_dir / "experiment_dataset.pkl").write_text("dummy data")
    (task_dir / "results.json").write_text("{}")
    
    # Add task to in-memory status
    main.tasks_status[task_id] = {"task_id": task_id}
    
    # Verify files exist before deletion
    assert (task_dir / "status.json").exists()
    assert (task_dir / "config.json").exists()
    assert (task_dir / "checkpoint.json").exists()
    assert (task_dir / "experiment_dataset.pkl").exists()
    assert (task_dir / "results.json").exists()
    
    # Call delete endpoint
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    
    # Verify entire directory was removed
    assert not task_dir.exists()
    assert task_id not in main.tasks_status
