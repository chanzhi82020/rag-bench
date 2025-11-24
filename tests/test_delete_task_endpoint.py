"""Integration test for DELETE /evaluate/delete/{task_id} endpoint"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_benchmark.api.main import app, tasks_status, TASKS_DIR


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def temp_task_dir(tmp_path):
    """Create a temporary task directory for testing"""
    # Save original TASKS_DIR
    original_tasks_dir = TASKS_DIR
    
    # Use temporary directory for testing
    test_tasks_dir = tmp_path / "tasks"
    test_tasks_dir.mkdir(parents=True, exist_ok=True)
    
    yield test_tasks_dir
    
    # Cleanup
    if test_tasks_dir.exists():
        shutil.rmtree(test_tasks_dir)


def create_test_task(task_id: str, task_dir: Path):
    """Helper function to create a test task with all files"""
    # Create task directory
    task_path = task_dir / task_id
    task_path.mkdir(parents=True, exist_ok=True)
    
    # Create status.json
    status_data = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0,
        "current_stage": "完成",
        "result": {"metrics": {"faithfulness": 0.85}},
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sample_info": None
    }
    with open(task_path / "status.json", 'w') as f:
        json.dump(status_data, f, indent=2)
    
    # Create config.json
    config_data = {
        "dataset_name": "test_dataset",
        "rag_name": "test_rag",
        "eval_type": "e2e"
    }
    with open(task_path / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Create checkpoint.json
    checkpoint_data = {
        "completed_stages": ["load_dataset", "prepare_experiment"],
        "current_stage": "run_evaluation"
    }
    with open(task_path / "checkpoint.json", 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Create a dummy experiment dataset file
    with open(task_path / "experiment_dataset.pkl", 'wb') as f:
        f.write(b"dummy pickle data")
    
    return task_path


def test_delete_task_success(client, monkeypatch, tmp_path):
    """Test successful deletion of an evaluation task"""
    # Setup: Create a temporary tasks directory
    test_tasks_dir = tmp_path / "tasks"
    test_tasks_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch TASKS_DIR to use temp directory
    import rag_benchmark.api.main as main_module
    monkeypatch.setattr(main_module, 'TASKS_DIR', test_tasks_dir)
    
    # Create a test task
    task_id = "test-task-123"
    task_path = create_test_task(task_id, test_tasks_dir)
    
    # Add task to in-memory status
    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0
    }
    
    # Verify task exists before deletion
    assert task_path.exists()
    assert task_id in tasks_status
    assert (task_path / "status.json").exists()
    assert (task_path / "config.json").exists()
    assert (task_path / "checkpoint.json").exists()
    assert (task_path / "experiment_dataset.pkl").exists()
    
    # Delete the task
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == f"Evaluation task '{task_id}' deleted successfully"
    assert data["task_id"] == task_id
    assert data["deleted_from_memory"] is True
    assert data["deleted_from_disk"] is True
    
    # Verify task is deleted from memory
    assert task_id not in tasks_status
    
    # Verify task directory is deleted from disk
    assert not task_path.exists()
    
    # Cleanup
    if task_id in tasks_status:
        del tasks_status[task_id]


def test_delete_task_not_found(client, monkeypatch, tmp_path):
    """Test deletion of non-existent task returns 404"""
    # Setup: Create a temporary tasks directory
    test_tasks_dir = tmp_path / "tasks"
    test_tasks_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch TASKS_DIR to use temp directory
    import rag_benchmark.api.main as main_module
    monkeypatch.setattr(main_module, 'TASKS_DIR', test_tasks_dir)
    
    # Try to delete non-existent task
    task_id = "non-existent-task"
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "Evaluation task not found"
    assert task_id in data["detail"]["details"]
    assert "action" in data["detail"]


def test_delete_task_only_in_memory(client, monkeypatch, tmp_path):
    """Test deletion of task that exists only in memory"""
    # Setup: Create a temporary tasks directory
    test_tasks_dir = tmp_path / "tasks"
    test_tasks_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch TASKS_DIR to use temp directory
    import rag_benchmark.api.main as main_module
    monkeypatch.setattr(main_module, 'TASKS_DIR', test_tasks_dir)
    
    # Create task only in memory (not on disk)
    task_id = "memory-only-task"
    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "running",
        "progress": 0.5
    }
    
    # Verify task doesn't exist on disk
    task_path = test_tasks_dir / task_id
    assert not task_path.exists()
    
    # Delete the task
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_from_memory"] is True
    assert data["deleted_from_disk"] is False
    
    # Verify task is deleted from memory
    assert task_id not in tasks_status
    
    # Cleanup
    if task_id in tasks_status:
        del tasks_status[task_id]


def test_delete_task_only_on_disk(client, monkeypatch, tmp_path):
    """Test deletion of task that exists only on disk"""
    # Setup: Create a temporary tasks directory
    test_tasks_dir = tmp_path / "tasks"
    test_tasks_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch TASKS_DIR to use temp directory
    import rag_benchmark.api.main as main_module
    monkeypatch.setattr(main_module, 'TASKS_DIR', test_tasks_dir)
    
    # Create task only on disk (not in memory)
    task_id = "disk-only-task"
    task_path = create_test_task(task_id, test_tasks_dir)
    
    # Verify task exists on disk but not in memory
    assert task_path.exists()
    assert task_id not in tasks_status
    
    # Delete the task
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_from_memory"] is False
    assert data["deleted_from_disk"] is True
    
    # Verify task directory is deleted from disk
    assert not task_path.exists()


def test_delete_task_removes_all_files(client, monkeypatch, tmp_path):
    """Test that deletion removes all task-related files"""
    # Setup: Create a temporary tasks directory
    test_tasks_dir = tmp_path / "tasks"
    test_tasks_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch TASKS_DIR to use temp directory
    import rag_benchmark.api.main as main_module
    monkeypatch.setattr(main_module, 'TASKS_DIR', test_tasks_dir)
    
    # Create a test task with multiple files
    task_id = "multi-file-task"
    task_path = create_test_task(task_id, test_tasks_dir)
    
    # Add some additional files
    with open(task_path / "results.json", 'w') as f:
        json.dump({"test": "data"}, f)
    
    with open(task_path / "extra_file.txt", 'w') as f:
        f.write("extra data")
    
    # Verify all files exist
    assert (task_path / "status.json").exists()
    assert (task_path / "config.json").exists()
    assert (task_path / "checkpoint.json").exists()
    assert (task_path / "experiment_dataset.pkl").exists()
    assert (task_path / "results.json").exists()
    assert (task_path / "extra_file.txt").exists()
    
    # Delete the task
    response = client.delete(f"/evaluate/delete/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    
    # Verify entire directory is deleted
    assert not task_path.exists()
    assert not (task_path / "status.json").exists()
    assert not (task_path / "config.json").exists()
    assert not (task_path / "checkpoint.json").exists()
    assert not (task_path / "experiment_dataset.pkl").exists()
    assert not (task_path / "results.json").exists()
    assert not (task_path / "extra_file.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
