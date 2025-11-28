"""Integration test for GET /evaluate/status/{task_id} endpoint with checkpoint info"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from rag_benchmark.api.main import app, tasks_status, TASKS_DIR, checkpoint_manager


@pytest.fixture
def client():
    """Create a test client"""
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
                import shutil
                shutil.rmtree(task_dir)


def test_get_status_with_checkpoint_info(client, cleanup_tasks):
    """Test GET /evaluate/status/{task_id} returns checkpoint info"""
    task_id = "test-checkpoint-status-123"
    
    # Create task directory and files
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create status.json
    status_data = {
        "task_id": task_id,
        "status": "running",
        "progress": 0.6,
        "current_stage": "运行e2e评测",
        "result": None,
        "error": None,
        "created_at": "2025-11-24T10:00:00",
        "updated_at": "2025-11-24T10:05:00",
        "sample_info": {
            "selection_strategy": "random",
            "total_samples": 100,
            "selected_samples": 50,
            "sample_ids": None,
            "completed_samples": 30,
            "failed_samples": 0
        }
    }
    
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(status_data, f)
    
    # Create checkpoint.json
    checkpoint_data = {
        "completed_stages": ["load_dataset", "prepare_experiment"],
        "current_stage": "run_evaluation",
        "stage_data": {
            "load_dataset": {
                "completed_at": "2025-11-24T10:01:00",
                "dataset_size": 100
            },
            "prepare_experiment": {
                "completed_at": "2025-11-24T10:03:00",
                "experiment_dataset_path": "experiment_dataset.pkl"
            }
        },
        "last_checkpoint_at": "2025-11-24T10:03:00"
    }
    
    checkpoint_file = task_dir / "checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    # Call the endpoint
    response = client.get(f"/evaluate/status/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    
    # Verify basic status
    assert data["task_id"] == task_id
    assert data["status"] == "running"
    assert data["progress"] == 0.6
    
    # Verify sample_info
    assert data["sample_info"] is not None
    assert data["sample_info"]["selection_strategy"] == "random"
    assert data["sample_info"]["total_samples"] == 100
    assert data["sample_info"]["selected_samples"] == 50
    assert data["sample_info"]["completed_samples"] == 30
    
    # Verify checkpoint_info
    assert data["checkpoint_info"] is not None
    assert data["checkpoint_info"]["has_checkpoint"] is True
    assert len(data["checkpoint_info"]["completed_stages"]) == 2
    assert "load_dataset" in data["checkpoint_info"]["completed_stages"]
    assert "prepare_experiment" in data["checkpoint_info"]["completed_stages"]
    assert data["checkpoint_info"]["current_stage"] == "run_evaluation"
    assert data["checkpoint_info"]["last_checkpoint_at"] == "2025-11-24T10:03:00"


def test_get_status_without_checkpoint(client, cleanup_tasks):
    """Test GET /evaluate/status/{task_id} when no checkpoint exists"""
    task_id = "test-no-checkpoint-456"
    
    # Create task directory and status file only (no checkpoint)
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    status_data = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "current_stage": "初始化",
        "result": None,
        "error": None,
        "created_at": "2025-11-24T10:00:00",
        "updated_at": "2025-11-24T10:00:00",
        "sample_info": None
    }
    
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(status_data, f)
    
    # Call the endpoint
    response = client.get(f"/evaluate/status/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    
    # Verify checkpoint_info indicates no checkpoint
    assert data["checkpoint_info"] is not None
    assert data["checkpoint_info"]["has_checkpoint"] is False
    assert len(data["checkpoint_info"]["completed_stages"]) == 0
    assert data["checkpoint_info"]["current_stage"] is None
    assert data["checkpoint_info"]["last_checkpoint_at"] is None


def test_get_status_completed_task_with_checkpoint(client, cleanup_tasks):
    """Test GET /evaluate/status/{task_id} for completed task with checkpoint"""
    task_id = "test-completed-789"
    
    # Create task directory and files
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create status.json for completed task
    status_data = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0,
        "current_stage": "完成",
        "result": {
            "metrics": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.78
            },
            "sample_count": 10
        },
        "error": None,
        "created_at": "2025-11-24T10:00:00",
        "updated_at": "2025-11-24T10:10:00",
        "sample_info": {
            "selection_strategy": "specific_ids",
            "total_samples": 100,
            "selected_samples": 10,
            "sample_ids": ["sample-1", "sample-2"],
            "completed_samples": 10,
            "failed_samples": 0
        }
    }
    
    status_file = task_dir / "status.json"
    with open(status_file, 'w') as f:
        json.dump(status_data, f)
    
    # Create checkpoint.json showing all stages completed
    checkpoint_data = {
        "completed_stages": ["load_dataset", "prepare_experiment", "run_evaluation"],
        "current_stage": "run_evaluation",
        "stage_data": {
            "load_dataset": {
                "completed_at": "2025-11-24T10:01:00"
            },
            "prepare_experiment": {
                "completed_at": "2025-11-24T10:03:00"
            },
            "run_evaluation": {
                "completed_at": "2025-11-24T10:08:00"
            }
        },
        "last_checkpoint_at": "2025-11-24T10:08:00"
    }
    
    checkpoint_file = task_dir / "checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    # Call the endpoint
    response = client.get(f"/evaluate/status/{task_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    
    # Verify task is completed
    assert data["status"] == "completed"
    assert data["progress"] == 1.0
    assert data["result"] is not None
    
    # Verify checkpoint_info shows all stages completed
    assert data["checkpoint_info"] is not None
    assert data["checkpoint_info"]["has_checkpoint"] is True
    assert len(data["checkpoint_info"]["completed_stages"]) == 3
    assert "load_dataset" in data["checkpoint_info"]["completed_stages"]
    assert "prepare_experiment" in data["checkpoint_info"]["completed_stages"]
    assert "run_evaluation" in data["checkpoint_info"]["completed_stages"]


def test_get_status_nonexistent_task(client):
    """Test GET /evaluate/status/{task_id} for non-existent task"""
    response = client.get("/evaluate/status/nonexistent-task-id")
    
    assert response.status_code == 404
    assert "任务不存在" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
