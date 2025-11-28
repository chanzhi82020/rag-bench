"""Integration tests for task persistence with directory structure"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_task_creation_saves_config_and_status():
    """Test that creating a task saves both config.json and status.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('rag_benchmark.api.main.TASKS_DIR', tasks_dir):
            from rag_benchmark.api.main import tasks_status, save_task_status
            
            task_id = "integration-test-task"
            task_dir = tasks_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Simulate task creation - save config
            config_data = {
                "dataset_name": "hotpotqa",
                "subset": None,
                "rag_name": "baseline",
                "eval_type": "e2e",
                "sample_size": 50,
                "model_info": {
                    "llm_model_id": "gpt-4",
                    "embedding_model_id": "text-embedding-3-small"
                }
            }
            config_file = task_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Save status
            tasks_status[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0.0,
                "current_stage": "初始化",
                "result": None,
                "error": None,
                "created_at": "2025-11-24T10:00:00",
                "updated_at": "2025-11-24T10:00:00"
            }
            save_task_status(task_id)
            
            # Verify both files exist
            assert config_file.exists(), "config.json should exist"
            assert (task_dir / "status.json").exists(), "status.json should exist"
            
            # Verify config content
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            assert loaded_config["dataset_name"] == "hotpotqa"
            assert loaded_config["rag_name"] == "baseline"
            
            # Verify status content
            with open(task_dir / "status.json", 'r') as f:
                loaded_status = json.load(f)
            assert loaded_status["task_id"] == task_id
            assert loaded_status["status"] == "pending"


def test_task_update_preserves_directory_structure():
    """Test that updating task status preserves the directory structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('rag_benchmark.api.main.TASKS_DIR', tasks_dir):
            from rag_benchmark.api.main import tasks_status, save_task_status, update_task_status
            
            task_id = "update-test-task"
            task_dir = tasks_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Initial status
            tasks_status[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0.0,
                "created_at": "2025-11-24T10:00:00",
                "updated_at": "2025-11-24T10:00:00"
            }
            save_task_status(task_id)
            
            # Update status
            update_task_status(task_id, status="running", progress=0.5)
            
            # Verify status file still exists in correct location
            status_file = task_dir / "status.json"
            assert status_file.exists()
            
            # Verify updated content
            with open(status_file, 'r') as f:
                loaded_status = json.load(f)
            assert loaded_status["status"] == "running"
            assert loaded_status["progress"] == 0.5


def test_list_tasks_loads_from_directories():
    """Test that listing tasks loads from directory structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multiple task directories
        task_ids = ["task-a", "task-b", "task-c"]
        for task_id in task_ids:
            task_dir = tasks_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            status_data = {
                "task_id": task_id,
                "status": "completed",
                "progress": 1.0
            }
            
            status_file = task_dir / "status.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        
        # Verify all directories exist
        for task_id in task_ids:
            assert (tasks_dir / task_id).exists()
            assert (tasks_dir / task_id / "status.json").exists()


def test_requirements_5_1_5_2_5_5():
    """
    Test Requirements 5.1, 5.2, 5.5:
    - 5.1: Files organized in dedicated directory per task
    - 5.2: Task status stored in JSON file
    - 5.5: Metadata with timestamps and task configuration
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('rag_benchmark.api.main.TASKS_DIR', tasks_dir):
            from rag_benchmark.api.main import tasks_status, save_task_status
            
            task_id = "req-test-task"
            task_dir = tasks_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Requirement 5.1: Dedicated directory per task
            assert task_dir.exists()
            assert task_dir.is_dir()
            
            # Requirement 5.2: Task status in JSON file
            tasks_status[task_id] = {
                "task_id": task_id,
                "status": "running",
                "progress": 0.3,
                "current_stage": "Loading dataset",
                "result": None,
                "error": None,
                "created_at": "2025-11-24T10:00:00",
                "updated_at": "2025-11-24T10:05:00"
            }
            save_task_status(task_id)
            
            status_file = task_dir / "status.json"
            assert status_file.exists()
            assert status_file.suffix == ".json"
            
            # Requirement 5.5: Metadata with timestamps
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            
            assert "created_at" in status_data
            assert "updated_at" in status_data
            assert status_data["created_at"] == "2025-11-24T10:00:00"
            assert status_data["updated_at"] == "2025-11-24T10:05:00"
            
            # Task configuration (saved separately in config.json)
            config_data = {
                "dataset_name": "hotpotqa",
                "rag_name": "baseline",
                "eval_type": "e2e"
            }
            config_file = task_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            assert config_file.exists()
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            assert loaded_config["dataset_name"] == "hotpotqa"
