"""Tests for task persistence with directory structure"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_save_task_status_creates_directory_structure():
    """Test that save_task_status creates the correct directory structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock the TASKS_DIR and tasks_status
        with patch('rag_benchmark.api.main.TASKS_DIR', tasks_dir):
            from rag_benchmark.api.main import save_task_status, tasks_status
            
            task_id = "test-task-123"
            tasks_status[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0.0,
                "created_at": "2025-11-24T10:00:00",
                "updated_at": "2025-11-24T10:00:00"
            }
            
            # Save task status
            save_task_status(task_id)
            
            # Verify directory structure
            task_dir = tasks_dir / task_id
            assert task_dir.exists(), "Task directory should be created"
            assert task_dir.is_dir(), "Task path should be a directory"
            
            # Verify status.json exists
            status_file = task_dir / "status.json"
            assert status_file.exists(), "status.json should be created"
            
            # Verify content
            with open(status_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["task_id"] == task_id
            assert saved_data["status"] == "pending"
            assert saved_data["progress"] == 0.0


def test_load_task_status_from_directory():
    """Test that load_task_status loads from directory structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create task directory and status file
        task_id = "test-task-456"
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        status_data = {
            "task_id": task_id,
            "status": "completed",
            "progress": 1.0,
            "result": {"metrics": {"accuracy": 0.95}}
        }
        
        status_file = task_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        # Mock the TASKS_DIR
        with patch('rag_benchmark.api.main.TASKS_DIR', tasks_dir):
            from rag_benchmark.api.main import load_task_status
            
            # Load task status
            loaded_data = load_task_status(task_id)
            
            assert loaded_data is not None
            assert loaded_data["task_id"] == task_id
            assert loaded_data["status"] == "completed"
            assert loaded_data["progress"] == 1.0
            assert loaded_data["result"]["metrics"]["accuracy"] == 0.95


def test_save_and_load_task_config():
    """Test that task configuration is saved separately"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        task_id = "test-task-config"
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_data = {
            "dataset_name": "hotpotqa",
            "subset": None,
            "rag_name": "baseline",
            "eval_type": "e2e",
            "sample_size": 100,
            "model_info": {
                "llm_model_id": "gpt-4",
                "embedding_model_id": "text-embedding-3-small"
            }
        }
        
        config_file = task_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Verify config file exists and can be loaded
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["dataset_name"] == "hotpotqa"
        assert loaded_config["rag_name"] == "baseline"
        assert loaded_config["eval_type"] == "e2e"
        assert loaded_config["sample_size"] == 100


def test_task_directory_isolation():
    """Test that multiple tasks have isolated directories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('rag_benchmark.api.main.TASKS_DIR', tasks_dir):
            from rag_benchmark.api.main import save_task_status, tasks_status
            
            # Create multiple tasks
            task_ids = ["task-1", "task-2", "task-3"]
            
            for task_id in task_ids:
                tasks_status[task_id] = {
                    "task_id": task_id,
                    "status": "pending",
                    "progress": 0.0
                }
                save_task_status(task_id)
            
            # Verify each task has its own directory
            for task_id in task_ids:
                task_dir = tasks_dir / task_id
                assert task_dir.exists()
                assert task_dir.is_dir()
                
                status_file = task_dir / "status.json"
                assert status_file.exists()
                
                with open(status_file, 'r') as f:
                    data = json.load(f)
                assert data["task_id"] == task_id
