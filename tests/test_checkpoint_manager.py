"""Tests for CheckpointManager module"""

import tempfile
from pathlib import Path

import pytest

from rag_benchmark.api.checkpoint_manager import CheckpointManager


def test_checkpoint_manager_initialization():
    """Test CheckpointManager initialization creates tasks directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        assert manager.tasks_dir == tasks_dir
        assert tasks_dir.exists()


def test_save_and_load_checkpoint():
    """Test saving and loading checkpoint data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-123"
        stage = "load_dataset"
        data = {
            "dataset_size": 100,
            "dataset_name": "hotpotqa"
        }
        
        # Save checkpoint
        manager.save_checkpoint(task_id, stage, data)
        
        # Verify checkpoint file exists
        checkpoint_file = tasks_dir / task_id / "checkpoint.json"
        assert checkpoint_file.exists()
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        
        assert checkpoint is not None
        assert stage in checkpoint["completed_stages"]
        assert checkpoint["current_stage"] == stage
        assert checkpoint["stage_data"][stage]["dataset_size"] == 100
        assert checkpoint["stage_data"][stage]["dataset_name"] == "hotpotqa"
        assert "completed_at" in checkpoint["stage_data"][stage]
        assert checkpoint["last_checkpoint_at"] is not None


def test_save_multiple_stages():
    """Test saving multiple stages to same checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-456"
        
        # Save first stage
        manager.save_checkpoint(task_id, "load_dataset", {"dataset_size": 100})
        
        # Save second stage
        manager.save_checkpoint(task_id, "prepare_experiment", {"experiment_size": 100})
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        
        assert len(checkpoint["completed_stages"]) == 2
        assert "load_dataset" in checkpoint["completed_stages"]
        assert "prepare_experiment" in checkpoint["completed_stages"]
        assert checkpoint["current_stage"] == "prepare_experiment"


def test_load_nonexistent_checkpoint():
    """Test loading checkpoint that doesn't exist returns None"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        checkpoint = manager.load_checkpoint("nonexistent-task")
        assert checkpoint is None


def test_save_and_load_experiment_dataset():
    """Test saving and loading experiment dataset with pickle"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-789"
        
        # Create a mock dataset object
        dataset = {
            "records": [
                {"query": "test1", "answer": "answer1"},
                {"query": "test2", "answer": "answer2"}
            ],
            "metadata": {"size": 2}
        }
        
        # Save dataset
        manager.save_experiment_dataset(task_id, dataset)
        
        # Verify file exists
        dataset_file = tasks_dir / task_id / "experiment_dataset.pkl"
        assert dataset_file.exists()
        
        # Load dataset
        loaded_dataset = manager.load_experiment_dataset(task_id)
        
        assert loaded_dataset is not None
        assert loaded_dataset["metadata"]["size"] == 2
        assert len(loaded_dataset["records"]) == 2
        assert loaded_dataset["records"][0]["query"] == "test1"


def test_load_nonexistent_experiment_dataset():
    """Test loading experiment dataset that doesn't exist raises ValueError"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        with pytest.raises(ValueError, match="No experiment dataset found"):
            manager.load_experiment_dataset("nonexistent-task")


def test_clear_checkpoints():
    """Test clearing checkpoint data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-clear"
        
        # Create checkpoint and dataset
        manager.save_checkpoint(task_id, "load_dataset", {"size": 100})
        manager.save_experiment_dataset(task_id, {"data": "test"})
        
        # Verify files exist
        task_dir = tasks_dir / task_id
        assert (task_dir / "checkpoint.json").exists()
        assert (task_dir / "experiment_dataset.pkl").exists()
        
        # Clear checkpoints
        manager.clear_checkpoints(task_id)
        
        # Verify checkpoint files are removed
        assert not (task_dir / "checkpoint.json").exists()
        assert not (task_dir / "experiment_dataset.pkl").exists()
        
        # Verify task directory still exists
        assert task_dir.exists()


def test_clear_checkpoints_nonexistent_task():
    """Test clearing checkpoints for nonexistent task doesn't raise error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        # Should not raise error
        manager.clear_checkpoints("nonexistent-task")


def test_corrupted_checkpoint_returns_none():
    """Test that corrupted checkpoint file returns None"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-corrupted"
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Write corrupted JSON
        checkpoint_file = task_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            f.write("{ invalid json }")
        
        # Should return None instead of raising error
        checkpoint = manager.load_checkpoint(task_id)
        assert checkpoint is None


def test_corrupted_dataset_returns_none():
    """Test that corrupted dataset file raises ValueError"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-corrupted-dataset"
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Write corrupted pickle
        dataset_file = task_dir / "experiment_dataset.pkl"
        with open(dataset_file, 'wb') as f:
            f.write(b"corrupted pickle data")
        
        # Should raise ValueError for corrupted data
        with pytest.raises(ValueError, match="Corrupted experiment dataset"):
            manager.load_experiment_dataset(task_id)
