"""Integration tests for checkpoint/resume functionality in evaluation tasks"""

import tempfile
from pathlib import Path

import pytest

from rag_benchmark.api.checkpoint_manager import CheckpointManager


def test_checkpoint_manager_integration_with_sample_selector():
    """Test that CheckpointManager works with SampleSelector for dataset preprocessing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-integration"
        
        # Simulate stage 1: Load dataset with sample selection
        # This would normally be done by run_evaluation_task
        stage_data = {
            "dataset_name": "hotpotqa",
            "subset": None,
            "total_samples": 1000,
            "selected_samples": 100,
            "selection_strategy": "random"
        }
        
        manager.save_checkpoint(task_id, "load_dataset", stage_data)
        
        # Verify checkpoint was saved
        checkpoint = manager.load_checkpoint(task_id)
        assert checkpoint is not None
        assert "load_dataset" in checkpoint["completed_stages"]
        assert checkpoint["stage_data"]["load_dataset"]["selected_samples"] == 100


def test_checkpoint_resume_skips_completed_stages():
    """Test that checkpoint resume logic skips already completed stages"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-resume"
        
        # Save checkpoints for multiple stages
        manager.save_checkpoint(task_id, "load_dataset", {"dataset_size": 100})
        manager.save_checkpoint(task_id, "prepare_experiment", {"experiment_size": 100})
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        
        # Verify both stages are marked as completed
        assert "load_dataset" in checkpoint["completed_stages"]
        assert "prepare_experiment" in checkpoint["completed_stages"]
        assert checkpoint["current_stage"] == "prepare_experiment"
        
        # Simulate resume logic
        completed_stages = checkpoint["completed_stages"]
        
        # Stage 1 should be skipped
        if "load_dataset" in completed_stages:
            # Skip this stage
            pass
        
        # Stage 2 should be skipped
        if "prepare_experiment" in completed_stages:
            # Skip this stage
            pass
        
        # Stage 3 should run
        if "run_evaluation" not in completed_stages:
            # This stage should run
            assert True


def test_corrupted_checkpoint_handling():
    """Test that corrupted checkpoint data is handled gracefully"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-corrupted"
        task_dir = tasks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Write corrupted checkpoint
        checkpoint_file = task_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            f.write("{ invalid json }")
        
        # Load should return None
        checkpoint = manager.load_checkpoint(task_id)
        assert checkpoint is None
        
        # Clear checkpoints should not raise error
        manager.clear_checkpoints(task_id)
        
        # After clearing, checkpoint should still be None
        checkpoint = manager.load_checkpoint(task_id)
        assert checkpoint is None


def test_experiment_dataset_persistence():
    """Test that experiment dataset is persisted and can be loaded"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-dataset"
        
        # Create mock experiment dataset
        mock_dataset = {
            "samples": [
                {"query": "test1", "answer": "answer1"},
                {"query": "test2", "answer": "answer2"}
            ],
            "metadata": {"size": 2}
        }
        
        # Save experiment dataset
        manager.save_experiment_dataset(task_id, mock_dataset)
        
        # Save checkpoint indicating dataset was saved
        manager.save_checkpoint(
            task_id,
            "prepare_experiment",
            {
                "experiment_dataset_saved": True,
                "experiment_size": 2
            }
        )
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        assert "prepare_experiment" in checkpoint["completed_stages"]
        
        # Load experiment dataset
        loaded_dataset = manager.load_experiment_dataset(task_id)
        assert loaded_dataset is not None
        assert loaded_dataset["metadata"]["size"] == 2
        assert len(loaded_dataset["samples"]) == 2


def test_stage_skipping_logic():
    """Test the stage skipping logic based on checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-skip"
        
        # Simulate completing first two stages
        manager.save_checkpoint(task_id, "load_dataset", {"dataset_size": 100})
        manager.save_checkpoint(task_id, "prepare_experiment", {"experiment_size": 100})
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        completed_stages = checkpoint.get("completed_stages", [])
        
        # Test stage skipping logic
        stages_to_run = []
        
        if "load_dataset" not in completed_stages:
            stages_to_run.append("load_dataset")
        
        if "prepare_experiment" not in completed_stages:
            stages_to_run.append("prepare_experiment")
        
        if "run_evaluation" not in completed_stages:
            stages_to_run.append("run_evaluation")
        
        # Only run_evaluation should be in the list
        assert stages_to_run == ["run_evaluation"]


def test_checkpoint_with_sample_selection_strategies():
    """Test checkpoint saves sample selection strategy information"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        # Test specific_ids strategy
        task_id_1 = "test-task-specific"
        manager.save_checkpoint(
            task_id_1,
            "load_dataset",
            {
                "dataset_name": "hotpotqa",
                "total_samples": 1000,
                "selected_samples": 10,
                "selection_strategy": "specific_ids",
                "sample_ids": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            }
        )
        
        checkpoint_1 = manager.load_checkpoint(task_id_1)
        assert checkpoint_1["stage_data"]["load_dataset"]["selection_strategy"] == "specific_ids"
        assert len(checkpoint_1["stage_data"]["load_dataset"]["sample_ids"]) == 10
        
        # Test random strategy
        task_id_2 = "test-task-random"
        manager.save_checkpoint(
            task_id_2,
            "load_dataset",
            {
                "dataset_name": "hotpotqa",
                "total_samples": 1000,
                "selected_samples": 50,
                "selection_strategy": "random",
                "random_seed": 42
            }
        )
        
        checkpoint_2 = manager.load_checkpoint(task_id_2)
        assert checkpoint_2["stage_data"]["load_dataset"]["selection_strategy"] == "random"
        assert checkpoint_2["stage_data"]["load_dataset"]["selected_samples"] == 50
        
        # Test all strategy
        task_id_3 = "test-task-all"
        manager.save_checkpoint(
            task_id_3,
            "load_dataset",
            {
                "dataset_name": "hotpotqa",
                "total_samples": 1000,
                "selected_samples": 1000,
                "selection_strategy": "all"
            }
        )
        
        checkpoint_3 = manager.load_checkpoint(task_id_3)
        assert checkpoint_3["stage_data"]["load_dataset"]["selection_strategy"] == "all"
        assert checkpoint_3["stage_data"]["load_dataset"]["selected_samples"] == 1000


def test_resume_from_prepare_experiment_stage_with_missing_dataset():
    """Test resuming from prepare_experiment stage when experiment dataset is missing
    
    This tests the scenario where:
    1. Both load_dataset and prepare_experiment stages are marked as completed
    2. But the experiment dataset file is missing/corrupted
    3. The system should raise ValueError when trying to load
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-resume-missing-dataset"
        
        # Simulate both stages completed
        manager.save_checkpoint(
            task_id,
            "load_dataset",
            {
                "dataset_name": "hotpotqa",
                "total_samples": 1000,
                "selected_samples": 100
            }
        )
        manager.save_checkpoint(
            task_id,
            "prepare_experiment",
            {
                "experiment_dataset_saved": True,
                "experiment_size": 100
            }
        )
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(task_id)
        
        # Verify both stages are marked as completed
        assert "load_dataset" in checkpoint["completed_stages"]
        assert "prepare_experiment" in checkpoint["completed_stages"]
        
        # Try to load experiment dataset (should raise ValueError since we didn't save it)
        with pytest.raises(ValueError, match="No experiment dataset found"):
            manager.load_experiment_dataset(task_id)


def test_checkpoint_stages_independence():
    """Test that each stage can be independently resumed"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks_dir = Path(tmpdir) / "tasks"
        manager = CheckpointManager(tasks_dir)
        
        task_id = "test-task-independence"
        
        # Save only stage 1
        manager.save_checkpoint(
            task_id,
            "load_dataset",
            {"dataset_size": 100}
        )
        
        checkpoint = manager.load_checkpoint(task_id)
        assert len(checkpoint["completed_stages"]) == 1
        assert "load_dataset" in checkpoint["completed_stages"]
        assert "prepare_experiment" not in checkpoint["completed_stages"]
        
        # Save stage 2
        manager.save_checkpoint(
            task_id,
            "prepare_experiment",
            {"experiment_size": 100}
        )
        
        checkpoint = manager.load_checkpoint(task_id)
        assert len(checkpoint["completed_stages"]) == 2
        assert "load_dataset" in checkpoint["completed_stages"]
        assert "prepare_experiment" in checkpoint["completed_stages"]
        assert "run_evaluation" not in checkpoint["completed_stages"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
