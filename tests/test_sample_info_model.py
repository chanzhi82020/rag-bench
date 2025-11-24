"""Tests for SampleInfo and TaskStatus models with sample tracking"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from rag_benchmark.api.main import SampleInfo, TaskStatus




class TestSampleInfoModel:
    """Test SampleInfo Pydantic model"""
    
    def test_sample_info_creation_with_all_fields(self):
        """Test creating SampleInfo with all fields"""
        sample_info = SampleInfo(
            selection_strategy="specific_ids",
            total_samples=100,
            selected_samples=10,
            sample_ids=["id1", "id2", "id3"],
            completed_samples=8,
            failed_samples=2
        )
        
        assert sample_info.selection_strategy == "specific_ids"
        assert sample_info.total_samples == 100
        assert sample_info.selected_samples == 10
        assert sample_info.sample_ids == ["id1", "id2", "id3"]
        assert sample_info.completed_samples == 8
        assert sample_info.failed_samples == 2
    
    def test_sample_info_creation_with_optional_fields(self):
        """Test creating SampleInfo with optional fields as None"""
        sample_info = SampleInfo(
            selection_strategy="all",
            total_samples=100,
            selected_samples=100,
            sample_ids=None,
            completed_samples=100,
            failed_samples=0
        )
        
        assert sample_info.selection_strategy == "all"
        assert sample_info.total_samples == 100
        assert sample_info.selected_samples == 100
        assert sample_info.sample_ids is None
        assert sample_info.completed_samples == 100
        assert sample_info.failed_samples == 0
    
    def test_sample_info_default_values(self):
        """Test SampleInfo default values for completed and failed samples"""
        sample_info = SampleInfo(
            selection_strategy="random",
            total_samples=100,
            selected_samples=50
        )
        
        assert sample_info.completed_samples == 0
        assert sample_info.failed_samples == 0
    
    def test_sample_info_serialization(self):
        """Test SampleInfo can be serialized to dict"""
        sample_info = SampleInfo(
            selection_strategy="specific_ids",
            total_samples=100,
            selected_samples=10,
            sample_ids=["id1", "id2"],
            completed_samples=5,
            failed_samples=1
        )
        
        data = sample_info.model_dump()
        
        assert data["selection_strategy"] == "specific_ids"
        assert data["total_samples"] == 100
        assert data["selected_samples"] == 10
        assert data["sample_ids"] == ["id1", "id2"]
        assert data["completed_samples"] == 5
        assert data["failed_samples"] == 1
    
    def test_sample_info_deserialization(self):
        """Test SampleInfo can be deserialized from dict"""
        data = {
            "selection_strategy": "random",
            "total_samples": 200,
            "selected_samples": 50,
            "sample_ids": None,
            "completed_samples": 45,
            "failed_samples": 5
        }
        
        sample_info = SampleInfo(**data)
        
        assert sample_info.selection_strategy == "random"
        assert sample_info.total_samples == 200
        assert sample_info.selected_samples == 50
        assert sample_info.sample_ids is None
        assert sample_info.completed_samples == 45
        assert sample_info.failed_samples == 5
    
    def test_sample_info_validation_negative_values(self):
        """Test SampleInfo validates non-negative values"""
        with pytest.raises(Exception):  # Pydantic validation error
            SampleInfo(
                selection_strategy="all",
                total_samples=-1,  # Invalid
                selected_samples=100
            )
        
        with pytest.raises(Exception):  # Pydantic validation error
            SampleInfo(
                selection_strategy="all",
                total_samples=100,
                selected_samples=-1  # Invalid
            )


class TestTaskStatusWithSampleInfo:
    """Test TaskStatus model with sample_info field"""
    
    def test_task_status_without_sample_info(self):
        """Test TaskStatus can be created without sample_info (backward compatibility)"""
        task_status = TaskStatus(
            task_id="test-123",
            status="pending",
            progress=0.0,
            current_stage="初始化",
            result=None,
            error=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        assert task_status.task_id == "test-123"
        assert task_status.status == "pending"
        assert task_status.sample_info is None
    
    def test_task_status_with_sample_info(self):
        """Test TaskStatus can be created with sample_info"""
        sample_info = SampleInfo(
            selection_strategy="specific_ids",
            total_samples=100,
            selected_samples=10,
            sample_ids=["id1", "id2"],
            completed_samples=5,
            failed_samples=1
        )
        
        task_status = TaskStatus(
            task_id="test-456",
            status="running",
            progress=0.5,
            current_stage="运行评测",
            result=None,
            error=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            sample_info=sample_info
        )
        
        assert task_status.task_id == "test-456"
        assert task_status.status == "running"
        assert task_status.sample_info is not None
        assert task_status.sample_info.selection_strategy == "specific_ids"
        assert task_status.sample_info.selected_samples == 10
    
    def test_task_status_serialization_with_sample_info(self):
        """Test TaskStatus with sample_info can be serialized to dict"""
        sample_info = SampleInfo(
            selection_strategy="random",
            total_samples=200,
            selected_samples=50,
            completed_samples=40,
            failed_samples=10
        )
        
        task_status = TaskStatus(
            task_id="test-789",
            status="completed",
            progress=1.0,
            current_stage="完成",
            result={"metrics": {"accuracy": 0.95}},
            error=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            sample_info=sample_info
        )
        
        data = task_status.model_dump()
        
        assert data["task_id"] == "test-789"
        assert data["sample_info"] is not None
        assert data["sample_info"]["selection_strategy"] == "random"
        assert data["sample_info"]["selected_samples"] == 50
        assert data["sample_info"]["completed_samples"] == 40
    
    def test_task_status_deserialization_with_sample_info(self):
        """Test TaskStatus with sample_info can be deserialized from dict"""
        data = {
            "task_id": "test-abc",
            "status": "running",
            "progress": 0.7,
            "current_stage": "评测中",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "sample_info": {
                "selection_strategy": "all",
                "total_samples": 150,
                "selected_samples": 150,
                "sample_ids": None,
                "completed_samples": 100,
                "failed_samples": 5
            }
        }
        
        task_status = TaskStatus(**data)
        
        assert task_status.task_id == "test-abc"
        assert task_status.sample_info is not None
        assert task_status.sample_info.selection_strategy == "all"
        assert task_status.sample_info.total_samples == 150
        assert task_status.sample_info.completed_samples == 100
    
    def test_task_status_json_round_trip(self):
        """Test TaskStatus can be serialized to JSON and back"""
        sample_info = SampleInfo(
            selection_strategy="specific_ids",
            total_samples=100,
            selected_samples=10,
            sample_ids=["id1", "id2", "id3"],
            completed_samples=8,
            failed_samples=2
        )
        
        original_status = TaskStatus(
            task_id="test-json",
            status="completed",
            progress=1.0,
            current_stage="完成",
            result={"metrics": {"score": 0.9}},
            error=None,
            created_at="2025-11-24T10:00:00",
            updated_at="2025-11-24T10:10:00",
            sample_info=sample_info
        )
        
        # Serialize to JSON
        json_str = original_status.model_dump_json()
        
        # Deserialize from JSON
        data = json.loads(json_str)
        restored_status = TaskStatus(**data)
        
        # Verify all fields match
        assert restored_status.task_id == original_status.task_id
        assert restored_status.status == original_status.status
        assert restored_status.sample_info.selection_strategy == original_status.sample_info.selection_strategy
        assert restored_status.sample_info.selected_samples == original_status.sample_info.selected_samples
        assert restored_status.sample_info.sample_ids == original_status.sample_info.sample_ids


class TestTaskStatusPersistence:
    """Test task status persistence with sample_info"""
    
    def test_save_and_load_task_status_with_sample_info(self):
        """Test saving and loading task status with sample_info to/from disk"""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "test-task-123"
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Create task status with sample_info
            task_status_dict = {
                "task_id": "test-task-123",
                "status": "running",
                "progress": 0.5,
                "current_stage": "评测中",
                "result": None,
                "error": None,
                "created_at": "2025-11-24T10:00:00",
                "updated_at": "2025-11-24T10:05:00",
                "sample_info": {
                    "selection_strategy": "specific_ids",
                    "total_samples": 100,
                    "selected_samples": 10,
                    "sample_ids": ["id1", "id2", "id3"],
                    "completed_samples": 5,
                    "failed_samples": 0
                }
            }
            
            # Save to disk
            status_file = task_dir / "status.json"
            with open(status_file, 'w') as f:
                json.dump(task_status_dict, f, indent=2)
            
            # Load from disk
            with open(status_file, 'r') as f:
                loaded_dict = json.load(f)
            
            # Verify data matches
            assert loaded_dict["task_id"] == "test-task-123"
            assert loaded_dict["sample_info"]["selection_strategy"] == "specific_ids"
            assert loaded_dict["sample_info"]["selected_samples"] == 10
            assert loaded_dict["sample_info"]["sample_ids"] == ["id1", "id2", "id3"]
            
            # Verify it can be converted to TaskStatus model
            task_status = TaskStatus(**loaded_dict)
            assert task_status.task_id == "test-task-123"
            assert task_status.sample_info.selection_strategy == "specific_ids"
            assert task_status.sample_info.selected_samples == 10
    
    def test_backward_compatibility_without_sample_info(self):
        """Test loading old task status without sample_info field"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "test-task-old"
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Create old-style task status without sample_info
            old_task_status = {
                "task_id": "test-task-old",
                "status": "completed",
                "progress": 1.0,
                "current_stage": "完成",
                "result": {"metrics": {"score": 0.85}},
                "error": None,
                "created_at": "2025-11-24T09:00:00",
                "updated_at": "2025-11-24T09:10:00"
            }
            
            # Save to disk
            status_file = task_dir / "status.json"
            with open(status_file, 'w') as f:
                json.dump(old_task_status, f, indent=2)
            
            # Load from disk
            with open(status_file, 'r') as f:
                loaded_dict = json.load(f)
            
            # Verify it can be converted to TaskStatus model (sample_info should be None)
            task_status = TaskStatus(**loaded_dict)
            assert task_status.task_id == "test-task-old"
            assert task_status.status == "completed"
            assert task_status.sample_info is None  # Should be None for backward compatibility


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
