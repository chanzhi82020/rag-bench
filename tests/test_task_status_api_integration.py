"""Integration test to verify TaskStatus API works with sample_info"""

import json

import pytest

from rag_benchmark.api.main import SampleInfo, TaskStatus


def test_api_response_format_with_sample_info():
    """Test that TaskStatus can be used in API responses with sample_info"""
    # Create a TaskStatus with sample_info as would be done in the API
    sample_info = SampleInfo(
        selection_strategy="specific_ids",
        total_samples=100,
        selected_samples=10,
        sample_ids=["sample-1", "sample-2", "sample-3"],
        completed_samples=8,
        failed_samples=2
    )
    
    task_status = TaskStatus(
        task_id="api-test-123",
        status="completed",
        progress=1.0,
        current_stage="完成",
        result={
            "metrics": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.78
            },
            "sample_count": 10
        },
        error=None,
        created_at="2025-11-24T10:00:00",
        updated_at="2025-11-24T10:10:00",
        sample_info=sample_info
    )
    
    # Simulate API response serialization
    response_dict = task_status.model_dump()
    
    # Verify the response has all expected fields
    assert response_dict["task_id"] == "api-test-123"
    assert response_dict["status"] == "completed"
    assert response_dict["sample_info"] is not None
    assert response_dict["sample_info"]["selection_strategy"] == "specific_ids"
    assert response_dict["sample_info"]["total_samples"] == 100
    assert response_dict["sample_info"]["selected_samples"] == 10
    assert response_dict["sample_info"]["completed_samples"] == 8
    assert response_dict["sample_info"]["failed_samples"] == 2
    assert len(response_dict["sample_info"]["sample_ids"]) == 3
    
    # Verify it can be JSON serialized (as FastAPI would do)
    json_str = json.dumps(response_dict)
    assert json_str is not None
    
    # Verify it can be deserialized back
    parsed = json.loads(json_str)
    assert parsed["sample_info"]["selection_strategy"] == "specific_ids"


def test_api_response_format_without_sample_info():
    """Test backward compatibility - TaskStatus without sample_info"""
    task_status = TaskStatus(
        task_id="api-test-456",
        status="running",
        progress=0.5,
        current_stage="运行评测",
        result=None,
        error=None,
        created_at="2025-11-24T10:00:00",
        updated_at="2025-11-24T10:05:00"
    )
    
    # Simulate API response serialization
    response_dict = task_status.model_dump()
    
    # Verify sample_info is None (backward compatibility)
    assert response_dict["sample_info"] is None
    
    # Verify it can be JSON serialized
    json_str = json.dumps(response_dict)
    assert json_str is not None


def test_get_evaluation_status_endpoint_format():
    """Test the format that would be returned by GET /evaluate/status/{task_id}"""
    # Simulate what the endpoint would return
    task_status_dict = {
        "task_id": "endpoint-test-789",
        "status": "running",
        "progress": 0.7,
        "current_stage": "运行e2e评测",
        "result": None,
        "error": None,
        "created_at": "2025-11-24T10:00:00",
        "updated_at": "2025-11-24T10:07:00",
        "sample_info": {
            "selection_strategy": "random",
            "total_samples": 200,
            "selected_samples": 50,
            "sample_ids": None,
            "completed_samples": 35,
            "failed_samples": 0
        }
    }
    
    # Verify it can be converted to TaskStatus model
    task_status = TaskStatus(**task_status_dict)
    
    # Verify all fields are correct
    assert task_status.task_id == "endpoint-test-789"
    assert task_status.status == "running"
    assert task_status.sample_info is not None
    assert task_status.sample_info.selection_strategy == "random"
    assert task_status.sample_info.selected_samples == 50
    assert task_status.sample_info.completed_samples == 35
    
    # Verify it can be serialized for API response
    response = task_status.model_dump()
    assert response["sample_info"]["selection_strategy"] == "random"


def test_task_status_with_checkpoint_info():
    """Test that TaskStatus includes checkpoint_info"""
    from rag_benchmark.api.main import CheckpointInfo
    
    # Create checkpoint info
    checkpoint_info = CheckpointInfo(
        has_checkpoint=True,
        completed_stages=["load_dataset", "prepare_experiment"],
        current_stage="run_evaluation",
        last_checkpoint_at="2025-11-24T10:05:00"
    )
    
    # Create task status with checkpoint info
    task_status = TaskStatus(
        task_id="checkpoint-test-123",
        status="running",
        progress=0.6,
        current_stage="运行e2e评测",
        result=None,
        error=None,
        created_at="2025-11-24T10:00:00",
        updated_at="2025-11-24T10:05:00",
        sample_info=None,
        checkpoint_info=checkpoint_info
    )
    
    # Verify checkpoint info is included
    response_dict = task_status.model_dump()
    assert response_dict["checkpoint_info"] is not None
    assert response_dict["checkpoint_info"]["has_checkpoint"] is True
    assert len(response_dict["checkpoint_info"]["completed_stages"]) == 2
    assert "load_dataset" in response_dict["checkpoint_info"]["completed_stages"]
    assert "prepare_experiment" in response_dict["checkpoint_info"]["completed_stages"]
    assert response_dict["checkpoint_info"]["current_stage"] == "run_evaluation"
    assert response_dict["checkpoint_info"]["last_checkpoint_at"] == "2025-11-24T10:05:00"
    
    # Verify JSON serialization
    json_str = json.dumps(response_dict)
    assert json_str is not None
    
    # Verify deserialization
    parsed = json.loads(json_str)
    assert parsed["checkpoint_info"]["has_checkpoint"] is True
    assert len(parsed["checkpoint_info"]["completed_stages"]) == 2


def test_task_status_without_checkpoint_info():
    """Test backward compatibility - TaskStatus without checkpoint_info"""
    task_status = TaskStatus(
        task_id="no-checkpoint-test-456",
        status="pending",
        progress=0.0,
        current_stage="初始化",
        result=None,
        error=None,
        created_at="2025-11-24T10:00:00",
        updated_at="2025-11-24T10:00:00"
    )
    
    # Verify checkpoint_info is None (backward compatibility)
    response_dict = task_status.model_dump()
    assert response_dict["checkpoint_info"] is None
    
    # Verify JSON serialization
    json_str = json.dumps(response_dict)
    assert json_str is not None


def test_task_status_with_no_checkpoint():
    """Test TaskStatus with checkpoint_info indicating no checkpoint exists"""
    from rag_benchmark.api.main import CheckpointInfo
    
    # Create checkpoint info for task with no checkpoint
    checkpoint_info = CheckpointInfo(
        has_checkpoint=False,
        completed_stages=[],
        current_stage=None,
        last_checkpoint_at=None
    )
    
    task_status = TaskStatus(
        task_id="no-checkpoint-data-789",
        status="pending",
        progress=0.0,
        current_stage="初始化",
        result=None,
        error=None,
        created_at="2025-11-24T10:00:00",
        updated_at="2025-11-24T10:00:00",
        checkpoint_info=checkpoint_info
    )
    
    # Verify checkpoint info indicates no checkpoint
    response_dict = task_status.model_dump()
    assert response_dict["checkpoint_info"] is not None
    assert response_dict["checkpoint_info"]["has_checkpoint"] is False
    assert len(response_dict["checkpoint_info"]["completed_stages"]) == 0
    assert response_dict["checkpoint_info"]["current_stage"] is None
    assert response_dict["checkpoint_info"]["last_checkpoint_at"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
