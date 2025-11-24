"""Tests for handling invalid float values in API responses"""

import pytest
import math
import json
from fastapi.testclient import TestClient
from rag_benchmark.api.main import app, tasks_status, sanitize_float_values


client = TestClient(app)


def test_sanitize_float_values_with_inf():
    """Test that infinity values are converted to None"""
    data = {"value": float('inf')}
    result = sanitize_float_values(data)
    assert result == {"value": None}


def test_sanitize_float_values_with_neg_inf():
    """Test that negative infinity values are converted to None"""
    data = {"value": float('-inf')}
    result = sanitize_float_values(data)
    assert result == {"value": None}


def test_sanitize_float_values_with_nan():
    """Test that NaN values are converted to None"""
    data = {"value": float('nan')}
    result = sanitize_float_values(data)
    assert result == {"value": None}


def test_sanitize_float_values_with_valid_floats():
    """Test that valid float values are preserved"""
    data = {"value": 1.5, "another": 0.0, "negative": -3.14}
    result = sanitize_float_values(data)
    assert result == data


def test_sanitize_float_values_nested_dict():
    """Test sanitization of nested dictionaries"""
    data = {
        "metrics": {
            "accuracy": 0.95,
            "precision": float('inf'),
            "recall": float('nan')
        }
    }
    result = sanitize_float_values(data)
    assert result == {
        "metrics": {
            "accuracy": 0.95,
            "precision": None,
            "recall": None
        }
    }


def test_sanitize_float_values_list():
    """Test sanitization of lists"""
    data = [1.0, 2.0, float('inf'), float('nan'), -3.5]
    result = sanitize_float_values(data)
    assert result == [1.0, 2.0, None, None, -3.5]


def test_sanitize_float_values_complex_structure():
    """Test sanitization of complex nested structures"""
    data = {
        "result": {
            "metrics": {"score": float('inf')},
            "detailed_results": [
                {"value": 1.0, "status": "ok"},
                {"value": float('nan'), "status": "failed"}
            ]
        },
        "count": 2
    }
    result = sanitize_float_values(data)
    assert result == {
        "result": {
            "metrics": {"score": None},
            "detailed_results": [
                {"value": 1.0, "status": "ok"},
                {"value": None, "status": "failed"}
            ]
        },
        "count": 2
    }


def test_sanitize_float_values_json_serializable():
    """Test that sanitized data can be JSON serialized"""
    data = {
        "metrics": {
            "accuracy": 0.95,
            "precision": float('inf'),
            "recall": float('nan')
        }
    }
    sanitized = sanitize_float_values(data)
    
    # Should not raise an exception
    json_str = json.dumps(sanitized)
    assert json_str is not None
    
    # Verify the deserialized data
    deserialized = json.loads(json_str)
    assert deserialized["metrics"]["accuracy"] == 0.95
    assert deserialized["metrics"]["precision"] is None
    assert deserialized["metrics"]["recall"] is None


def test_list_tasks_with_invalid_floats(tmp_path):
    """Test that /evaluate/tasks endpoint handles invalid floats correctly"""
    # Create a task with invalid float values
    task_id = "test-task-with-invalid-floats"
    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "completed",
        "progress": 1.0,
        "result": {
            "metrics": {
                "accuracy": 0.95,
                "precision": float('inf'),
                "recall": float('nan')
            },
            "detailed_results": [
                {"score": 1.0},
                {"score": float('nan')}
            ]
        }
    }
    
    # Request should not fail with JSON serialization error
    response = client.get("/evaluate/tasks")
    assert response.status_code == 200
    
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= 1
    
    # Find our test task
    test_task = None
    for task in data["tasks"]:
        if task["task_id"] == task_id:
            test_task = task
            break
    
    assert test_task is not None
    assert test_task["result"]["metrics"]["accuracy"] == 0.95
    assert test_task["result"]["metrics"]["precision"] is None
    assert test_task["result"]["metrics"]["recall"] is None
    assert test_task["result"]["detailed_results"][0]["score"] == 1.0
    assert test_task["result"]["detailed_results"][1]["score"] is None
    
    # Cleanup
    del tasks_status[task_id]


def test_sanitize_preserves_other_types():
    """Test that sanitization preserves non-float types"""
    data = {
        "string": "hello",
        "int": 42,
        "bool": True,
        "none": None,
        "list": [1, "two", 3.0],
        "nested": {
            "key": "value"
        }
    }
    result = sanitize_float_values(data)
    assert result == data
