"""Unit tests for sample info tracking in evaluation tasks"""

import pytest
from rag_benchmark.api.main import SampleInfo, SampleSelection


def test_sample_info_model_creation():
    """Test SampleInfo model can be created with all fields"""
    sample_info = SampleInfo(
        selection_strategy="specific_ids",
        total_samples=100,
        selected_samples=10,
        sample_ids=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        completed_samples=8,
        failed_samples=2
    )
    
    assert sample_info.selection_strategy == "specific_ids"
    assert sample_info.total_samples == 100
    assert sample_info.selected_samples == 10
    assert len(sample_info.sample_ids) == 10
    assert sample_info.completed_samples == 8
    assert sample_info.failed_samples == 2


def test_sample_info_model_without_sample_ids():
    """Test SampleInfo model works without sample_ids (for random/all strategies)"""
    sample_info = SampleInfo(
        selection_strategy="random",
        total_samples=200,
        selected_samples=50,
        sample_ids=None,
        completed_samples=45,
        failed_samples=5
    )
    
    assert sample_info.selection_strategy == "random"
    assert sample_info.total_samples == 200
    assert sample_info.selected_samples == 50
    assert sample_info.sample_ids is None
    assert sample_info.completed_samples == 45
    assert sample_info.failed_samples == 5


def test_sample_info_serialization():
    """Test SampleInfo can be serialized to dict"""
    sample_info = SampleInfo(
        selection_strategy="all",
        total_samples=150,
        selected_samples=150,
        sample_ids=None,
        completed_samples=150,
        failed_samples=0
    )
    
    data = sample_info.model_dump()
    
    assert data["selection_strategy"] == "all"
    assert data["total_samples"] == 150
    assert data["selected_samples"] == 150
    assert data["sample_ids"] is None
    assert data["completed_samples"] == 150
    assert data["failed_samples"] == 0


def test_sample_selection_specific_ids():
    """Test SampleSelection model with specific_ids strategy"""
    selection = SampleSelection(
        strategy="specific_ids",
        sample_ids=["0", "1", "2", "3", "4"]
    )
    
    assert selection.strategy == "specific_ids"
    assert len(selection.sample_ids) == 5
    assert selection.sample_size is None
    assert selection.random_seed is None


def test_sample_selection_random():
    """Test SampleSelection model with random strategy"""
    selection = SampleSelection(
        strategy="random",
        sample_size=20,
        random_seed=42
    )
    
    assert selection.strategy == "random"
    assert selection.sample_ids is None
    assert selection.sample_size == 20
    assert selection.random_seed == 42


def test_sample_selection_all():
    """Test SampleSelection model with all strategy"""
    selection = SampleSelection(
        strategy="all"
    )
    
    assert selection.strategy == "all"
    assert selection.sample_ids is None
    assert selection.sample_size is None
    assert selection.random_seed is None


def test_sample_info_default_values():
    """Test SampleInfo model with default values for completed/failed"""
    sample_info = SampleInfo(
        selection_strategy="specific_ids",
        total_samples=100,
        selected_samples=10
    )
    
    # Default values should be 0
    assert sample_info.completed_samples == 0
    assert sample_info.failed_samples == 0
    assert sample_info.sample_ids is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
