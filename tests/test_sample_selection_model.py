"""Tests for SampleSelection and EvaluateRequest models"""

import pytest
from pydantic import ValidationError

from rag_benchmark.api.main import SampleSelection, EvaluateRequest, ModelConfig


def test_sample_selection_specific_ids():
    """Test SampleSelection with specific_ids strategy"""
    selection = SampleSelection(
        strategy="specific_ids",
        sample_ids=["0", "1", "2"]
    )
    assert selection.strategy == "specific_ids"
    assert selection.sample_ids == ["0", "1", "2"]
    assert selection.sample_size is None
    assert selection.random_seed is None


def test_sample_selection_random():
    """Test SampleSelection with random strategy"""
    selection = SampleSelection(
        strategy="random",
        sample_size=10,
        random_seed=42
    )
    assert selection.strategy == "random"
    assert selection.sample_size == 10
    assert selection.random_seed == 42
    assert selection.sample_ids is None


def test_sample_selection_all():
    """Test SampleSelection with all strategy"""
    selection = SampleSelection(strategy="all")
    assert selection.strategy == "all"
    assert selection.sample_ids is None
    assert selection.sample_size is None
    assert selection.random_seed is None


def test_sample_selection_invalid_sample_size():
    """Test SampleSelection rejects invalid sample_size"""
    with pytest.raises(ValidationError):
        SampleSelection(strategy="random", sample_size=0)
    
    with pytest.raises(ValidationError):
        SampleSelection(strategy="random", sample_size=-1)


def test_evaluate_request_with_sample_selection():
    """Test EvaluateRequest with sample_selection field"""
    model_config = ModelConfig(
        llm_model_id="gpt-4",
        embedding_model_id="text-embedding-3-small"
    )
    
    sample_selection = SampleSelection(
        strategy="specific_ids",
        sample_ids=["0", "1", "2"]
    )
    
    request = EvaluateRequest(
        dataset_name="hotpotqa",
        rag_name="baseline",
        eval_type="e2e",
        model_info=model_config,
        sample_selection=sample_selection
    )
    
    assert request.dataset_name == "hotpotqa"
    assert request.rag_name == "baseline"
    assert request.eval_type == "e2e"
    assert request.sample_selection is not None
    assert request.sample_selection.strategy == "specific_ids"
    assert request.sample_selection.sample_ids == ["0", "1", "2"]


def test_evaluate_request_without_sample_selection():
    """Test EvaluateRequest without sample_selection (backward compatibility)"""
    model_config = ModelConfig(
        llm_model_id="gpt-4",
        embedding_model_id="text-embedding-3-small"
    )
    
    request = EvaluateRequest(
        dataset_name="hotpotqa",
        rag_name="baseline",
        eval_type="e2e",
        model_info=model_config
    )
    
    assert request.dataset_name == "hotpotqa"
    assert request.sample_selection is None


def test_evaluate_request_with_both_sample_size_and_selection():
    """Test EvaluateRequest with sample_selection"""
    model_config = ModelConfig(
        llm_model_id="gpt-4",
        embedding_model_id="text-embedding-3-small"
    )
    
    sample_selection = SampleSelection(
        strategy="random",
        sample_size=20
    )
    
    # sample_selection is the new way to specify sampling
    request = EvaluateRequest(
        dataset_name="hotpotqa",
        rag_name="baseline",
        eval_type="e2e",
        model_info=model_config,
        sample_selection=sample_selection
    )
    
    assert request.sample_selection is not None
    assert request.sample_selection.sample_size == 20
    assert request.sample_selection.strategy == "random"


def test_evaluate_request_serialization():
    """Test EvaluateRequest can be serialized and deserialized"""
    model_config = ModelConfig(
        llm_model_id="gpt-4",
        embedding_model_id="text-embedding-3-small"
    )
    
    sample_selection = SampleSelection(
        strategy="specific_ids",
        sample_ids=["0", "1", "2"]
    )
    
    request = EvaluateRequest(
        dataset_name="hotpotqa",
        rag_name="baseline",
        eval_type="e2e",
        model_info=model_config,
        sample_selection=sample_selection
    )
    
    # Serialize to dict
    request_dict = request.model_dump()
    assert "sample_selection" in request_dict
    assert request_dict["sample_selection"]["strategy"] == "specific_ids"
    
    # Deserialize from dict
    request2 = EvaluateRequest(**request_dict)
    assert request2.sample_selection.strategy == "specific_ids"
    assert request2.sample_selection.sample_ids == ["0", "1", "2"]
