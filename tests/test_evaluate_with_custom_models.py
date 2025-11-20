"""Integration tests for evaluate_with_custom_models example.

Feature: fix-ragas-api-compatibility

This module contains integration tests to verify that the example runs without
API compatibility errors when using the new ragas API pattern.

Validates: Requirements 1.1, 1.5
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path to allow direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


def test_example_imports_successfully():
    """Test that the example file can be imported without errors."""
    # This verifies that all imports are correct and the file is syntactically valid
    import examples.evaluate_with_custom_models as example_module
    
    assert hasattr(example_module, "example_with_custom_models")
    assert hasattr(example_module, "example_with_different_providers")
    assert hasattr(example_module, "example_with_run_config")


def test_client_creation_pattern():
    """Test that langchain LLMs can be created with the pattern used in the example.
    
    This verifies that the langchain API pattern works correctly without requiring actual API calls.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Test NVIDIA API LLM creation (as used in the example)
    nvidia_llm = ChatOpenAI(
        model="deepseek-ai/deepseek-v3.1",
        api_key="test-nvidia-key",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
    )
    
    assert isinstance(nvidia_llm, ChatOpenAI)
    assert nvidia_llm.model_name == "deepseek-ai/deepseek-v3.1"
    # API key is stored as SecretStr, so we check it's set
    assert nvidia_llm.openai_api_key is not None
    assert "integrate.api.nvidia.com" in nvidia_llm.openai_api_base
    
    # Test GLM Embedding creation (as used in the example)
    glm_embeddings = OpenAIEmbeddings(
        model="embedding-3",
        openai_api_key="test-glm-key",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )
    
    assert isinstance(glm_embeddings, OpenAIEmbeddings)
    assert glm_embeddings.model == "embedding-3"
    assert glm_embeddings.openai_api_key is not None


@patch("examples.evaluate_with_custom_models.ChatOpenAI")
@patch("examples.evaluate_with_custom_models.OpenAIEmbeddings")
@patch("examples.evaluate_with_custom_models.evaluate")
@patch("examples.evaluate_with_custom_models.prepare_experiment_dataset")
@patch("examples.evaluate_with_custom_models.GoldenDataset")
def test_example_with_custom_models_runs_without_api_errors(
    mock_golden_dataset,
    mock_prepare_dataset,
    mock_evaluate,
    mock_openai_embeddings,
    mock_chat_openai,
):
    """Test that example_with_custom_models() runs without API compatibility errors.
    
    This integration test mocks the API calls to verify that:
    1. Langchain LLMs are created correctly
    2. LLMs are created with proper API credentials
    3. The evaluation completes without errors
    
    Validates: Requirements 1.1, 1.5
    """
    # Mock the dataset and evaluation components
    mock_dataset_instance = Mock()
    mock_dataset_instance.__len__ = Mock(return_value=3)
    mock_dataset_instance.__iter__ = Mock(return_value=iter([
        {"question": "Q1", "answer": "A1", "contexts": ["C1"]},
        {"question": "Q2", "answer": "A2", "contexts": ["C2"]},
        {"question": "Q3", "answer": "A3", "contexts": ["C3"]},
    ]))
    mock_golden_dataset.return_value = mock_dataset_instance
    
    mock_exp_dataset = Mock()
    mock_exp_dataset.__len__ = Mock(return_value=3)
    mock_prepare_dataset.return_value = mock_exp_dataset
    
    # Mock the LLM and embedding instances
    mock_llm = Mock()
    mock_embedding = Mock()
    mock_chat_openai.return_value = mock_llm
    mock_openai_embeddings.return_value = mock_embedding
    
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.name = "custom_models_evaluation"
    mock_result.dataset_size = 3
    mock_result.list_metrics.return_value = ["faithfulness", "answer_relevancy"]
    mock_result.get_score.return_value = 0.85
    
    mock_metric_detail = Mock()
    mock_metric_detail.min_score = 0.75
    mock_metric_detail.max_score = 0.95
    mock_metric_detail.std_score = 0.08
    mock_result.get_metric.return_value = mock_metric_detail
    
    mock_result.save = Mock()
    mock_evaluate.return_value = mock_result
    
    # Import and run the example function
    from examples.evaluate_with_custom_models import example_with_custom_models
    
    # Run the example - should not raise any errors
    example_with_custom_models()
    
    # Verify that ChatOpenAI was called with proper parameters
    assert mock_chat_openai.called, "ChatOpenAI should be called"
    llm_call_kwargs = mock_chat_openai.call_args[1]
    assert "model" in llm_call_kwargs, "ChatOpenAI should be called with 'model' parameter"
    assert "api_key" in llm_call_kwargs, "ChatOpenAI should be called with 'api_key' parameter"
    assert "base_url" in llm_call_kwargs, "ChatOpenAI should be called with 'base_url' parameter"
    
    # Verify that OpenAIEmbeddings was called with proper parameters
    assert mock_openai_embeddings.called, "OpenAIEmbeddings should be called"
    embedding_call_kwargs = mock_openai_embeddings.call_args[1]
    assert "model" in embedding_call_kwargs, "OpenAIEmbeddings should be called with 'model' parameter"
    # The parameter can be either 'api_key' or 'openai_api_key' depending on langchain version
    assert "api_key" in embedding_call_kwargs or "openai_api_key" in embedding_call_kwargs, \
        "OpenAIEmbeddings should be called with 'api_key' or 'openai_api_key' parameter"
    
    # Verify that evaluate was called with the mocked LLM and embedding instances
    assert mock_evaluate.called, "evaluate should be called"
    eval_call_kwargs = mock_evaluate.call_args[1]
    assert eval_call_kwargs["llm"] == mock_llm, "evaluate should receive the LLM instance"
    assert eval_call_kwargs["embeddings"] == mock_embedding, "evaluate should receive the embedding instance"


def test_example_with_different_providers_displays_correctly(capsys):
    """Test that example_with_different_providers() displays the langchain API pattern.
    
    This test verifies that the documentation examples show the correct
    langchain-based API pattern.
    """
    from examples.evaluate_with_custom_models import example_with_different_providers
    
    # Run the example
    example_with_different_providers()
    
    # Capture the output
    captured = capsys.readouterr()
    output = captured.out
    
    # Verify that the output shows the langchain API pattern
    assert "from langchain_openai import ChatOpenAI" in output, "Should show ChatOpenAI import"
    assert "ChatOpenAI(" in output, "Should show ChatOpenAI usage"
    assert "OpenAIEmbeddings" in output, "Should show OpenAIEmbeddings usage"
    
    # Verify that all provider examples are shown
    assert "OpenAI (官方)" in output or "OpenAI" in output
    assert "DeepSeek" in output or "NVIDIA" in output
    assert "GLM" in output or "智谱" in output
    assert "Azure" in output
    assert "Ollama" in output
    
    # Verify migration guide is present
    assert "Migration" in output or "migration" in output
    assert "langchain" in output.lower(), "Should mention langchain in migration guide"


def test_example_error_handling_with_invalid_credentials():
    """Test that the example handles errors gracefully when API calls fail.
    
    This verifies that error handling is in place and provides helpful messages.
    """
    with patch("examples.evaluate_with_custom_models.evaluate") as mock_evaluate:
        # Simulate an API error
        mock_evaluate.side_effect = Exception("API authentication failed")
        
        # Mock other dependencies
        with patch("examples.evaluate_with_custom_models.GoldenDataset"):
            with patch("examples.evaluate_with_custom_models.prepare_experiment_dataset"):
                with patch("examples.evaluate_with_custom_models.ChatOpenAI"):
                    with patch("examples.evaluate_with_custom_models.OpenAIEmbeddings"):
                        from examples.evaluate_with_custom_models import example_with_custom_models
                        
                        # Should not raise an exception - errors should be caught and handled
                        try:
                            example_with_custom_models()
                        except Exception as e:
                            pytest.fail(f"Example should handle errors gracefully, but raised: {e}")


def test_client_instances_are_independent():
    """Test that multiple langchain LLM instances can be created independently.
    
    This verifies that creating multiple LLMs (e.g., for different providers)
    doesn't cause conflicts or shared state issues.
    """
    from langchain_openai import ChatOpenAI
    
    # Create multiple LLMs as done in the example
    llm1 = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="key1",
        base_url="https://api1.example.com/v1",
    )
    
    llm2 = ChatOpenAI(
        model="gpt-4o",
        api_key="key2",
        base_url="https://api2.example.com/v1",
    )
    
    # Verify they are independent
    assert llm1.openai_api_key != llm2.openai_api_key
    assert llm1.model_name != llm2.model_name
    assert llm1 is not llm2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
