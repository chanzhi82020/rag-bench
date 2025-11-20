"""Tests for RAGAS integration of retrieval metrics"""

import pytest
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from rag_benchmark.evaluate.metrics_retrieval import (
    RecallAtK,
    PrecisionAtK,
    F1AtK,
    NDCGAtK,
    MRRMetric,
    MAPMetric,
)


def test_recall_at_k_metric():
    """Test RecallAtK as a RAGAS metric"""
    metric = RecallAtK(k=3)
    
    # Check metric properties
    assert metric.name == "recall@3"
    assert "retrieved_context_ids" in metric.get_required_columns()[list(metric.get_required_columns().keys())[0]]
    
    # Test scoring
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2", "ctx3"],
        response="test response",
        reference="test reference",
        retrieved_context_ids=["doc1", "doc2", "doc3"],
        reference_context_ids=["doc2", "doc5"]
    )
    
    score = metric.single_turn_score(sample)
    assert score == 0.5  # Found doc2 out of 2 reference docs


def test_precision_at_k_metric():
    """Test PrecisionAtK as a RAGAS metric"""
    metric = PrecisionAtK(k=3)
    
    assert metric.name == "precision@3"
    
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2", "ctx3"],
        response="test response",
        reference="test reference",
        retrieved_context_ids=["doc1", "doc2", "doc3"],
        reference_context_ids=["doc2", "doc5"]
    )
    
    score = metric.single_turn_score(sample)
    assert abs(score - 0.333) < 0.01  # 1 relevant out of 3


def test_f1_at_k_metric():
    """Test F1AtK as a RAGAS metric"""
    metric = F1AtK(k=3)
    
    assert metric.name == "f1@3"
    
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2", "ctx3"],
        response="test response",
        reference="test reference",
        retrieved_context_ids=["doc1", "doc2", "doc3"],
        reference_context_ids=["doc2", "doc5"]
    )
    
    score = metric.single_turn_score(sample)
    assert 0.0 <= score <= 1.0


def test_ndcg_at_k_metric():
    """Test NDCGAtK as a RAGAS metric"""
    metric = NDCGAtK(k=4)
    
    assert metric.name == "ndcg@4"
    
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2", "ctx3", "ctx4"],
        response="test response",
        reference="test reference",
        retrieved_context_ids=["doc1", "doc2", "doc3", "doc4"],
        reference_context_ids=["doc2", "doc3"]
    )
    
    score = metric.single_turn_score(sample)
    assert 0.0 <= score <= 1.0


def test_mrr_metric():
    """Test MRRMetric as a RAGAS metric"""
    metric = MRRMetric()
    
    assert metric.name == "mrr"
    
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2", "ctx3"],
        response="test response",
        reference="test reference",
        retrieved_context_ids=["doc1", "doc2", "doc3"],
        reference_context_ids=["doc2", "doc5"]
    )
    
    score = metric.single_turn_score(sample)
    assert score == 0.5  # doc2 is at position 2, so MRR = 1/2


def test_map_metric():
    """Test MAPMetric as a RAGAS metric"""
    metric = MAPMetric()
    
    assert metric.name == "map"
    
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2", "ctx3", "ctx4"],
        response="test response",
        reference="test reference",
        retrieved_context_ids=["doc1", "doc2", "doc3", "doc5"],
        reference_context_ids=["doc2", "doc5"]
    )
    
    score = metric.single_turn_score(sample)
    assert 0.0 <= score <= 1.0



def test_missing_context_ids():
    """Test handling of missing context_ids"""
    metric = RecallAtK(k=5)
    
    # Sample without context_ids
    sample = SingleTurnSample(
        user_input="test query",
        retrieved_contexts=["ctx1", "ctx2"],
        response="test response",
        reference="test reference"
    )
    
    score = metric.single_turn_score(sample)
    assert score == 0.0  # Should return 0.0 for missing data


def test_metric_with_evaluate_function():
    """Test that metrics can be used with evaluate() function"""
    from rag_benchmark.evaluate import evaluate
    from ragas.metrics import faithfulness
    
    # Create a small test dataset
    samples = [
        SingleTurnSample(
            user_input="query1",
            retrieved_contexts=["ctx1", "ctx2", "ctx3"],
            response="response1",
            reference="ref1",
            retrieved_context_ids=["doc1", "doc2", "doc3"],
            reference_context_ids=["doc2", "doc5"]
        ),
        SingleTurnSample(
            user_input="query2",
            retrieved_contexts=["ctx4", "ctx5", "ctx6"],
            response="response2",
            reference="ref2",
            retrieved_context_ids=["doc4", "doc5", "doc6"],
            reference_context_ids=["doc5", "doc7"]
        ),
    ]
    
    dataset = EvaluationDataset(samples=samples)
    
    # Test with only IR metrics (no LLM needed)
    result = evaluate(
        dataset=dataset,
        metrics=[RecallAtK(k=3), PrecisionAtK(k=3), MRRMetric()],
        name="test_ir_metrics",
        show_progress=False
    )
    
    # Check results
    assert "recall@3" in result.list_metrics()
    assert "precision@3" in result.list_metrics()
    assert "mrr" in result.list_metrics()
    
    # Check scores are valid
    assert 0.0 <= result.get_score("recall@3") <= 1.0
    assert 0.0 <= result.get_score("precision@3") <= 1.0
    assert 0.0 <= result.get_score("mrr") <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
