"""Tests for retrieval metrics module"""

import pytest
from rag_benchmark.evaluate.metrics_retrieval import (
    recall_at_k,
    precision_at_k,
    f1_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision,
    compute_retrieval_metrics,
)


def test_recall_at_k():
    """Test recall@k calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    reference = ["doc2", "doc5"]
    
    # Recall@3: found doc2 out of 2 reference docs = 0.5
    assert recall_at_k(retrieved, reference, k=3) == 0.5
    
    # Recall@5: still only found doc2 = 0.5
    assert recall_at_k(retrieved, reference, k=5) == 0.5
    
    # All relevant docs found
    retrieved_all = ["doc1", "doc2", "doc3", "doc5"]
    assert recall_at_k(retrieved_all, reference, k=4) == 1.0


def test_precision_at_k():
    """Test precision@k calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    reference = ["doc2", "doc5"]
    
    # Precision@3: 1 relevant out of 3 = 0.333...
    assert abs(precision_at_k(retrieved, reference, k=3) - 0.333) < 0.01
    
    # Precision@4: 1 relevant out of 4 = 0.25
    assert precision_at_k(retrieved, reference, k=4) == 0.25


def test_f1_at_k():
    """Test F1@k calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    reference = ["doc2", "doc5"]
    
    recall = recall_at_k(retrieved, reference, k=3)
    precision = precision_at_k(retrieved, reference, k=3)
    expected_f1 = 2 * (precision * recall) / (precision + recall)
    
    assert abs(f1_at_k(retrieved, reference, k=3) - expected_f1) < 0.001


def test_mean_reciprocal_rank():
    """Test MRR calculation"""
    # First relevant doc at position 2
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    reference = ["doc2", "doc5"]
    assert mean_reciprocal_rank(retrieved, reference) == 0.5  # 1/2
    
    # First relevant doc at position 1
    retrieved_first = ["doc2", "doc1", "doc3", "doc4"]
    assert mean_reciprocal_rank(retrieved_first, reference) == 1.0  # 1/1
    
    # No relevant docs found
    retrieved_none = ["doc1", "doc3", "doc4"]
    assert mean_reciprocal_rank(retrieved_none, reference) == 0.0


def test_ndcg_at_k():
    """Test NDCG@k calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    reference = ["doc2", "doc3"]
    
    # NDCG should be between 0 and 1
    ndcg = ndcg_at_k(retrieved, reference, k=4)
    assert 0.0 <= ndcg <= 1.0
    
    # Perfect ranking should give NDCG = 1.0
    perfect_retrieved = ["doc2", "doc3", "doc1", "doc4"]
    assert ndcg_at_k(perfect_retrieved, reference, k=4) == 1.0


def test_average_precision():
    """Test Average Precision calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc5"]
    reference = ["doc2", "doc5"]
    
    # AP should be between 0 and 1
    ap = average_precision(retrieved, reference)
    assert 0.0 <= ap <= 1.0
    
    # Perfect ranking
    perfect_retrieved = ["doc2", "doc5", "doc1", "doc3"]
    assert average_precision(perfect_retrieved, reference) == 1.0


def test_compute_retrieval_metrics():
    """Test batch computation of retrieval metrics"""
    retrieved_list = [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],
        ["doc6", "doc7", "doc8", "doc9", "doc10"],
        ["doc2", "doc11", "doc12", "doc13", "doc14"],
    ]
    
    reference_list = [
        ["doc2", "doc5", "doc15"],
        ["doc6", "doc16"],
        ["doc2", "doc17"],
    ]
    
    metrics = compute_retrieval_metrics(
        retrieved_list,
        reference_list,
        k_values=[1, 3, 5]
    )
    
    # Check all expected metrics are present
    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "recall@5" in metrics
    assert "precision@1" in metrics
    assert "precision@3" in metrics
    assert "precision@5" in metrics
    assert "f1@1" in metrics
    assert "f1@3" in metrics
    assert "f1@5" in metrics
    assert "ndcg@1" in metrics
    assert "ndcg@3" in metrics
    assert "ndcg@5" in metrics
    assert "mrr" in metrics
    assert "map" in metrics
    
    # All metrics should be between 0 and 1
    for metric_name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{metric_name} = {value} is out of range"


def test_empty_reference():
    """Test handling of empty reference list"""
    retrieved = ["doc1", "doc2", "doc3"]
    reference = []
    
    assert recall_at_k(retrieved, reference, k=3) == 0.0
    assert average_precision(retrieved, reference) == 0.0


def test_empty_retrieved():
    """Test handling of empty retrieved list"""
    retrieved = []
    reference = ["doc1", "doc2"]
    
    assert precision_at_k(retrieved, reference, k=3) == 0.0
    assert mean_reciprocal_rank(retrieved, reference) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
