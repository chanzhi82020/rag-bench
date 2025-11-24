"""Tests for sample counting logic in evaluation results"""

import pytest
import math


def count_samples(detailed_results, metric_names=None):
    """Helper function that mimics the sample counting logic in main.py
    
    Args:
        detailed_results: List of result records
        metric_names: Set of metric names to consider. If None, infers from first record.
    """
    completed_samples = 0
    failed_samples = 0
    
    # If metric_names not provided, infer from the first record
    # by excluding known non-metric columns
    if metric_names is None and detailed_results:
        non_metric_columns = {
            'user_input', 'response', 'retrieved_contexts', 'reference',
            'reference_contexts', 'reference_context_ids', 'retrieved_context_ids'
        }
        first_record = detailed_results[0]
        metric_names = {k for k in first_record.keys() if k not in non_metric_columns}
    
    if metric_names is None:
        metric_names = set()
    
    for record in detailed_results:
        # Extract metric values using the actual metric names
        metric_values = []
        for key, value in record.items():
            if key in metric_names:
                metric_values.append(value)
        
        # Check if all metrics are NaN/None (complete failure)
        # or if at least one metric has a valid value (partial or complete success)
        if not metric_values:
            # No metrics found, consider as failed
            failed_samples += 1
        else:
            all_invalid = all(
                value is None or (isinstance(value, float) and math.isnan(value))
                for value in metric_values
            )
            if all_invalid:
                failed_samples += 1
            else:
                completed_samples += 1
    
    return completed_samples, failed_samples


def test_all_metrics_valid():
    """Test sample with all valid metrics is counted as completed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': 0.95,
            'answer_relevancy': 0.87,
            'context_precision': 0.92
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 1
    assert failed == 0


def test_all_metrics_nan():
    """Test sample with all NaN metrics is counted as failed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': float('nan'),
            'answer_relevancy': float('nan'),
            'context_precision': float('nan')
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 0
    assert failed == 1


def test_all_metrics_none():
    """Test sample with all None metrics is counted as failed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': None,
            'answer_relevancy': None,
            'context_precision': None
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 0
    assert failed == 1


def test_partial_metrics_valid():
    """Test sample with some valid and some NaN metrics is counted as completed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': 0.95,
            'answer_relevancy': float('nan'),
            'context_precision': 0.92
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 1
    assert failed == 0


def test_one_valid_metric():
    """Test sample with only one valid metric is counted as completed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': 0.95,
            'answer_relevancy': float('nan'),
            'context_precision': None
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 1
    assert failed == 0


def test_mixed_samples():
    """Test multiple samples with different success states"""
    detailed_results = [
        # Fully successful
        {
            'user_input': 'q1',
            'response': 'a1',
            'faithfulness': 0.95,
            'answer_relevancy': 0.87
        },
        # Partially successful
        {
            'user_input': 'q2',
            'response': 'a2',
            'faithfulness': 0.80,
            'answer_relevancy': float('nan')
        },
        # Completely failed
        {
            'user_input': 'q3',
            'response': 'a3',
            'faithfulness': float('nan'),
            'answer_relevancy': float('nan')
        },
        # Another successful
        {
            'user_input': 'q4',
            'response': 'a4',
            'faithfulness': 0.92,
            'answer_relevancy': 0.91
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 3  # First, second, and fourth samples
    assert failed == 1     # Third sample


def test_no_metrics():
    """Test sample with no metric columns is counted as failed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'retrieved_contexts': ['context1']
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 0
    assert failed == 1


def test_empty_results():
    """Test empty results list"""
    detailed_results = []
    completed, failed = count_samples(detailed_results)
    assert completed == 0
    assert failed == 0


def test_zero_value_is_valid():
    """Test that 0.0 is considered a valid metric value"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': 0.0,
            'answer_relevancy': 0.0
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 1
    assert failed == 0


def test_mixed_none_and_nan():
    """Test sample with mix of None and NaN is counted as failed"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': None,
            'answer_relevancy': float('nan'),
            'context_precision': None
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 0
    assert failed == 1


def test_inf_values_are_valid():
    """Test that infinity values are considered valid (before sanitization)"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'faithfulness': float('inf'),
            'answer_relevancy': 0.87
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 1
    assert failed == 0


def test_reference_contexts_excluded():
    """Test that reference_contexts and reference_context_ids are not treated as metrics"""
    detailed_results = [
        {
            'user_input': 'question',
            'response': 'answer',
            'reference': 'reference answer',
            'reference_contexts': ['context1', 'context2'],
            'reference_context_ids': ['id1', 'id2'],
            'retrieved_contexts': ['retrieved1'],
            'retrieved_context_ids': ['ret_id1'],
            'faithfulness': float('nan'),
            'answer_relevancy': float('nan')
        }
    ]
    completed, failed = count_samples(detailed_results)
    # Should be failed because all actual metrics (faithfulness, answer_relevancy) are NaN
    # reference_contexts and reference_context_ids should not be counted as metrics
    assert completed == 0
    assert failed == 1


def test_real_world_failed_sample():
    """Test with real-world failed sample structure"""
    detailed_results = [
        {
            "user_input": "谁是梅尔菲伯爵？",
            "retrieved_contexts": ["context1", "context2"],
            "reference_contexts": ["ref_context"],
            "reference_context_ids": ["xquad_zh_0011"],
            "response": "我不知道。",
            "reference": "铁臂威廉 (William Iron Arm)",
            "faithfulness": float('nan'),
            "answer_relevancy": float('nan'),
            "context_precision": float('nan'),
            "context_recall": float('nan')
        }
    ]
    completed, failed = count_samples(detailed_results)
    assert completed == 0
    assert failed == 1


def test_with_explicit_metric_names():
    """Test using explicit metric names (simulating real evaluation flow)"""
    # Simulate metrics dict from ragas result
    metric_names = {'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'}
    
    detailed_results = [
        {
            "user_input": "question",
            "response": "answer",
            "reference": "ref",
            "retrieved_contexts": ["ctx1"],
            "reference_contexts": ["ref_ctx"],
            "reference_context_ids": ["id1"],
            "faithfulness": 0.95,
            "answer_relevancy": 0.87,
            "context_precision": float('nan'),
            "context_recall": 0.92
        }
    ]
    
    completed, failed = count_samples(detailed_results, metric_names)
    # Should be completed because 3 out of 4 metrics are valid
    assert completed == 1
    assert failed == 0


def test_with_explicit_metric_names_all_failed():
    """Test with explicit metric names where all metrics failed"""
    metric_names = {'faithfulness', 'answer_relevancy'}
    
    detailed_results = [
        {
            "user_input": "question",
            "response": "answer",
            "reference": "ref",
            "retrieved_contexts": ["ctx1"],
            "some_other_field": "value",  # This should be ignored
            "faithfulness": float('nan'),
            "answer_relevancy": float('nan')
        }
    ]
    
    completed, failed = count_samples(detailed_results, metric_names)
    # Should be failed because both metrics are NaN
    assert completed == 0
    assert failed == 1


def test_metric_names_filter_correctly():
    """Test that only specified metric names are considered"""
    # Only consider these two as metrics
    metric_names = {'faithfulness', 'answer_relevancy'}
    
    detailed_results = [
        {
            "user_input": "question",
            "response": "answer",
            "faithfulness": 0.95,
            "answer_relevancy": 0.87,
            "some_random_field": float('nan'),  # Should be ignored
            "another_field": None  # Should be ignored
        }
    ]
    
    completed, failed = count_samples(detailed_results, metric_names)
    # Should be completed because the two actual metrics are valid
    # even though other fields have NaN/None
    assert completed == 1
    assert failed == 0
