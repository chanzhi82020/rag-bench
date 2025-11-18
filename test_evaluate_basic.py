"""测试evaluate模块的基本功能（不需要OpenAI API）"""

from rag_benchmark.evaluate import EvaluationResult, MetricResult


def test_metric_result():
    """测试MetricResult"""
    print("Testing MetricResult...")
    
    metric = MetricResult(
        name="faithfulness",
        score=0.85,
        scores=[0.9, 0.8, 0.85, 0.9, 0.75],
        metadata={"test": "value"}
    )
    
    print(f"  Name: {metric.name}")
    print(f"  Score: {metric.score}")
    print(f"  Count: {metric.count}")
    print(f"  Min: {metric.min_score}")
    print(f"  Max: {metric.max_score}")
    print(f"  Std: {metric.std_score:.4f}")
    
    # 测试to_dict
    metric_dict = metric.to_dict()
    print(f"  Dict keys: {list(metric_dict.keys())}")
    
    print("✓ MetricResult test passed\n")


def test_evaluation_result():
    """测试EvaluationResult"""
    print("Testing EvaluationResult...")
    
    metrics = {
        "faithfulness": MetricResult(
            name="faithfulness",
            score=0.85,
            scores=[0.9, 0.8, 0.85, 0.9, 0.75]
        ),
        "answer_relevancy": MetricResult(
            name="answer_relevancy",
            score=0.78,
            scores=[0.8, 0.75, 0.8, 0.75, 0.8]
        ),
    }
    
    result = EvaluationResult(
        name="test_evaluation",
        metrics=metrics,
        dataset_size=5,
        metadata={"test": "metadata"}
    )
    
    print(f"  Name: {result.name}")
    print(f"  Dataset size: {result.dataset_size}")
    print(f"  Metrics: {result.list_metrics()}")
    
    # 测试get_score
    faith_score = result.get_score("faithfulness")
    print(f"  Faithfulness score: {faith_score}")
    
    # 测试summary
    summary = result.summary()
    print(f"  Summary keys: {list(summary.keys())}")
    print(f"  Average score: {summary['average_score']:.4f}")
    
    # 测试保存和加载
    print("\n  Testing save/load...")
    result.save("output/test_result.json")
    loaded = EvaluationResult.load("output/test_result.json")
    print(f"  Loaded name: {loaded.name}")
    print(f"  Loaded metrics: {loaded.list_metrics()}")
    
    print("✓ EvaluationResult test passed\n")


def test_comparison():
    """测试结果对比"""
    print("Testing result comparison...")
    
    result1 = EvaluationResult(
        name="v1",
        metrics={
            "faithfulness": MetricResult(
                name="faithfulness",
                score=0.85,
                scores=[0.85] * 5
            ),
        },
        dataset_size=5
    )
    
    result2 = EvaluationResult(
        name="v2",
        metrics={
            "faithfulness": MetricResult(
                name="faithfulness",
                score=0.90,
                scores=[0.90] * 5
            ),
        },
        dataset_size=5
    )
    
    comparison = result1.compare_with(result2)
    print(f"  Comparing: {comparison['self_name']} vs {comparison['other_name']}")
    print(f"  Common metrics: {comparison['common_metrics']}")
    
    faith_comp = comparison['comparison']['faithfulness']
    print(f"  V1 score: {faith_comp['self_score']}")
    print(f"  V2 score: {faith_comp['other_score']}")
    print(f"  Improvement: {faith_comp['improvement']:.2f}%")
    
    print("✓ Comparison test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Evaluate Module Basic Functionality")
    print("=" * 60 + "\n")
    
    test_metric_result()
    test_evaluation_result()
    test_comparison()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
