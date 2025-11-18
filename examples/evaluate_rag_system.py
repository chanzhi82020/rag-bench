"""示例：评测RAG系统

演示如何使用evaluate模块对RAG系统进行评测
"""

import os

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate, evaluate_e2e, evaluate_retrieval
from rag_benchmark.prepare import DummyRAG, RAGConfig, prepare_experiment_dataset
from ragas.metrics import answer_relevancy, faithfulness


def example_basic_evaluation():
    """基本评测示例"""
    print("=" * 80)
    print("Example 1: Basic Evaluation")
    print("=" * 80)

    # 1. 准备数据
    print("\n1. Preparing experiment dataset...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    
    # 只取前5条记录作为演示
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class SubsetLoader(BaseLoader):
        def __init__(self, original_dataset, limit=5):
            self.original_dataset = original_dataset
            self.limit = limit
        
        def load_golden_records(self):
            count = 0
            for record in self.original_dataset:
                if count >= self.limit:
                    break
                yield record
                count += 1
        
        def load_corpus_records(self):
            return iter([])
        
        def count_records(self):
            return min(self.limit, len(self.original_dataset))
    
    subset_loader = SubsetLoader(golden_ds, limit=5)
    golden_sample = GoldenDataset("xquad_subset", loader=subset_loader)
    
    rag = DummyRAG(config=RAGConfig(top_k=3))
    exp_ds = prepare_experiment_dataset(golden_sample, rag, show_progress=False)
    print(f"   Prepared {len(exp_ds)} samples")

    # 2. 进行评测
    print("\n2. Running evaluation...")
    
    # 设置OpenAI API密钥（RAGAS需要）
    if "OPENAI_API_KEY" not in os.environ:
        print("   ⚠️  Warning: OPENAI_API_KEY not set. Evaluation will fail.")
        print("   Please set OPENAI_API_KEY environment variable to run evaluation.")
        print("   Example: export OPENAI_API_KEY='your-api-key'")
        return
    
    result = evaluate(
        dataset=exp_ds,
        metrics=[faithfulness, answer_relevancy],
        name="basic_evaluation",
        show_progress=True,
    )

    # 3. 查看结果
    print("\n3. Evaluation Results:")
    print(f"   Name: {result.name}")
    print(f"   Dataset size: {result.dataset_size}")
    print(f"   Metrics: {result.list_metrics()}")
    print(f"\n   Scores:")
    for metric_name in result.list_metrics():
        score = result.get_score(metric_name)
        print(f"     - {metric_name}: {score:.4f}")

    # 4. 查看摘要
    print("\n4. Summary:")
    summary = result.summary()
    print(f"   Average score: {summary['average_score']:.4f}")
    print(f"   Min score: {summary['min_score']:.4f}")
    print(f"   Max score: {summary['max_score']:.4f}")

    # 5. 保存结果
    print("\n5. Saving results...")
    result.save("output/evaluation_basic.json")
    print("   Saved to: output/evaluation_basic.json")

    print("\n✓ Example 1 completed successfully!\n")


def example_predefined_metrics():
    """使用预定义指标组合"""
    print("=" * 80)
    print("Example 2: Using Predefined Metric Groups")
    print("=" * 80)

    if "OPENAI_API_KEY" not in os.environ:
        print("   ⚠️  Warning: OPENAI_API_KEY not set. Skipping this example.")
        return

    # 准备数据
    print("\n1. Preparing data...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class SubsetLoader(BaseLoader):
        def __init__(self, original_dataset, limit=5):
            self.original_dataset = original_dataset
            self.limit = limit
        
        def load_golden_records(self):
            count = 0
            for record in self.original_dataset:
                if count >= self.limit:
                    break
                yield record
                count += 1
        
        def load_corpus_records(self):
            return iter([])
        
        def count_records(self):
            return min(self.limit, len(self.original_dataset))
    
    subset_loader = SubsetLoader(golden_ds, limit=5)
    golden_sample = GoldenDataset("xquad_subset", loader=subset_loader)
    
    rag = DummyRAG()
    exp_ds = prepare_experiment_dataset(golden_sample, rag, show_progress=False)

    # 使用预定义的指标组合
    print("\n2. Evaluating with 'e2e' metric group...")
    result = evaluate_e2e(exp_ds, name="e2e_evaluation", show_progress=True)

    print("\n3. Results:")
    for metric_name in result.list_metrics():
        score = result.get_score(metric_name)
        print(f"   {metric_name}: {score:.4f}")

    print("\n✓ Example 2 completed successfully!\n")


def example_compare_results():
    """对比两次评测结果"""
    print("=" * 80)
    print("Example 3: Comparing Evaluation Results")
    print("=" * 80)

    if "OPENAI_API_KEY" not in os.environ:
        print("   ⚠️  Warning: OPENAI_API_KEY not set. Skipping this example.")
        return

    # 准备数据
    print("\n1. Preparing data...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class SubsetLoader(BaseLoader):
        def __init__(self, original_dataset, limit=5):
            self.original_dataset = original_dataset
            self.limit = limit
        
        def load_golden_records(self):
            count = 0
            for record in self.original_dataset:
                if count >= self.limit:
                    break
                yield record
                count += 1
        
        def load_corpus_records(self):
            return iter([])
        
        def count_records(self):
            return min(self.limit, len(self.original_dataset))
    
    subset_loader = SubsetLoader(golden_ds, limit=5)
    golden_sample = GoldenDataset("xquad_subset", loader=subset_loader)

    # 评测两个不同的RAG系统
    print("\n2. Evaluating RAG System V1...")
    rag_v1 = DummyRAG(config=RAGConfig(top_k=3), seed=42)
    exp_ds_v1 = prepare_experiment_dataset(golden_sample, rag_v1, show_progress=False)
    result_v1 = evaluate(
        exp_ds_v1,
        metrics=[faithfulness, answer_relevancy],
        name="rag_v1",
        show_progress=False,
    )

    print("\n3. Evaluating RAG System V2...")
    rag_v2 = DummyRAG(config=RAGConfig(top_k=5), seed=123)
    
    # 重新创建subset_loader
    subset_loader2 = SubsetLoader(golden_ds, limit=5)
    golden_sample2 = GoldenDataset("xquad_subset2", loader=subset_loader2)
    
    exp_ds_v2 = prepare_experiment_dataset(golden_sample2, rag_v2, show_progress=False)
    result_v2 = evaluate(
        exp_ds_v2,
        metrics=[faithfulness, answer_relevancy],
        name="rag_v2",
        show_progress=False,
    )

    # 对比结果
    print("\n4. Comparing results...")
    comparison = result_v1.compare_with(result_v2)

    print(f"\n   Comparing: {comparison['self_name']} vs {comparison['other_name']}")
    print(f"   Common metrics: {comparison['common_metrics']}")
    print("\n   Comparison:")
    for metric, data in comparison["comparison"].items():
        print(f"     {metric}:")
        print(f"       V1: {data['self_score']:.4f}")
        print(f"       V2: {data['other_score']:.4f}")
        print(f"       Difference: {data['difference']:.4f}")
        print(f"       Improvement: {data['improvement']:.2f}%")

    print("\n✓ Example 3 completed successfully!\n")


def example_save_and_load():
    """保存和加载评测结果"""
    print("=" * 80)
    print("Example 4: Save and Load Results")
    print("=" * 80)

    if "OPENAI_API_KEY" not in os.environ:
        print("   ⚠️  Warning: OPENAI_API_KEY not set. Skipping this example.")
        return

    # 准备数据并评测
    print("\n1. Running evaluation...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class SubsetLoader(BaseLoader):
        def __init__(self, original_dataset, limit=5):
            self.original_dataset = original_dataset
            self.limit = limit
        
        def load_golden_records(self):
            count = 0
            for record in self.original_dataset:
                if count >= self.limit:
                    break
                yield record
                count += 1
        
        def load_corpus_records(self):
            return iter([])
        
        def count_records(self):
            return min(self.limit, len(self.original_dataset))
    
    subset_loader = SubsetLoader(golden_ds, limit=5)
    golden_sample = GoldenDataset("xquad_subset", loader=subset_loader)
    
    rag = DummyRAG()
    exp_ds = prepare_experiment_dataset(golden_sample, rag, show_progress=False)
    result = evaluate(
        exp_ds,
        metrics=[faithfulness, answer_relevancy],
        name="save_load_test",
        show_progress=False,
    )

    # 保存为JSON
    print("\n2. Saving as JSON...")
    result.save("output/evaluation_save_load.json", format="json")
    print("   Saved to: output/evaluation_save_load.json")

    # 保存为CSV
    print("\n3. Saving as CSV...")
    result.save("output/evaluation_save_load.csv", format="csv")
    print("   Saved to: output/evaluation_save_load.csv")

    # 加载结果
    print("\n4. Loading from JSON...")
    from rag_benchmark.evaluate import EvaluationResult

    loaded_result = EvaluationResult.load("output/evaluation_save_load.json")
    print(f"   Loaded: {loaded_result.name}")
    print(f"   Metrics: {loaded_result.list_metrics()}")
    print(f"   Dataset size: {loaded_result.dataset_size}")

    print("\n✓ Example 4 completed successfully!\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Benchmark - Evaluate Module Examples")
    print("=" * 80 + "\n")

    # 检查API密钥
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  IMPORTANT: OPENAI_API_KEY environment variable is not set.")
        print("RAGAS requires OpenAI API to run evaluations.")
        print("Please set it before running these examples:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nRunning examples in demo mode (will show warnings)...\n")

    # 运行示例
    example_basic_evaluation()
    example_predefined_metrics()
    example_compare_results()
    example_save_and_load()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
