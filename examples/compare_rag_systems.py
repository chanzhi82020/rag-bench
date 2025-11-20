"""对比多个RAG系统的评测结果

演示如何使用analysis模块对比不同RAG系统的性能。
"""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from rag_benchmark.analysis import compare_results, plot_metrics
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate_e2e
from rag_benchmark.prepare import DummyRAG, SimpleRAG, prepare_experiment_dataset

# 创建输出目录
output_dir = Path("output/compare_rag_system")
output_dir.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("RAG系统对比评测示例")
    print("=" * 60)

    llm = ChatOpenAI(
        model="deepseek-ai/deepseek-v3.1",
        api_key=SecretStr("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"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
    )
    # Create 智谱 Embedding model using langchain
    embedding = OpenAIEmbeddings(
        model="embedding-3",
        api_key=SecretStr("7f08f66caad549708238a57e0f7f33f7.EfQ9HoYpYZqBCRFX"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    # 1. 加载Golden Dataset
    print("\n1. 加载Golden Dataset...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    print(f"   加载了 {len(golden_ds)} 条记录")
    
    # 取一个小子集用于快速演示
    sample_size = 30
    records = golden_ds.head(sample_size)
    print(f"   使用 {sample_size} 条记录进行演示")
    
    # 创建简化的数据集
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class TestLoader(BaseLoader):
        def __init__(self, records):
            self.records = records
        
        def load_golden_records(self):
            return iter(self.records)
        
        def load_corpus_records(self):
            return iter([])
    
    test_ds = GoldenDataset("test", loader=TestLoader(records))
    test_ds_path = output_dir / "test_ds"
    test_ds.export(test_ds_path)

    # 2. 准备两个不同的RAG系统
    print("\n2. 准备RAG系统...")
    rag1 = DummyRAG()
    
    # 收集corpus用于SimpleRAG
    corpus = []
    for record in records:
        if record.reference_contexts:
            corpus.extend(record.reference_contexts)
    
    rag2 = SimpleRAG(corpus=corpus)
    print("   - DummyRAG (Baseline)")
    print("   - SimpleRAG (Improved)")
    
    # 3. 生成实验数据集
    print("\n3. 生成实验数据集...")
    print("   准备DummyRAG数据集...")
    exp_ds1 = prepare_experiment_dataset(test_ds, rag1, batch_size=10)
    exp_ds1_path = output_dir / "dummy_rag_exp_ds.csv"
    exp_ds1.to_csv(exp_ds1_path)
    
    print("   准备SimpleRAG数据集...")
    exp_ds2 = prepare_experiment_dataset(test_ds, rag2, batch_size=10)
    exp_ds2_path = output_dir / "simple_rag_exp_ds.csv"
    exp_ds2.to_csv(exp_ds2_path)

    # 4. 评测两个系统
    print("\n4. 评测RAG系统...")
    print("   评测DummyRAG...")
    result1 = evaluate_e2e(exp_ds1, experiment_name="dummy_rag", llm=llm, embeddings=embedding)
    result1_path = output_dir / "dummy_rag_eval_result.csv"
    result1.to_pandas().to_csv(result1_path, index=False)

    print("   评测SimpleRAG...")
    result2 = evaluate_e2e(exp_ds2, experiment_name="simple_rag", llm=llm, embeddings=embedding)
    result2_path = output_dir / "simple_rag_eval_result.csv"
    result2.to_pandas().to_csv(result2_path, index=False)

    # 5. 对比结果
    print("\n5. 对比分析...")
    comparison = compare_results(
        results=[result1, result2],
        names=["DummyRAG", "SimpleRAG"],
        metrics=["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    )
    
    # 6. 显示对比摘要
    print("\n" + "=" * 60)
    print("对比摘要")
    print("=" * 60)
    summary = comparison.summary()
    print(summary.to_string(index=False))
    
    # 7. 找出最佳模型
    print("\n" + "=" * 60)
    print("最佳模型")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
        best = comparison.get_best(metric)
        print(f"{metric:20s}: {best['name']:15s} (score: {best['score']:.4f})")
    
    # 8. 分析最差样本
    print("\n" + "=" * 60)
    print("DummyRAG在faithfulness上表现最差的3个样本")
    print("=" * 60)
    worst_cases = comparison.get_worst_cases("faithfulness", n=3, model_idx=0)
    for idx, row in worst_cases.iterrows():
        print(f"\n样本 {idx + 1}:")
        print(f"  问题: {row['user_input'][:100]}...")
        print(f"  答案: {row['response'][:100]}...")
        print(f"  Faithfulness: {row['faithfulness']:.4f}")
    
    # 9. 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    # 保存对比结果
    comparison_path = output_dir / "comparison_results.csv"
    comparison.save(str(comparison_path))
    print(f"✓ 对比结果已保存: {comparison_path}")
    
    # 保存可视化图表
    try:
        import matplotlib.pyplot as plt
        
        viz_path = output_dir / "comparison_chart.png"
        plot_metrics(
            comparison,
            metrics=["faithfulness", "answer_relevancy", "context_recall", "context_precision"],
            save_path=str(viz_path)
        )
        print(f"✓ 可视化图表已保存: {viz_path}")
        
        # 如果在交互环境中，显示图表
        if os.environ.get("DISPLAY"):
            plt.show()
    except ImportError:
        print("⚠ matplotlib未安装，跳过可视化")
    
    print("\n" + "=" * 60)
    print("评测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
