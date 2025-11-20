"""示例：生成阶段评测

演示如何评测RAG系统的生成性能，包括：
1. Faithfulness（忠实度）
2. Answer Correctness（答案正确性）
3. 其他生成质量指标
"""
import traceback

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate_generation
from rag_benchmark.prepare import DummyRAG, RAGConfig, prepare_experiment_dataset


def main():
    """生成阶段评测示例"""
    print("=" * 80)
    print("RAG Benchmark - 生成阶段评测示例")
    print("=" * 80)

    # 1. 准备实验数据集
    print("\n1. 准备实验数据集...")
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
    print(f"   准备了 {len(exp_ds)} 条实验数据")

    # 2. 配置评测模型（可选）
    print("\n2. 配置评测模型...")
    print("   注意：需要真实的API密钥才能运行评测")
    
    # 示例：使用自定义模型
    # eval_llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     api_key="your-openai-api-key",
    #     temperature=0.0,
    # )
    # 
    # eval_embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     api_key="your-openai-api-key",
    # )

    # Create DeepSeek LLM via NVIDIA API using langchain
    # Note: ragas evaluation requires langchain LLMs, not instructor LLMs
    eval_llm = ChatOpenAI(
        model="deepseek-ai/deepseek-v3.1",
        api_key=SecretStr("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"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
    )
    # Create 智谱 Embedding model using langchain
    eval_embeddings = OpenAIEmbeddings(
        model="embedding-3",
        api_key=SecretStr("7f08f66caad549708238a57e0f7f33f7.EfQ9HoYpYZqBCRFX"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    print("   使用RAGAS默认模型")

    # 3. 运行生成阶段评测
    print("\n3. 运行生成阶段评测...")
    print("   评测指标: faithfulness, answer_correctness")
    print("   这可能需要几分钟...")
    
    try:
        result = evaluate_generation(
            dataset=exp_ds,
            llm=eval_llm,
            embeddings=eval_embeddings,
            experiment_name="generation_test",
            show_progress=True,
        )

        # 4. 查看结果
        print("\n4. 评测结果:")
        print(f"  {result}")


        # 5. 分析结果
        print("\n5. 结果分析:")
        
        # Faithfulness分析
        faith_score = result._repr_dict["faithfulness"]
        print(f"\n   Faithfulness（忠实度）分析:")
        print(f"     平均得分: {faith_score:.4f}")
        print(f"     解释: 衡量生成的答案是否忠实于检索到的上下文")
        print(f"     得分范围: 0.0（完全不忠实）到 1.0（完全忠实）")
        if faith_score < 0.5:
            print(f"     ⚠️  警告: 忠实度较低，生成的答案可能包含幻觉内容")
        elif faith_score < 0.7:
            print(f"     ℹ️  提示: 忠实度中等，建议检查生成逻辑")
        else:
            print(f"     ✓  良好: 生成的答案较好地基于检索到的上下文")
        
        # Answer Correctness分析
        correct_score = result._repr_dict["answer_correctness"]
        print(f"\n   Answer Correctness（答案正确性）分析:")
        print(f"     平均得分: {correct_score:.4f}")
        print(f"     解释: 衡量生成的答案与参考答案的相似度")
        print(f"     得分范围: 0.0（完全不正确）到 1.0（完全正确）")
        if correct_score < 0.5:
            print(f"     ⚠️  警告: 答案正确性较低，需要改进生成质量")
        elif correct_score < 0.7:
            print(f"     ℹ️  提示: 答案正确性中等，还有提升空间")
        else:
            print(f"     ✓  良好: 生成的答案质量较高")

        # 6. 保存结果
        print("\n6. 保存结果...")
        result.to_pandas().to_csv("output/generation_evaluation.csv")
        print("   结果已保存到: output/generation_evaluation.csv")

        print("\n✓ 生成阶段评测完成！")
        
    except Exception as e:
        print(f"\n✗ 评测失败: {e}, {traceback.format_exc()}")
        print("\n可能的原因:")
        print("  1. 没有配置有效的API密钥")
        print("  2. 网络连接问题")
        print("  3. API配额不足")
        print("\n解决方案:")
        print("  - 配置有效的OpenAI API密钥")
        print("  - 或使用其他兼容的LLM服务")
        print("  - 参考 examples/evaluate_with_custom_models.py 了解如何配置自定义模型")


def explain_generation_metrics():
    """解释生成阶段的各种指标"""
    print("\n" + "=" * 80)
    print("生成阶段指标说明")
    print("=" * 80)
    
    print("""
    1. Faithfulness（忠实度）
       - 定义: 衡量生成的答案是否忠实于检索到的上下文
       - 计算方式: 使用LLM判断答案中的每个陈述是否能从上下文中推断出来
       - 重要性: 防止模型产生幻觉（hallucination）
       - 理想值: > 0.8
    
    2. Answer Correctness（答案正确性）
       - 定义: 衡量生成的答案与参考答案的相似度
       - 计算方式: 综合考虑语义相似度和事实准确性
       - 重要性: 直接反映RAG系统的整体质量
       - 理想值: > 0.7
    
    3. Answer Relevancy（答案相关性）
       - 定义: 衡量答案是否直接回答了用户的问题
       - 计算方式: 检查答案是否包含问题的关键信息
       - 重要性: 确保答案切题，不偏离主题
       - 理想值: > 0.8
    
    提示：
    - 这些指标需要LLM来计算，因此需要配置API密钥
    - 不同的LLM可能会给出略有不同的评分
    - 建议使用相同的评测LLM来对比不同RAG系统的性能
    """)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Benchmark - 生成阶段评测示例")
    print("=" * 80 + "\n")

    # 运行主示例
    main()

    # 显示指标说明
    explain_generation_metrics()

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
