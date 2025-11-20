"""示例：完整的RAG系统端到端评测

演示如何使用RAG Benchmark框架进行完整的端到端评测流程：
1. 加载Golden Dataset
2. 使用RAG系统准备实验数据集
3. 进行端到端评测
4. 查看和保存结果
"""
import traceback

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate_e2e
from rag_benchmark.prepare import DummyRAG, RAGConfig, prepare_experiment_dataset


def main():
    """完整的RAG评测流程示例"""
    print("=" * 80)
    print("RAG Benchmark - 完整端到端评测示例")
    print("=" * 80)

    # 1. 加载Golden Dataset
    print("\n1. 加载Golden Dataset...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    print(f"   加载了 {len(golden_ds)} 条记录")
    print(f"   数据集统计: {golden_ds.stats()}")

    # 2. 准备实验数据集
    print("\n2. 准备实验数据集...")
    print("   使用DummyRAG填充retrieved_contexts和response...")
    
    # 创建RAG系统（这里使用DummyRAG作为示例）
    rag = DummyRAG(config=RAGConfig(top_k=3))
    
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
    
    # 准备实验数据集
    exp_ds = prepare_experiment_dataset(golden_sample, rag, show_progress=True)
    print(f"   准备了 {len(exp_ds)} 条实验数据")

    # 3. 配置评测模型（可选）
    print("\n3. 配置评测模型...")
    print("   注意：这里需要真实的API密钥才能运行评测")
    print("   如果没有API密钥，可以跳过这一步，使用RAGAS默认模型")
    
    # 示例：使用自定义模型（需要真实的API密钥）
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
    
    # 4. 运行端到端评测
    print("\n4. 运行端到端评测...")
    print("   评测指标: faithfulness, answer_relevancy")
    print("   这可能需要几分钟...")
    
    try:
        result = evaluate_e2e(
            dataset=exp_ds,
            llm=eval_llm,
            embeddings=eval_embeddings,
            experiment_name="rag_e2e_evaluation",
            show_progress=True,
        )

        # 5. 查看结果
        print("\n5. 评测结果:")
        print(f" {result}")

        # 6. 保存结果
        print("\n6. 保存结果...")
        result.to_pandas().to_csv("output/rag_e2e_evaluation.csv")
        print("   结果已保存到: output/rag_e2e_evaluation.csv")

        print("\n✓ 评测完成！")
        
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


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Benchmark - 完整端到端评测示例")
    print("=" * 80 + "\n")

    main()

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
