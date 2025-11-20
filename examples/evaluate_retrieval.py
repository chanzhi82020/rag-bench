"""示例：检索阶段评测

演示如何评测RAG系统的检索性能，包括：
1. 使用RAGAS的context_recall和context_precision
2. 使用传统IR指标（recall@k, precision@k, MRR, NDCG）
"""
import traceback

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate.evaluator import evaluate_retrieval
from rag_benchmark.prepare import DummyRAG, RAGConfig, prepare_experiment_dataset


def example_ragas_retrieval_metrics():
    """使用RAGAS的检索指标进行评测"""
    print("=" * 80)
    print("示例 1: 使用RAGAS检索指标")
    print("=" * 80)

    # 1. 准备数据
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

    # 2. 运行检索评测
    print("\n2. 运行检索阶段评测...")
    eval_llm = ChatOpenAI(
        model="deepseek-ai/deepseek-v3.1",
        api_key=SecretStr("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"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
    )
    print("   ✓ Configured DeepSeek LLM via NVIDIA API")

    # Create 智谱 Embedding model using langchain
    embedding_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=SecretStr("7f08f66caad549708238a57e0f7f33f7.EfQ9HoYpYZqBCRFX"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    try:
        result = evaluate_retrieval(
            dataset=exp_ds,
            experiment_name="retrieval_test",
            llm=eval_llm,
            embeddings=embedding_model,
            show_progress=True,
        )

        # 3. 查看结果
        print("\n3. 评测结果:")
        print(f"   {result}")

        print("\n✓ RAGAS检索评测完成！")
        
    except Exception as e:
        print(f"\n✗ 评测失败: {e}, {traceback.format_exc()}")
        print("   提示：需要配置有效的API密钥才能运行RAGAS评测")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Benchmark - 检索阶段评测示例")
    print("=" * 80 + "\n")

    # 运行示例1：RAGAS检索指标
    example_ragas_retrieval_metrics()

    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80)
