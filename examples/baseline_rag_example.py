"""Baseline RAG使用示例

演示如何使用内置的BaselineRAG进行快速基准测试。
"""

from pathlib import Path

from pydantic import SecretStr

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate_e2e
from rag_benchmark.prepare import BaselineRAG, prepare_experiment_dataset

# 创建输出目录
output_dir = Path("output/baseline_rag_example")
output_dir.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Baseline RAG示例")
    print("=" * 60)

    # 注意：BaselineRAG需要安装额外依赖
    print("\n注意：使用BaselineRAG需要安装以下依赖：")
    print("  - faiss-cpu 或 faiss-gpu")
    print("  - langchain")
    print("  - langchain-openai (如果使用OpenAI)")
    print("\n安装命令：")
    print("  pip install faiss-cpu langchain langchain-openai")
    print("=" * 60)

    try:
        # 1. 加载Golden Dataset
        print("\n1. 加载Golden Dataset...")
        golden_ds = GoldenDataset("xquad", subset="zh")
        print(f"   加载了 {len(golden_ds)} 条记录")

        # 取一个小子集用于快速演示
        sample_size = 30
        records = golden_ds.head(sample_size)
        print(f"   使用 {sample_size} 条记录进行演示")

        # 2. 创建Baseline RAG
        print("\n2. 创建Baseline RAG...")
        print("   注意：需要配置OpenAI API密钥")
        print("   export OPENAI_API_KEY='your-api-key'")

        # 尝试导入依赖
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            from rag_benchmark.prepare import RAGConfig

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

            # 创建RAG（传入模型实例）
            rag = BaselineRAG(
                embedding_model=embedding,
                llm=llm,
                config=RAGConfig(top_k=3)
            )
            print("   ✓ BaselineRAG创建成功")

            # 3. 索引文档
            print("\n3. 索引文档...")
            # 收集所有reference_contexts作为corpus
            corpus = []
            for record in records:
                if record.reference_contexts:
                    corpus.extend(record.reference_contexts)

            # 去重
            corpus = list(set(corpus))
            print(f"   共 {len(corpus)} 个文档片段")

            rag.index_documents(corpus)
            print("   ✓ 文档索引完成")

            # 4. 测试检索
            print("\n4. 测试检索...")
            test_query = records[0].user_input
            print(f"   查询: {test_query[:100]}...")

            retrieval_result = rag.retrieve(test_query, top_k=3)
            print(f"   检索到 {len(retrieval_result.contexts)} 个相关文档:")
            for i, ctx in enumerate(retrieval_result.contexts, 1):
                print(f"     [{i}] {ctx[:80]}...")

            # 5. 测试生成
            print("\n5. 测试生成...")
            generation_result = rag.generate(test_query, retrieval_result.contexts)
            print(f"   生成答案: {generation_result.response[:200]}...")

            # 6. 准备实验数据集（使用简化的数据集）
            print("\n6. 准备实验数据集...")
            # 创建一个简化的数据集用于测试
            from rag_benchmark.datasets.loaders.base import BaseLoader

            class TestLoader(BaseLoader):
                def __init__(self, records):
                    self.records = records

                def load_golden_records(self):
                    return iter(self.records)

                def load_corpus_records(self):
                    return iter([])

            test_ds = GoldenDataset("test", loader=TestLoader(records))
            exp_ds = prepare_experiment_dataset(test_ds, rag, batch_size=10)
            print(f"   ✓ 实验数据集准备完成")

            # 7. 评测
            print("\n7. 评测Baseline RAG...")
            result = evaluate_e2e(
                exp_ds,
                experiment_name="baseline_rag",
                llm=llm,
                embeddings=embedding
            )

            # 8. 显示结果
            print("\n" + "=" * 60)
            print("评测结果")
            print("=" * 60)
            df = result.to_pandas()

            # 计算平均分
            metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            for metric in metrics:
                if metric in df.columns:
                    mean_score = df[metric].mean()
                    print(f"{metric:20s}: {mean_score:.4f}")

            # 9. 保存结果
            print("\n" + "=" * 60)
            print("保存结果")
            print("=" * 60)

            result_path = output_dir / "baseline_rag_results.csv"
            df.to_csv(result_path, index=False)
            print(f"✓ 评测结果已保存: {result_path}")

            print("\n" + "=" * 60)
            print("评测完成！")
            print("=" * 60)

        except ImportError as e:
            print(f"\n⚠ 缺少依赖: {e}")
            print("\n请安装所需依赖：")
            print("  pip install faiss-cpu langchain langchain-openai")
            print("\n或者使用DummyRAG/SimpleRAG进行演示")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
