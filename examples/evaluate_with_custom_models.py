"""示例：使用自定义LLM和Embedding模型进行评测

演示如何使用自定义的LLM和Embedding模型（如DeepSeek、智谱等）进行RAG评测
"""

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate
from rag_benchmark.prepare import DummyRAG, RAGConfig, prepare_experiment_dataset
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import answer_relevancy, faithfulness


def example_with_custom_models():
    """使用自定义模型进行评测"""
    print("=" * 80)
    print("Example: Evaluation with Custom Models")
    print("=" * 80)

    # 1. 配置自定义模型
    print("\n1. Configuring custom models...")
    
    # DeepSeek LLM (通过NVIDIA API)
    eval_llm = llm_factory(
        model="deepseek-ai/deepseek-v3.1",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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",
    )
    print("   ✓ Configured DeepSeek LLM via NVIDIA API")
    
    # 智谱 Embedding (GLM)
    embedding_model = embedding_factory(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="7f08f66caad549708238a57e0f7f33f7.EfQ9HoYpYZqBCRFX",
    )
    print("   ✓ Configured GLM Embedding model")

    # 2. 准备实验数据
    print("\n2. Preparing experiment dataset...")
    golden_ds = GoldenDataset("xquad", subset="zh")
    
    # 只取前3条记录作为演示
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class SubsetLoader(BaseLoader):
        def __init__(self, original_dataset, limit=3):
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
    
    subset_loader = SubsetLoader(golden_ds, limit=3)
    golden_sample = GoldenDataset("xquad_subset", loader=subset_loader)
    
    rag = DummyRAG(config=RAGConfig(top_k=3))
    exp_ds = prepare_experiment_dataset(golden_sample, rag, show_progress=False)
    print(f"   Prepared {len(exp_ds)} samples")

    # 3. 使用自定义模型进行评测
    print("\n3. Running evaluation with custom models...")
    print("   This may take a few minutes...")
    
    try:
        result = evaluate(
            dataset=exp_ds,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=embedding_model,
            name="custom_models_evaluation",
            show_progress=True,
        )

        # 4. 查看结果
        print("\n4. Evaluation Results:")
        print(f"   Name: {result.name}")
        print(f"   Dataset size: {result.dataset_size}")
        print(f"\n   Scores:")
        for metric_name in result.list_metrics():
            score = result.get_score(metric_name)
            metric_detail = result.get_metric(metric_name)
            print(f"     - {metric_name}:")
            print(f"         Average: {score:.4f}")
            print(f"         Min: {metric_detail.min_score:.4f}")
            print(f"         Max: {metric_detail.max_score:.4f}")
            print(f"         Std: {metric_detail.std_score:.4f}")

        # 5. 保存结果
        print("\n5. Saving results...")
        result.save("output/evaluation_custom_models.json")
        print("   Saved to: output/evaluation_custom_models.json")

        print("\n✓ Example completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        print("   Please check:")
        print("   1. API keys are valid")
        print("   2. Network connection is stable")
        print("   3. API endpoints are accessible")


def example_with_different_providers():
    """演示使用不同提供商的模型"""
    print("=" * 80)
    print("Example: Using Different Model Providers")
    print("=" * 80)
    
    print("\n可用的模型配置示例：")
    print("\n1. OpenAI (官方):")
    print("""
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    
    llm = llm_factory(
        model="gpt-4",
        api_key="your-openai-api-key"
    )
    embeddings = embedding_factory(
        model="text-embedding-3-small",
        api_key="your-openai-api-key"
    )
    """)
    
    print("\n2. DeepSeek (通过NVIDIA API):")
    print("""
    llm = llm_factory(
        model="deepseek-ai/deepseek-v3.1",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="your-nvidia-api-key"
    )
    """)
    
    print("\n3. 智谱AI (GLM):")
    print("""
    embeddings = embedding_factory(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="your-glm-api-key"
    )
    """)
    
    print("\n4. Azure OpenAI:")
    print("""
    llm = llm_factory(
        model="gpt-4",
        base_url="https://your-resource.openai.azure.com/",
        api_key="your-azure-api-key",
        api_version="2024-02-15-preview"
    )
    """)
    
    print("\n5. 本地模型 (Ollama):")
    print("""
    llm = llm_factory(
        model="llama2",
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama不需要真实的API key
    )
    """)


def example_with_run_config():
    """使用RunConfig配置超时和重试"""
    print("=" * 80)
    print("Example: Using RunConfig for Timeout and Retries")
    print("=" * 80)
    
    print("\n配置超时和重试示例：")
    print("""
    from ragas.run_config import RunConfig
    from rag_benchmark.evaluate import evaluate
    
    # 配置运行参数
    run_config = RunConfig(
        timeout=60,  # 60秒超时
        max_retries=3,  # 最多重试3次
        max_wait=10,  # 最大等待时间10秒
    )
    
    result = evaluate(
        dataset=exp_ds,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=embedding_model,
        run_config=run_config,
        name="with_run_config"
    )
    """)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Benchmark - Custom Models Evaluation Examples")
    print("=" * 80 + "\n")

    # 运行主示例
    example_with_custom_models()
    
    # 显示其他配置示例
    print("\n")
    example_with_different_providers()
    
    print("\n")
    example_with_run_config()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
