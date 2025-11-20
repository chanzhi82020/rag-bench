"""示例：使用自定义LLM和Embedding模型进行评测

演示如何使用自定义的LLM和Embedding模型（如DeepSeek、智谱等）进行RAG评测

Migration Note:
    The ragas library now uses langchain LLMs and embeddings for evaluation.
    Use ChatOpenAI from langchain_openai for custom model providers.
    
    For evaluation, use langchain LLMs:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(model="gpt-4", api_key="...", base_url="...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key="...")
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate
from rag_benchmark.prepare import DummyRAG, RAGConfig, prepare_experiment_dataset
from ragas.metrics import answer_relevancy, faithfulness


def example_with_custom_models():
    """使用自定义模型进行评测
    
    This example demonstrates using langchain LLMs with custom API endpoints
    for ragas evaluation.
    """
    print("=" * 80)
    print("Example: Evaluation with Custom Models")
    print("=" * 80)

    # 1. 配置自定义模型
    print("\n1. Configuring custom models...")
    
    # Create DeepSeek LLM via NVIDIA API using langchain
    # Note: ragas evaluation requires langchain LLMs, not instructor LLMs
    eval_llm = ChatOpenAI(
        model="deepseek-ai/deepseek-v3.1",
        api_key=SecretStr("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"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0,
    )
    # Create 智谱 Embedding model using langchain
    embedding_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=SecretStr("7f08f66caad549708238a57e0f7f33f7.EfQ9HoYpYZqBCRFX"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

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
        print(f"    {result}")

        print("\n✓ Example completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        print("   Please check:")
        print("   1. API keys are valid")
        print("   2. Network connection is stable")
        print("   3. API endpoints are accessible")


def example_with_different_providers():
    """演示使用不同提供商的模型
    
    Shows how to use langchain LLMs with various model providers for ragas evaluation.
    """
    print("=" * 80)
    print("Example: Using Different Model Providers")
    print("=" * 80)
    
    print("\n可用的模型配置示例 (Using Langchain):")
    print("\n1. OpenAI (官方):")
    print("""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Create LLM and embeddings using langchain
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="your-openai-api-key",
        temperature=0.0
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key="your-openai-api-key"
    )
    """)
    
    print("\n2. DeepSeek (通过NVIDIA API):")
    print("""
    from langchain_openai import ChatOpenAI
    
    # Use ChatOpenAI with custom base_url for NVIDIA API
    llm = ChatOpenAI(
        model="deepseek-ai/deepseek-v3.1",
        api_key="your-nvidia-api-key",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.0
    )
    """)
    
    print("\n3. 智谱AI (GLM):")
    print("""
    from langchain_openai import OpenAIEmbeddings
    
    # Use OpenAIEmbeddings with custom API base for GLM
    embeddings = OpenAIEmbeddings(
        model="embedding-3",
        openai_api_key="your-glm-api-key",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    """)
    
    print("\n4. Azure OpenAI:")
    print("""
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    
    # Use Azure-specific langchain classes
    llm = AzureChatOpenAI(
        azure_deployment="your-deployment-name",
        api_key="your-azure-api-key",
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_version="2024-02-15-preview"
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="your-embedding-deployment",
        api_key="your-azure-api-key",
        azure_endpoint="https://your-resource.openai.azure.com/"
    )
    """)
    
    print("\n5. 本地模型 (Ollama):")
    print("""
    from langchain_openai import ChatOpenAI
    
    # Use ChatOpenAI with Ollama endpoint
    llm = ChatOpenAI(
        model="llama2",
        api_key="ollama",  # Ollama doesn't require a real API key
        base_url="http://localhost:11434/v1",
        temperature=0.0
    )
    """)
    
    print("\n" + "=" * 80)
    print("Migration Guide:")
    print("=" * 80)
    print("""
    For ragas evaluation, use langchain LLMs and embeddings:
    
    1. Import from langchain_openai:
       - ChatOpenAI for LLMs
       - OpenAIEmbeddings for embeddings
    
    2. Configure with your API credentials and endpoints
    
    3. Pass directly to evaluate():
       result = evaluate(
           dataset=exp_ds,
           metrics=[faithfulness, answer_relevancy],
           llm=llm,
           embeddings=embeddings
       )
    
    Note: llm_factory() and embedding_factory() create instructor-based
    instances for structured output, not for evaluation. Use langchain
    classes for evaluation instead.
    
    For more details, see: https://docs.ragas.io/en/latest/
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
