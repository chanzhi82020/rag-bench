"""示例：准备实验数据集

演示如何使用prepare模块从Golden Dataset生成Experiment Dataset
"""
from ragas import EvaluationDataset

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import (
    prepare_experiment_dataset,
    DummyRAG,
    SimpleRAG,
    RAGConfig,
)


def example_with_dummy_rag():
    """使用DummyRAG准备实验数据集"""
    print("=" * 80)
    print("Example 1: Using DummyRAG")
    print("=" * 80)
    
    # 1. 加载Golden Dataset
    print("\n1. Loading Golden Dataset...")
    try:
        golden_ds = GoldenDataset("xquad", subset="zh")
        print(f"   Loaded {len(golden_ds)} records")
        print(f"   Dataset: {golden_ds}")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        print("   Skipping this example...")
        return
    
    # 2. 创建DummyRAG实例
    print("\n2. Creating DummyRAG instance...")
    config = RAGConfig(top_k=3, max_length=256)
    rag = DummyRAG(config=config, seed=42)
    print(f"   RAG config: top_k={config.top_k}, max_length={config.max_length}")
    
    # 3. 准备实验数据集（只处理前5条记录作为演示）
    print("\n3. Preparing experiment dataset (first 5 records)...")
    # 创建一个只包含前5条记录的子集
    from rag_benchmark.datasets.loaders.base import BaseLoader
    from rag_benchmark.datasets.schemas.golden import GoldenRecord
    
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
    
    exp_ds = prepare_experiment_dataset(
        golden_dataset=golden_sample,
        rag_system=rag,
        show_progress=True,
    )
    
    # 4. 查看第一条记录
    print("\n4. First experiment record:")
    if len(exp_ds) > 0:
        first_record = exp_ds[0]
        print(f"   Question: {first_record.user_input[:100]}...")
        print(f"   Reference: {first_record.reference[:100]}...")
        print(f"   Retrieved contexts: {len(first_record.retrieved_contexts)} contexts")
        print(f"   Response: {first_record.response[:150]}...")
    
    # 5. 保存实验数据集
    print("\n5. Saving experiment dataset...")
    output_path = "output/experiment_dummy.jsonl"
    # 使用UTF-8编码保存，避免Windows上的编码问题
    import json
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in exp_ds:
            f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
    print(f"   Saved to: {output_path}")
    
    # 6. 加载实验数据集
    print("\n6. Loading experiment dataset...")
    # 使用UTF-8编码加载，避免Windows上的编码问题
    import json
    from ragas.dataset_schema import SingleTurnSample
    samples = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(SingleTurnSample(**data))
    loaded_ds = EvaluationDataset(samples=samples)
    print(f"   Loaded {len(loaded_ds)} records")
    

    print("\n✓ Example 1 completed successfully!\n")


def example_with_simple_rag():
    """使用SimpleRAG准备实验数据集"""
    print("=" * 80)
    print("Example 2: Using SimpleRAG with custom corpus")
    print("=" * 80)
    
    # 1. 创建一个小型语料库
    print("\n1. Creating a small corpus...")
    corpus = [
        "Python is a high-level programming language known for its simplicity.",
        "Java is a popular object-oriented programming language.",
        "JavaScript is primarily used for web development.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
    ]
    print(f"   Corpus size: {len(corpus)} documents")
    
    # 2. 创建SimpleRAG实例
    print("\n2. Creating SimpleRAG instance...")
    config = RAGConfig(top_k=2, max_length=128)
    rag = SimpleRAG(corpus=corpus, config=config)
    
    # 3. 创建一个简单的Golden Dataset（手动构造）
    print("\n3. Creating a simple golden dataset...")
    from rag_benchmark.datasets.schemas.golden import GoldenRecord
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class MockLoader(BaseLoader):
        def __init__(self, records):
            self.records = records
        
        def load_golden_records(self):
            return iter(self.records)
        
        def load_corpus_records(self):
            return iter([])
        
        def count_records(self):
            return len(self.records)
    
    golden_records = [
        GoldenRecord(
            user_input="What is Python?",
            reference="Python is a programming language.",
            reference_contexts=["Python is a high-level programming language."],
        ),
        GoldenRecord(
            user_input="What is machine learning?",
            reference="Machine learning is a type of AI.",
            reference_contexts=["Machine learning is a subset of artificial intelligence."],
        ),
    ]
    
    mock_loader = MockLoader(golden_records)
    golden_ds = GoldenDataset("mock", loader=mock_loader)
    print(f"   Created {len(golden_ds)} golden records")
    
    # 4. 准备实验数据集
    print("\n4. Preparing experiment dataset...")
    exp_ds = prepare_experiment_dataset(
        golden_dataset=golden_ds,
        rag_system=rag,
        show_progress=True,
    )
    
    # 5. 查看结果
    print("\n5. Results:")
    for i, record in enumerate(exp_ds):
        print(f"\n   Record {i + 1}:")
        print(f"   Question: {record.user_input}")
        print(f"   Retrieved contexts: {record.retrieved_contexts}")
        print(f"   Generated answer: {record.response}")
    
    print("\n✓ Example 2 completed successfully!\n")


def example_error_handling():
    """演示错误处理"""
    print("=" * 80)
    print("Example 3: Error handling")
    print("=" * 80)
    
    # 创建一个会失败的RAG系统
    from rag_benchmark.prepare import RAGInterface
    
    class FailingRAG(RAGInterface):
        def __init__(self, fail_rate=0.5):
            super().__init__()
            self.fail_rate = fail_rate
            self.call_count = 0
        
        def retrieve(self, query, top_k=None):
            from rag_benchmark.prepare import RetrievalResult
            self.call_count += 1
            if self.call_count % 2 == 0:  # 每隔一次失败
                raise RuntimeError("Simulated retrieval failure")
            return RetrievalResult(
                contexts=["Context 1", "Context 2"],
                context_ids=["id1", "id2"],
                scores=[0.9, 0.8]
            )
        
        def generate(self, query, contexts):
            from rag_benchmark.prepare import GenerationResult
            return GenerationResult(response="Generated answer")
    
    # 创建测试数据
    from rag_benchmark.datasets.schemas.golden import GoldenRecord
    from rag_benchmark.datasets.loaders.base import BaseLoader
    
    class MockLoader(BaseLoader):
        def __init__(self, records):
            self.records = records
        
        def load_golden_records(self):
            return iter(self.records)
        
        def load_corpus_records(self):
            return iter([])
        
        def count_records(self):
            return len(self.records)
    
    golden_records = [
        GoldenRecord(
            user_input=f"Question {i}",
            reference=f"Answer {i}",
            reference_contexts=[f"Context {i}"],
        )
        for i in range(4)
    ]
    
    mock_loader = MockLoader(golden_records)
    golden_ds = GoldenDataset("mock", loader=mock_loader)
    
    print("\n1. Testing with skip_on_error=True (default)...")
    rag = FailingRAG()
    exp_ds = prepare_experiment_dataset(
        golden_dataset=golden_ds,
        rag_system=rag,
        skip_on_error=True,
        show_progress=True,
    )
    print(f"   Successfully processed: {len(exp_ds)} records")
    print(f"   Expected: 2 records (50% failure rate)")
    
    print("\n✓ Example 3 completed successfully!\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Benchmark - Prepare Module Examples")
    print("=" * 80 + "\n")
    
    # 运行示例
    example_with_dummy_rag()
    example_with_simple_rag()
    example_error_handling()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
