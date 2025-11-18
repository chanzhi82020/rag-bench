# RAG Benchmark Prepare Module

准备模块（Prepare Module）是RAG评测流程的第一个核心阶段，负责将Golden Dataset转换为Experiment Dataset，通过调用RAG系统填充检索上下文和生成答案。

## 功能概述

- **实验数据集Schema**: 标准化的数据格式，兼容RAGAS
- **RAG系统接口**: 抽象接口，支持集成任何RAG系统
- **自动化准备**: 批量处理Golden Dataset，自动填充实验数据
- **错误处理**: 健壮的错误处理和失败记录追踪
- **数据持久化**: 保存和加载JSONL格式的实验数据集
- **示例实现**: DummyRAG和SimpleRAG用于测试和学习

## 快速开始

### 安装

```bash
# 安装依赖
uv sync

# 或使用pip
pip install pydantic tqdm
```

### 基本使用

```python
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, DummyRAG

# 1. 加载Golden Dataset
golden_ds = GoldenDataset("hotpotqa")

# 2. 创建RAG系统实例
rag = DummyRAG()

# 3. 准备实验数据集（返回RAGAS的EvaluationDataset）
exp_ds = prepare_experiment_dataset(golden_ds, rag)

# 4. 保存结果（直接使用RAGAS API）
exp_ds.to_jsonl("output/experiment.jsonl")

# 5. 加载结果（直接使用RAGAS API）
loaded_ds = EvaluationDataset.from_jsonl("output/experiment.jsonl")

# 6. 查看统计信息
print(f"Dataset length: {len(exp_ds)}")
print(f"First sample: {exp_ds[0].user_input}")
```

## 核心概念

### Experiment Dataset

本模块直接使用RAGAS的数据结构：

- `SingleTurnSample`: 单条实验记录
- `EvaluationDataset`: 实验数据集容器

```python
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# 创建实验记录
sample = SingleTurnSample(
    user_input="用户问题",              # 用户输入
    reference="正确答案",                # 参考答案
    reference_contexts=["参考上下文"],   # 参考上下文列表
    retrieved_contexts=["检索上下文"],   # RAG检索到的上下文
    response="生成的答案",               # RAG生成的答案
    reference_context_ids=["id1"],      # 可选：参考上下文ID
    retrieved_context_ids=["id2"],      # 可选：检索上下文ID
)

# 创建数据集
dataset = EvaluationDataset(samples=[sample])
```

**优势**：
- 直接兼容RAGAS评测
- 无需格式转换
- 使用RAGAS的原生API

### RAG Interface

所有RAG系统必须实现`RAGInterface`接口：

```python
class RAGInterface(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """检索相关上下文"""
        pass
    
    @abstractmethod
    def generate(self, query: str, contexts: List[str]) -> str:
        """基于上下文生成答案"""
        pass
```

## 集成自定义RAG系统

### 方式1: 实现RAGInterface

```python
from rag_benchmark.prepare import RAGInterface, RAGConfig

class MyRAG(RAGInterface):
    def __init__(self, config: Optional[RAGConfig] = None):
        super().__init__(config)
        # 初始化你的RAG系统
        self.retriever = MyRetriever()
        self.generator = MyGenerator()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        k = top_k or self.config.top_k
        # 实现检索逻辑
        results = self.retriever.search(query, k)
        return [doc.text for doc in results]
    
    def generate(self, query: str, contexts: List[str]) -> str:
        # 实现生成逻辑
        prompt = self._build_prompt(query, contexts)
        return self.generator.generate(prompt)
```

### 方式2: 使用现有RAG框架

如果你使用LangChain、LlamaIndex等框架：

```python
from langchain.chains import RetrievalQA
from rag_benchmark.prepare import RAGInterface

class LangChainRAG(RAGInterface):
    def __init__(self, qa_chain: RetrievalQA, config=None):
        super().__init__(config)
        self.qa_chain = qa_chain
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        # 使用LangChain的检索器
        docs = self.qa_chain.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs[:top_k or self.config.top_k]]
    
    def generate(self, query: str, contexts: List[str]) -> str:
        # 使用LangChain的生成
        return self.qa_chain.run(query)
```

## 配置选项

### RAGConfig

```python
from rag_benchmark.prepare import RAGConfig

config = RAGConfig(
    # 检索配置
    top_k=5,                          # 检索top-k个结果
    similarity_threshold=0.7,         # 相似度阈值
    retrieval_mode="dense",           # dense, sparse, hybrid
    
    # 生成配置
    max_length=512,                   # 最大生成长度
    temperature=0.7,                  # 生成温度
    top_p=0.9,                        # nucleus sampling
    
    # 其他配置
    batch_size=1,                     # 批处理大小
    timeout=30,                       # 超时时间（秒）
)
```

### prepare_experiment_dataset参数

```python
exp_ds = prepare_experiment_dataset(
    golden_dataset=golden_ds,         # Golden数据集
    rag_system=rag,                   # RAG系统实例
    top_k=5,                          # 覆盖RAG配置的top_k
    batch_size=1,                     # 批处理大小
    show_progress=True,               # 显示进度条
    skip_on_error=True,               # 遇到错误时跳过
)
```

## 批量处理

对于大规模数据集，可以使用批量处理提高效率：

```python
# 如果你的RAG系统支持批量API，重写批量方法
class BatchRAG(RAGInterface):
    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> List[List[str]]:
        # 批量检索实现
        return self.retriever.batch_search(queries, top_k)
    
    def batch_generate(self, queries: List[str], contexts_list: List[List[str]]) -> List[str]:
        # 批量生成实现
        return self.generator.batch_generate(queries, contexts_list)

# 使用批量处理
exp_ds = prepare_experiment_dataset(
    golden_ds, 
    rag, 
    batch_size=32  # 每次处理32条记录
)
```

## 错误处理

### 跳过失败的记录

```python
# 默认行为：跳过失败的记录，继续处理
exp_ds = prepare_experiment_dataset(
    golden_ds, 
    rag, 
    skip_on_error=True  # 默认值
)

# 查看失败记录会在日志中显示
```

### 遇到错误时停止

```python
# 遇到任何错误立即停止
try:
    exp_ds = prepare_experiment_dataset(
        golden_ds, 
        rag, 
        skip_on_error=False
    )
except PrepareError as e:
    print(f"Preparation failed: {e}")
```

## 数据持久化

### 保存实验数据集

```python
from rag_benchmark.prepare import save_experiment_dataset

# 保存为JSONL格式
save_experiment_dataset(
    dataset=exp_ds,
    output_path="output/experiment.jsonl",
    overwrite=True  # 覆盖已存在的文件
)
```

### 加载实验数据集

```python
from rag_benchmark.prepare import load_experiment_dataset

# 加载并验证
exp_ds = load_experiment_dataset(
    input_path="output/experiment.jsonl",
    validate=True  # 验证数据格式
)

# 查看统计信息
print(exp_ds.stats())
```

## 数据验证

### 验证数据集完整性

```python
# 验证数据集
validation = exp_ds.validate()

print(f"Is valid: {validation['is_valid']}")
print(f"Errors: {validation['errors']}")
print(f"Warnings: {validation['warnings']}")
```

### 过滤完整记录

```python
# 只保留包含retrieved_contexts和response的记录
complete_ds = exp_ds.filter_complete()
print(f"Complete records: {len(complete_ds)}/{len(exp_ds)}")
```

## RAGAS集成

### 转换为RAGAS格式

```python
# 转换为RAGAS数据集格式
ragas_data = exp_ds.to_ragas_dataset()

# RAGAS格式示例
# {
#     "question": "What is Python?",
#     "ground_truth": "Python is a programming language.",
#     "contexts": ["Python is a high-level language..."],
#     "answer": "Python is a programming language...",
#     "reference_contexts": ["Python is a high-level language..."]
# }

# 使用RAGAS评测
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance

results = evaluate(
    dataset=ragas_data,
    metrics=[faithfulness, answer_relevance]
)
```

## 示例实现

### DummyRAG

用于测试和演示的虚拟RAG系统：

```python
from rag_benchmark.prepare import DummyRAG

rag = DummyRAG(seed=42)  # 设置随机种子以获得可复现结果
contexts = rag.retrieve("What is Python?")
answer = rag.generate("What is Python?", contexts)
```

### SimpleRAG

基于简单关键词匹配的RAG系统：

```python
from rag_benchmark.prepare import SimpleRAG

corpus = [
    "Python is a programming language.",
    "Java is also popular.",
    "Machine learning uses algorithms."
]

rag = SimpleRAG(corpus=corpus)
contexts = rag.retrieve("What is Python?")  # 基于关键词匹配
answer = rag.generate("What is Python?", contexts)  # 基于模板生成
```

## 完整示例

查看 `examples/` 目录获取完整示例：

- `prepare_experiment_dataset.py` - 基本使用示例
- `custom_rag_integration.py` - 自定义RAG集成示例

运行示例：

```bash
python src/rag_benchmark/examples/prepare_experiment_dataset.py
python src/rag_benchmark/examples/custom_rag_integration.py
```

## API参考

### 核心函数

#### prepare_experiment_dataset

```python
def prepare_experiment_dataset(
    golden_dataset: GoldenDataset,
    rag_system: RAGInterface,
    top_k: Optional[int] = None,
    batch_size: int = 1,
    show_progress: bool = True,
    skip_on_error: bool = True,
) -> ExperimentDataset
```

从Golden Dataset准备Experiment Dataset。

#### save_experiment_dataset

```python
def save_experiment_dataset(
    dataset: ExperimentDataset,
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> None
```

保存实验数据集到JSONL文件。

#### load_experiment_dataset

```python
def load_experiment_dataset(
    input_path: Union[str, Path],
    validate: bool = True,
) -> ExperimentDataset
```

从JSONL文件加载实验数据集。

### 数据类

#### ExperimentRecord

实验记录的数据结构。

**方法**:
- `to_dict()` - 转换为字典
- `to_ragas_format()` - 转换为RAGAS格式

#### ExperimentDataset

实验数据集容器。

**方法**:
- `stats()` - 获取统计信息
- `validate()` - 验证数据集
- `to_ragas_dataset()` - 转换为RAGAS格式
- `filter_complete()` - 过滤完整记录

### 接口类

#### RAGInterface

RAG系统抽象接口。

**必须实现的方法**:
- `retrieve(query, top_k)` - 检索上下文
- `generate(query, contexts)` - 生成答案

**可选优化的方法**:
- `batch_retrieve(queries, top_k)` - 批量检索
- `batch_generate(queries, contexts_list)` - 批量生成
- `batch_retrieve_and_generate(queries, top_k)` - 批量检索和生成

#### RAGConfig

RAG系统配置。

**主要参数**:
- `top_k` - 检索top-k
- `max_length` - 最大生成长度
- `temperature` - 生成温度
- `batch_size` - 批处理大小

## 最佳实践

1. **使用批量处理**: 对于大数据集，实现批量方法可以显著提高性能
2. **设置合理的超时**: 避免单个请求阻塞整个流程
3. **启用错误跳过**: 使用`skip_on_error=True`确保部分失败不影响整体
4. **验证数据**: 保存前验证数据集完整性
5. **使用进度条**: 启用`show_progress=True`监控处理进度
6. **保存中间结果**: 定期保存实验数据集，避免重复处理

## 故障排除

### 常见问题

**Q: RAG系统调用失败怎么办？**

A: 使用`skip_on_error=True`跳过失败的记录，查看日志了解失败原因。

**Q: 如何提高处理速度？**

A: 
1. 实现批量方法（`batch_retrieve`, `batch_generate`）
2. 增加`batch_size`参数
3. 使用更快的检索和生成模型

**Q: 数据格式验证失败？**

A: 检查ExperimentRecord的所有必需字段是否正确填充，使用`validate()`方法查看详细错误。

**Q: 如何处理超时？**

A: 在RAGConfig中设置合理的`timeout`值，并在RAG实现中处理超时异常。

## 下一步

完成实验数据集准备后，可以：

1. 使用`evaluate`模块进行评测
2. 使用`analysis`模块分析结果
3. 对比不同RAG系统的性能

## 许可证

本模块是RAG Benchmark框架的一部分。查看项目许可证了解详情。
