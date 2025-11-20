# RAG Benchmark Framework

一个用于评测RAG（Retrieval-Augmented Generation）系统性能的Python框架。该框架集成RAGAS评估框架，支持端到端和分阶段的RAG评测。

## 特性

- ✅ **Golden Dataset管理**: 标准化的数据集格式，支持多种公开数据集
- ✅ **实验数据集准备**: 自动化填充检索上下文和生成答案
- 🚧 **评测指标**: 集成RAGAS，支持检索和生成阶段的多种指标
- 🚧 **结果分析**: 对比分析不同RAG系统的性能
- 🚧 **Baseline RAG**: 内置RAG系统用于快速基准测试

## 快速开始

### 环境设置

使用conda创建虚拟环境（推荐）：

```bash
# 方式1: 使用environment.yml
conda env create -f environment.yml
conda activate rag-bench
uv sync

# 方式2: 手动创建
conda create -n rag-bench python=3.11 -y
conda activate rag-bench
pip install uv
uv sync
```

详细设置说明请查看 [SETUP.md](SETUP.md)

### 快速上手

查看 [QUICK_START.md](QUICK_START.md) 获取快速入门指南，包括：
- 基本使用流程
- 常见问题解答
- API使用示例

### 基本使用

#### 1. 加载Golden Dataset

```python
from rag_benchmark.datasets import GoldenDataset

# 加载数据集
dataset = GoldenDataset("xquad", subset="zh")

# 查看统计信息
print(dataset.stats())

# 遍历记录
for record in dataset:
    print(f"Question: {record.user_input}")
    print(f"Answer: {record.reference}")
    break
```

#### 2. 准备实验数据集

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import (
    prepare_experiment_dataset,
    DummyRAG,
)

# 加载Golden Dataset
golden_ds = GoldenDataset("xquad", subset="zh")

# 创建RAG系统（这里使用DummyRAG作为示例）
rag = DummyRAG()

# 准备实验数据集
exp_ds = prepare_experiment_dataset(golden_ds, rag)

# 保存结果
exp_ds.to_jsonl("output/experiment.json")
```


#### 3. 集成自定义RAG系统

```python
from rag_benchmark.prepare import RAGInterface, RAGConfig, RetrievalResult, GenerationResult

class MyRAG(RAGInterface):
    def __init__(self, config=None):
        super().__init__(config)
        # 初始化你的RAG系统
        
    def retrieve(self, query, top_k=None):
        # 实现检索逻辑
        contexts = ["context1", "context2"]
        return RetrievalResult(contexts=contexts)
    
    def generate(self, query, contexts):
        # 实现生成逻辑
        answer = "generated answer"
        return GenerationResult(response=answer)
    
    # 可选：实现批量处理以提升性能
    def batch_retrieve(self, queries, top_k=None):
        # 批量检索逻辑
        return [self.retrieve(q, top_k) for q in queries]
    
    def batch_generate(self, queries, contexts_list):
        # 批量生成逻辑
        return [self.generate(q, c) for q, c in zip(queries, contexts_list)]

# 使用自定义RAG
my_rag = MyRAG()
exp_ds = prepare_experiment_dataset(golden_ds, my_rag)
```

#### 4. 评测RAG系统

```python
from rag_benchmark.evaluate import evaluate_e2e

# 运行评测
result = evaluate_e2e(
    dataset=exp_ds,
    experiment_name="my_rag_evaluation"
)

# 查看结果
df = result.to_pandas()
print(f"Faithfulness: {df['faithfulness'].mean():.4f}")

# 保存结果
df.to_csv("output/evaluation.csv", index=False)
```

#### 5. 性能优化（批量处理）

BaselineRAG支持批量处理，可显著提升性能（2-5倍）：

```python
# 批量检索（推荐用于大规模评测）
queries = ["query1", "query2", "query3"]
retrieval_results = rag.batch_retrieve(queries, top_k=3)

# 批量生成
contexts_list = [r.contexts for r in retrieval_results]
generation_results = rag.batch_generate(queries, contexts_list)

# 查看性能对比示例
# python examples/batch_processing_demo.py
```

## 项目结构

```
rag_benchmark/
├── datasets/           # Golden Dataset管理 ✅
│   ├── data/          # 内置数据集
│   ├── loaders/       # 数据加载器
│   ├── converters/    # 数据转换器
│   └── validators/    # 数据验证器
│
├── prepare/           # 实验数据集准备 ✅
│   ├── schema.py      # 数据Schema定义
│   ├── rag_interface.py  # RAG接口
│   ├── prepare.py     # 核心prepare函数
│   ├── dummy_rag.py   # 示例RAG实现
│   └── baseline_rag.py # Baseline RAG实现
│
├── evaluate/          # 评测模块 ✅
│   ├── evaluator.py   # 核心评估器
│   ├── results.py     # 结果管理
│   ├── metrics_retrieval.py  # 检索指标
│   └── README.md      # 详细文档
│
├── analysis/          # 结果分析 ✅
│   ├── compare.py     # 对比分析
│   └── visualize.py   # 可视化
│
└── examples/          # 示例代码 ✅
    ├── load_dataset.py
    ├── prepare_experiment_dataset.py
    ├── custom_rag_integration.py
    ├── evaluate_rag_system.py
    ├── evaluate_retrieval.py
    ├── evaluate_generation.py
    ├── evaluate_with_custom_models.py
    ├── compare_rag_systems.py
    └── baseline_rag_example.py
```

## 模块文档

- [Datasets模块](src/rag_benchmark/datasets/README.md) - Golden Dataset管理
- [Prepare模块](src/rag_benchmark/prepare/README.md) - 实验数据集准备
- [Evaluate模块](src/rag_benchmark/evaluate/README.md) - RAG系统评测
- [Analysis模块](src/rag_benchmark/analysis/README.md) - 结果分析和对比

## 示例

查看 `examples/` 目录获取完整示例：

```bash
# 加载数据集示例
python examples/load_dataset.py

# 准备实验数据集示例
python examples/prepare_experiment_dataset.py

# 自定义RAG集成示例
python examples/custom_rag_integration.py

# 完整端到端评测示例
python examples/evaluate_rag_system.py

# 检索阶段评测示例
python examples/evaluate_retrieval.py

# 生成阶段评测示例
python examples/evaluate_generation.py

# 使用自定义模型评测
python examples/evaluate_with_custom_models.py

# 对比多个RAG系统
python examples/compare_rag_systems.py

# 使用Baseline RAG
python examples/baseline_rag_example.py

# 简单对比示例（推荐新手）
python examples/simple_comparison_demo.py

# 批量处理性能对比
python examples/batch_processing_demo.py
```

## 支持的数据集

- **HotpotQA**: 多跳问答数据集
- **Natural Questions**: Google搜索真实用户问题
- **XQuAD**: 跨语言问答数据集（支持中文）
- **Customer Service**: 智能客服数据集（私有）

## 开发路线图

### ✅ 已完成

- [x] Golden Dataset管理模块
- [x] 数据加载和验证
- [x] 数据集转换工具
- [x] Prepare模块（实验数据集准备）
- [x] RAG系统接口
- [x] 示例RAG实现
- [x] Evaluate模块（评测指标）
- [x] RAGAS集成（端到端评测）
- [x] 检索阶段指标（recall@k, precision@k, MRR, NDCG）
- [x] 生成阶段指标（faithfulness, answer_correctness）
- [x] 自定义模型支持
- [x] 完整示例代码（7个示例）

### ✅ 已完成（v0.2.0）

- [x] Evaluate模块（评测指标）
- [x] RAGAS集成
- [x] 检索阶段指标（recall@k, precision@k, MRR, NDCG）
- [x] 生成阶段指标（faithfulness, answer_correctness）
- [x] 自定义模型支持（langchain集成）
- [x] Analysis模块（结果分析和对比）
- [x] Baseline RAG实现（FAISS + LLM）
- [x] 可视化工具（matplotlib集成）
- [x] 完整示例代码（9个示例）

### 🚧 进行中

- [ ] 更多数据集支持
- [ ] Web UI界面

### 📋 计划中

- [ ] 从文档自动构建Golden Dataset
- [ ] Prompt-based数据生成
- [ ] 数据集自动清洗与增强
- [ ] 更多Baseline RAG变体

## 技术栈

- **Python**: >=3.11
- **datasets**: >=4.4.1 - 数据集加载
- **ragas**: >=0.3.9 - RAG评估框架
- **pydantic**: >=2.0.0 - 数据验证
- **tqdm**: >=4.64.0 - 进度显示

### 可选依赖

- **Analysis模块**: matplotlib, numpy, pandas
- **Baseline RAG**: faiss-cpu, langchain, langchain-openai

安装可选依赖：
```bash
# 安装分析工具
uv pip install -e ".[analysis]"

# 安装Baseline RAG
uv pip install -e ".[baseline]"

# 安装所有可选依赖
uv pip install -e ".[analysis,baseline]"
```

## 开发

### 运行测试

```bash
# 简单测试
python test_prepare_simple.py

# 完整测试（需要先实现）
pytest tests/
```

### 代码格式化

```bash
# 格式化代码
black src/

# 排序导入
isort src/

# 类型检查
mypy src/
```

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 许可证

MIT License

## 致谢

本项目参考了以下优秀框架：
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG评估框架
- [ARES](https://github.com/stanford-futuredata/ARES) - 自动RAG评估系统
- [BEIR](https://github.com/beir-cellar/beir) - 信息检索基准测试

## 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/rag-bench/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/rag-bench/discussions)

## 更新日志

### v0.2.0 (2025-11-20)

**Evaluate模块**
- ✅ 完成Evaluate模块
- ✅ 集成RAGAS评测框架
- ✅ 实现传统IR指标（recall@k, precision@k, MRR, NDCG, MAP）
- ✅ 支持分阶段评测（检索/生成/端到端）
- ✅ 支持自定义LLM和Embedding模型

**Analysis模块**
- ✅ 完成Analysis模块
- ✅ 多模型结果对比
- ✅ 指标统计分析（均值、标准差、最大/最小值）
- ✅ 可视化图表生成（matplotlib集成）
- ✅ 最差样本分析

**Baseline RAG**
- ✅ 实现BaselineRAG（FAISS + LLM）
- ✅ 支持自定义Embedding和LLM模型
- ✅ 文档索引和向量检索
- ✅ 端到端查询功能
- ✅ 批量处理优化（batch_retrieve和batch_generate）

**示例和文档**
- ✅ 新增2个完整示例（对比分析、Baseline RAG）
- ✅ 完善文档和API参考
- ✅ 总计9个示例代码

### v0.1.0 (2025-11-19)

- ✅ 实现datasets模块
- ✅ 实现prepare模块
- ✅ 支持HotpotQA、NQ、XQuAD数据集
- ✅ 提供DummyRAG和SimpleRAG示例
- ✅ 完整的文档和示例代码
