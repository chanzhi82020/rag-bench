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
    save_experiment_dataset,
    DummyRAG,
)

# 加载Golden Dataset
golden_ds = GoldenDataset("xquad", subset="zh")

# 创建RAG系统（这里使用DummyRAG作为示例）
rag = DummyRAG()

# 准备实验数据集
exp_ds = prepare_experiment_dataset(golden_ds, rag)

# 保存结果
save_experiment_dataset(exp_ds, "output/experiment.jsonl")

# 查看统计
print(exp_ds.stats())
```

#### 3. 集成自定义RAG系统

```python
from rag_benchmark.prepare import RAGInterface, RAGConfig

class MyRAG(RAGInterface):
    def __init__(self, config=None):
        super().__init__(config)
        # 初始化你的RAG系统
        
    def retrieve(self, query, top_k=None):
        # 实现检索逻辑
        return ["context1", "context2"]
    
    def generate(self, query, contexts):
        # 实现生成逻辑
        return "generated answer"

# 使用自定义RAG
my_rag = MyRAG()
exp_ds = prepare_experiment_dataset(golden_ds, my_rag)
```

## 项目结构

```
rag_benchmark/
├── datasets/           # Golden Dataset管理
│   ├── data/          # 内置数据集
│   ├── loaders/       # 数据加载器
│   ├── converters/    # 数据转换器
│   └── validators/    # 数据验证器
│
├── prepare/           # 实验数据集准备 ✅
│   ├── schema.py      # 数据Schema定义
│   ├── rag_interface.py  # RAG接口
│   ├── prepare.py     # 核心prepare函数
│   └── dummy_rag.py   # 示例RAG实现
│
├── evaluate/          # 评测模块 🚧
│   ├── metrics/       # 评测指标
│   └── evaluator.py   # 评估器
│
├── analysis/          # 结果分析 🚧
│   ├── compare.py     # 对比分析
│   └── visualize.py   # 可视化
│
└── examples/          # 示例代码
    ├── load_dataset.py
    ├── prepare_experiment_dataset.py
    └── custom_rag_integration.py
```

## 模块文档

- [Datasets模块](src/rag_benchmark/datasets/README.md) - Golden Dataset管理
- [Prepare模块](src/rag_benchmark/prepare/README.md) - 实验数据集准备
- Evaluate模块 - 即将推出
- Analysis模块 - 即将推出

## 示例

查看 `src/rag_benchmark/examples/` 目录获取完整示例：

```bash
# 加载数据集示例
python src/rag_benchmark/examples/load_dataset.py

# 准备实验数据集示例
python src/rag_benchmark/examples/prepare_experiment_dataset.py

# 自定义RAG集成示例
python src/rag_benchmark/examples/custom_rag_integration.py
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

### 🚧 进行中

- [ ] Evaluate模块（评测指标）
- [ ] RAGAS集成
- [ ] 检索阶段指标（recall@k, precision@k, MRR, NDCG）
- [ ] 生成阶段指标（faithfulness, grounding, coherence）

### 📋 计划中

- [ ] Analysis模块（结果分析）
- [ ] Baseline RAG实现
- [ ] 性能对比工具
- [ ] 可视化报告
- [ ] 更多数据集支持

## 技术栈

- **Python**: >=3.11
- **datasets**: >=4.4.1 - 数据集加载
- **ragas**: >=0.3.9 - RAG评估框架
- **pydantic**: >=2.0.0 - 数据验证
- **tqdm**: >=4.64.0 - 进度显示

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

### v0.1.0 (2025-11-19)

- ✅ 实现datasets模块
- ✅ 实现prepare模块
- ✅ 支持HotpotQA、NQ、XQuAD数据集
- ✅ 提供DummyRAG和SimpleRAG示例
- ✅ 完整的文档和示例代码
