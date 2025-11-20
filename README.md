# RAG Benchmark Framework

一个用于评测RAG（Retrieval-Augmented Generation）系统性能的Python框架，集成RAGAS评估框架，支持端到端和分阶段的RAG评测。

## ✨ 特性

- 📊 **Golden Dataset管理**: 标准化的数据集格式，支持多种公开数据集
- 🤖 **模型仓库**: 统一管理模型配置（base_url, api_key, model_name）
- 🔧 **实验数据集准备**: 自动化填充检索上下文和生成答案
- 📈 **评测指标**: 集成RAGAS，支持检索和生成阶段的多种指标
- 📉 **结果分析**: 对比分析不同RAG系统的性能
- 🌐 **Web界面**: React前端 + FastAPI后端，可视化操作
- 🔄 **异步任务**: 支持长时间评测任务，断点续传
- 🎯 **Baseline RAG**: 内置RAG系统用于快速基准测试

## 🚀 快速开始

### 安装

```bash
# 创建环境
conda create -n rag-bench python=3.11 -y
conda activate rag-bench

# 安装依赖
pip install uv
uv sync
uv pip install -e ".[api]"
```

### 启动Web服务

```bash
# 终端1: 启动API
./start_api.sh

# 终端2: 启动前端
./start_frontend.sh

# 访问 http://localhost:3000
```

### Python API使用

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import BaselineRAG, RAGConfig, prepare_experiment_dataset
from rag_benchmark.evaluate import evaluate_e2e
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 加载数据集
dataset = GoldenDataset("xquad", subset="zh")

# 创建RAG系统
rag = BaselineRAG(
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    config=RAGConfig(top_k=5)
)

# 准备实验数据集
exp_ds = prepare_experiment_dataset(dataset.sample(10), rag)

# 运行评测
result = evaluate_e2e(exp_ds, experiment_name="test")
print(result.to_pandas()[['faithfulness', 'answer_correctness']].mean())
```

## 📚 文档

- [快速开始](docs/QUICKSTART.md) - 5分钟上手指南
- [API文档](docs/API.md) - 完整的API接口说明
- [架构设计](docs/ARCHITECTURE.md) - 系统架构和设计

## 🎯 主要功能

### 1. 模型仓库

统一管理所有模型配置，避免重复输入API Key：

- 注册LLM和Embedding模型
- 配置base_url和api_key
- 在创建RAG和评测时引用模型

### 2. 数据集管理

支持多种公开数据集：

- **XQuAD**: 跨语言问答（支持中文）
- **HotpotQA**: 多跳问答
- **Natural Questions**: Google搜索真实问题

### 3. RAG系统评测

支持三种评测模式：

- **端到端评测**: 完整的RAG流程评测
- **检索阶段评测**: 只评测检索质量
- **生成阶段评测**: 只评测生成质量

### 4. 评测指标

#### 检索指标
- Recall@K, Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Context Recall, Context Precision

#### 生成指标
- Faithfulness (忠实度)
- Answer Correctness (答案正确性)
- Answer Relevancy (答案相关性)

### 5. 结果分析

- 多模型性能对比
- 指标统计分析
- 可视化图表
- 最差样本分析

## 🏗️ 项目结构

```
rag-bench/
├── src/rag_benchmark/
│   ├── datasets/          # 数据集管理
│   ├── prepare/           # RAG系统准备
│   ├── evaluate/          # 评测模块
│   ├── analysis/          # 结果分析
│   └── api/              # Web API服务
├── frontend/             # React前端
├── docs/                 # 文档
├── examples/             # 示例代码
└── tests/               # 测试
```

## 🌐 Web界面

### 功能页面

1. **数据集**: 浏览数据集，查看统计信息和样本
2. **模型仓库**: 注册和管理模型配置
3. **RAG系统**: 创建和管理RAG实例
4. **评测**: 配置和启动评测任务
5. **结果**: 查看评测结果和性能对比

### 界面预览

```
┌─────────────────────────────────────────┐
│ RAG Benchmark                            │
├─────────────────────────────────────────┤
│ 数据集 | 模型仓库 | RAG系统 | 评测 | 结果 │
├─────────────────────────────────────────┤
│                                          │
│  [功能区域]                              │
│                                          │
└─────────────────────────────────────────┘
```

## 📊 使用示例

### 示例1: 基础评测

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import DummyRAG, prepare_experiment_dataset
from rag_benchmark.evaluate import evaluate_e2e

# 加载数据集
dataset = GoldenDataset("xquad", subset="zh")

# 创建RAG系统
rag = DummyRAG()

# 准备实验数据集
exp_ds = prepare_experiment_dataset(dataset.sample(5), rag)

# 运行评测
result = evaluate_e2e(exp_ds)
print(result.to_pandas())
```

### 示例2: 自定义RAG

```python
from rag_benchmark.prepare import RAGInterface, RetrievalResult, GenerationResult

class MyRAG(RAGInterface):
    def retrieve(self, query, top_k=None):
        # 实现检索逻辑
        contexts = self.my_retriever.search(query, top_k)
        return RetrievalResult(contexts=contexts)
    
    def generate(self, query, contexts):
        # 实现生成逻辑
        answer = self.my_generator.generate(query, contexts)
        return GenerationResult(response=answer)

# 使用自定义RAG
my_rag = MyRAG()
exp_ds = prepare_experiment_dataset(dataset, my_rag)
result = evaluate_e2e(exp_ds)
```

### 示例3: 批量处理

```python
# 批量检索（性能提升2-5倍）
queries = ["query1", "query2", "query3"]
retrieval_results = rag.batch_retrieve(queries, top_k=3)

# 批量生成
contexts_list = [r.contexts for r in retrieval_results]
generation_results = rag.batch_generate(queries, contexts_list)
```

更多示例请查看 `examples/` 目录。

## 🔧 技术栈

### 后端
- **Python 3.11+**
- **FastAPI**: Web框架
- **Pydantic**: 数据验证
- **LangChain**: LLM集成
- **RAGAS**: 评测框架
- **FAISS**: 向量检索

### 前端
- **React 18**: UI框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **Tailwind CSS**: 样式框架
- **Recharts**: 图表库

### 依赖管理
- **uv**: Python包管理
- **npm**: 前端包管理

## 🐳 Docker部署

```bash
# 配置环境变量
cp .env.example .env

# 启动所有服务
docker-compose up

# 访问
# 前端: http://localhost:3000
# API: http://localhost:8000
```

## 📝 开发

### 运行测试

```bash
# Python测试
pytest tests/

# API测试
uv run python test_setup.py
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

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

MIT License

## 🙏 致谢

本项目参考了以下优秀框架：
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG评估框架
- [ARES](https://github.com/stanford-futuredata/ARES) - 自动RAG评估系统
- [BEIR](https://github.com/beir-cellar/beir) - 信息检索基准测试

## 📞 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/rag-bench/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/rag-bench/discussions)

## 📈 更新日志

### v0.3.0 (2025-11-20)

**新功能**
- ✅ 模型仓库：统一管理模型配置
- ✅ Web界面：React前端 + FastAPI后端
- ✅ 异步任务：支持断点续传
- ✅ 实时进度：显示评测阶段和进度

**改进**
- ✅ 模型配置界面化，不再依赖环境变量
- ✅ 任务状态持久化到磁盘
- ✅ 批量处理优化性能

### v0.2.0 (2025-11-19)

**Evaluate模块**
- ✅ 集成RAGAS评测框架
- ✅ 实现传统IR指标
- ✅ 支持自定义模型

**Analysis模块**
- ✅ 多模型结果对比
- ✅ 可视化图表生成

**Baseline RAG**
- ✅ FAISS + LLM实现
- ✅ 批量处理优化

### v0.1.0 (2025-11-18)

- ✅ 实现datasets模块
- ✅ 实现prepare模块
- ✅ 支持HotpotQA、NQ、XQuAD数据集
