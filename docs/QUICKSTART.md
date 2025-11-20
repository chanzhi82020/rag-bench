# 快速开始

## 环境要求

- Python 3.11+
- Node.js 18+
- Conda (推荐)

## 安装步骤

### 1. 创建环境

```bash
# 创建conda环境
conda create -n rag-bench python=3.11 -y
conda activate rag-bench

# 安装uv
pip install uv
```

### 2. 安装项目

```bash
# 同步依赖
uv sync

# 安装项目和API依赖
uv pip install -e ".[api]"
```

### 3. 验证安装

```bash
# 测试导入
uv run python -c "from rag_benchmark.datasets import GoldenDataset; print('✅ 安装成功')"
```

## 启动Web服务

### 1. 启动API服务

```bash
# 终端1
./start_api.sh
```

访问: http://localhost:8000/docs

### 2. 启动前端

```bash
# 终端2
./start_frontend.sh
```

访问: http://localhost:3000

## 使用流程

### 1. 注册模型

1. 访问"模型仓库"标签页
2. 点击"注册模型"
3. 填写模型信息：
   - 模型ID: `gpt-3.5-turbo-default`
   - 模型类型: `llm`
   - 模型名称: `gpt-3.5-turbo`
   - API Key: `sk-...`
4. 点击"注册"

同样注册一个Embedding模型：
   - 模型ID: `text-embedding-3-small-default`
   - 模型类型: `embedding`
   - 模型名称: `text-embedding-3-small`
   - API Key: `sk-...`

### 2. 浏览数据集

1. 访问"数据集"标签页
2. 点击数据集查看统计信息
3. 查看数据样本

### 3. 创建RAG系统

1. 访问"RAG系统"标签页
2. 点击"创建RAG"
3. 填写配置：
   - 名称: `my_first_rag`
   - LLM模型: 选择已注册的LLM模型
   - Embedding模型: 选择已注册的Embedding模型
   - Top K: `5`
   - Temperature: `0.7`
4. 点击"创建"

### 4. 运行评测

1. 访问"评测"标签页
2. 选择：
   - 数据集: `xquad`
   - RAG系统: `my_first_rag`
   - 评测类型: `端到端`
   - 样本数量: `5`
   - LLM模型: 选择评测用的LLM模型
   - Embedding模型: 选择评测用的Embedding模型
3. 点击"启动评测"
4. 等待评测完成（查看实时进度）

### 5. 查看结果

1. 访问"结果"标签页
2. 查看评测指标
3. 对比不同RAG系统的性能

## Python API使用

### 基础示例

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import BaselineRAG, RAGConfig
from rag_benchmark.evaluate import evaluate_e2e
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 1. 加载数据集
dataset = GoldenDataset("xquad", subset="zh")
print(f"数据集大小: {dataset.count()}")

# 2. 创建RAG系统
rag = BaselineRAG(
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    config=RAGConfig(top_k=5)
)

# 3. 索引文档（示例）
documents = ["Python是一种编程语言", "RAG是检索增强生成"]
rag.index_documents(documents)

# 4. 查询测试
result = rag.query("什么是Python?")
print(f"答案: {result['answer']}")

# 5. 准备实验数据集
from rag_benchmark.prepare import prepare_experiment_dataset
exp_ds = prepare_experiment_dataset(dataset.sample(5), rag)

# 6. 运行评测
result = evaluate_e2e(exp_ds, experiment_name="test")
df = result.to_pandas()
print(df[['faithfulness', 'answer_correctness']].mean())
```

### 自定义RAG集成

```python
from rag_benchmark.prepare import RAGInterface, RetrievalResult, GenerationResult

class MyRAG(RAGInterface):
    def retrieve(self, query, top_k=None):
        # 实现检索逻辑
        contexts = ["context1", "context2"]
        return RetrievalResult(contexts=contexts)
    
    def generate(self, query, contexts):
        # 实现生成逻辑
        answer = "generated answer"
        return GenerationResult(response=answer)

# 使用自定义RAG
my_rag = MyRAG()
exp_ds = prepare_experiment_dataset(dataset, my_rag)
```

## Docker部署

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑.env文件

# 2. 启动所有服务
docker-compose up

# 3. 访问
# 前端: http://localhost:3000
# API: http://localhost:8000
```

## 常见问题

### Q: 模块导入失败

```bash
# 确保在正确的环境中
conda activate rag-bench

# 重新安装
uv pip install -e ".[api]"
```

### Q: API启动失败

```bash
# 检查端口占用
lsof -i :8000

# 检查依赖
uv run python -c "import fastapi; print('OK')"
```

### Q: 前端启动失败

```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```

### Q: 评测失败

- 检查模型仓库中是否已注册模型
- 检查API Key是否正确
- 查看API日志获取详细错误信息

## 下一步

- 查看 [API文档](API.md) 了解详细接口
- 查看 [架构文档](ARCHITECTURE.md) 了解系统设计
- 查看 `examples/` 目录获取更多示例
