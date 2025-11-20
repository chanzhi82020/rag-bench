# RAG Benchmark 架构设计

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 (React)                          │
│  数据集浏览 | 模型仓库 | RAG管理 | 评测控制 | 结果分析      │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST API
┌───────────────────────────▼─────────────────────────────────┐
│                      API服务 (FastAPI)                        │
│  数据集API | 模型仓库API | RAG API | 评测API                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   核心框架 (rag_benchmark)                    │
│  datasets | prepare | evaluate | analysis | api             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      外部服务                                 │
│  OpenAI API | RAGAS | FAISS | LangChain                     │
└───────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. datasets - 数据集管理
- **职责**: 加载、验证、管理Golden Dataset
- **核心类**: `GoldenDataset`, `GoldenRecord`, `CorpusRecord`
- **功能**: 数据集加载、统计、采样、验证

### 2. prepare - RAG系统准备
- **职责**: RAG系统接口定义和实现
- **核心类**: `RAGInterface`, `BaselineRAG`, `RAGConfig`
- **功能**: RAG系统集成、文档索引、查询处理

### 3. evaluate - 评测模块
- **职责**: RAG系统评测
- **核心函数**: `evaluate_e2e()`, `evaluate_retrieval()`, `evaluate_generation()`
- **指标**: Recall@K, Precision@K, NDCG, Faithfulness, Answer Correctness

### 4. analysis - 结果分析
- **职责**: 评测结果分析和可视化
- **功能**: 多模型对比、统计分析、图表生成

### 5. api - Web API服务
- **职责**: 提供RESTful API接口
- **技术**: FastAPI + Pydantic
- **功能**: 数据集管理、模型仓库、RAG管理、评测任务

## 数据流

### 评测流程

```
1. 加载数据集
   GoldenDataset → 数据记录

2. 准备实验数据
   数据记录 + RAG系统 → 实验数据集
   (包含检索结果和生成答案)

3. 运行评测
   实验数据集 + 评测指标 → 评测结果

4. 分析结果
   评测结果 → 统计分析 + 可视化
```

## 模型仓库设计

### 核心概念
- **模型注册**: 统一管理模型配置（base_url, api_key, model_name）
- **模型引用**: 通过model_id引用已注册的模型
- **持久化**: 配置保存在 `data/models.json`

### 数据模型

```python
ModelInfo:
  - model_id: str          # 唯一标识
  - model_name: str        # 实际模型名
  - model_type: str        # llm | embedding
  - base_url: Optional[str]
  - api_key: str
  - description: Optional[str]
```

## 任务管理设计

### 异步任务
- 使用FastAPI BackgroundTasks
- 任务状态持久化到 `data/tasks/`
- 支持断点续传

### 任务阶段
1. 初始化 (0%)
2. 加载数据集 (10-20%)
3. 准备实验数据集 (30-50%)
4. 运行评测 (60-80%)
5. 处理结果 (80-100%)

## 技术栈

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
- **Tailwind CSS**: 样式
- **Recharts**: 图表

### 依赖管理
- **uv**: Python包管理
- **npm**: 前端包管理

## 扩展点

### 1. 添加新数据集
```python
class MyDatasetLoader(BaseLoader):
    def load_golden_records(self):
        # 实现加载逻辑
        pass

DATASET_REGISTRY.register("my_dataset", MyDatasetLoader)
```

### 2. 集成自定义RAG
```python
class MyRAG(RAGInterface):
    def retrieve(self, query, top_k):
        # 实现检索
        pass
    
    def generate(self, query, contexts):
        # 实现生成
        pass
```

### 3. 添加新评测指标
```python
from ragas.metrics import Metric

class MyMetric(Metric):
    def score(self, row):
        # 实现评分逻辑
        pass
```

## 部署架构

### 开发环境
```
localhost:3000 (Frontend) → localhost:8000 (API)
```

### 生产环境
```
Nginx (反向代理)
  ├── Frontend (静态文件)
  └── API (Gunicorn + Uvicorn)
```

### Docker部署
```bash
docker-compose up
  ├── api (Python容器)
  └── frontend (Node容器)
```

## 性能优化

### 1. 批量处理
- `batch_retrieve()` - 批量检索
- `batch_generate()` - 批量生成
- 性能提升2-5倍

### 2. 任务持久化
- 状态保存到磁盘
- 支持断点续传
- 服务重启不丢失

### 3. 模型缓存
- RAG实例缓存
- 模型配置缓存
- 减少重复初始化

## 安全考虑

### API安全
- CORS配置
- 输入验证（Pydantic）
- API密钥管理（环境变量）

### 数据安全
- API Key加密存储
- 敏感信息不记录日志
- 数据验证和清洗
