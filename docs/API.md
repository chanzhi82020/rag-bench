# RAG Benchmark API 文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs`
- **版本**: v0.3.0

## 模型仓库 API

### POST /models/register
注册模型到仓库

**请求体**:
```json
{
  "model_id": "gpt-3.5-turbo-default",
  "model_name": "gpt-3.5-turbo",
  "model_type": "llm",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "description": "默认的GPT-3.5模型"
}
```

**响应**:
```json
{
  "message": "模型 'gpt-3.5-turbo-default' 注册成功",
  "model_id": "gpt-3.5-turbo-default"
}
```

### GET /models/list
列出所有已注册的模型

**响应**:
```json
{
  "llm_models": [
    {
      "model_id": "gpt-3.5-turbo-default",
      "model_name": "gpt-3.5-turbo",
      "model_type": "llm",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-...",
      "description": "默认的GPT-3.5模型"
    }
  ],
  "embedding_models": [...],
  "total": 5
}
```

### GET /models/{model_id}
获取模型信息

### PUT /models/{model_id}
更新模型配置

### DELETE /models/{model_id}
删除模型

## 数据集 API

### GET /datasets
列出所有可用数据集

**响应**:
```json
["xquad", "hotpotqa", "nq"]
```

### POST /datasets/stats
获取数据集统计信息

**请求体**:
```json
{
  "name": "xquad",
  "subset": "zh"
}
```

**响应**:
```json
{
  "dataset_name": "xquad",
  "subset": "zh",
  "record_count": 1000,
  "avg_input_length": 50.5,
  "avg_reference_length": 30.2,
  "avg_contexts_per_record": 3.0,
  "corpus_count": 5000
}
```

### POST /datasets/sample
获取数据集样本

**参数**: `n` (默认5)

**请求体**:
```json
{
  "name": "xquad",
  "subset": "zh"
}
```

## RAG 系统 API

### POST /rag/create
创建RAG实例

**请求体**:
```json
{
  "name": "my_rag",
  "model_config": {
    "llm_model_id": "gpt-3.5-turbo-default",
    "embedding_model_id": "text-embedding-3-small-default"
  },
  "rag_config": {
    "top_k": 5,
    "temperature": 0.7,
    "max_length": 512
  }
}
```

**响应**:
```json
{
  "message": "RAG实例 'my_rag' 创建成功",
  "name": "my_rag",
  "model_config": {...},
  "rag_config": {...}
}
```

### GET /rag/list
列出所有RAG实例

**响应**:
```json
{
  "rags": [
    {
      "name": "my_rag",
      "model_config": {...},
      "rag_config": {...}
    }
  ],
  "count": 1
}
```

### POST /rag/index
为RAG实例索引文档

**请求体**:
```json
{
  "rag_name": "my_rag",
  "documents": ["doc1", "doc2", "doc3"]
}
```

### POST /rag/query
查询RAG系统

**请求体**:
```json
{
  "rag_name": "my_rag",
  "query": "What is Python?",
  "top_k": 3
}
```

**响应**:
```json
{
  "query": "What is Python?",
  "answer": "Python is a programming language...",
  "contexts": ["context1", "context2", "context3"],
  "scores": [0.95, 0.87, 0.82]
}
```

## 评测 API

### POST /evaluate/start
启动评测任务（异步）

**请求体**:
```json
{
  "dataset_name": "xquad",
  "subset": "zh",
  "rag_name": "my_rag",
  "eval_type": "e2e",
  "sample_size": 10,
  "model_config": {
    "llm_model_id": "gpt-3.5-turbo-default",
    "embedding_model_id": "text-embedding-3-small-default"
  }
}
```

**评测类型**:
- `e2e`: 端到端评测
- `retrieval`: 检索阶段评测
- `generation`: 生成阶段评测

**响应**:
```json
{
  "task_id": "uuid",
  "message": "评测任务已启动",
  "status_url": "/evaluate/status/uuid"
}
```

### GET /evaluate/status/{task_id}
获取评测任务状态

**响应**:
```json
{
  "task_id": "uuid",
  "status": "running",
  "progress": 0.5,
  "current_stage": "准备实验数据集",
  "result": null,
  "error": null,
  "created_at": "2025-11-20T...",
  "updated_at": "2025-11-20T..."
}
```

**状态值**:
- `pending`: 等待中
- `running`: 运行中
- `completed`: 已完成
- `failed`: 失败

**当status为completed时**:
```json
{
  "task_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "current_stage": "完成",
  "result": {
    "metrics": {
      "faithfulness": 0.85,
      "answer_correctness": 0.78,
      "context_recall": 0.92
    },
    "sample_count": 10,
    "eval_type": "e2e",
    "model_config": {...}
  },
  "error": null,
  "created_at": "2025-11-20T...",
  "updated_at": "2025-11-20T..."
}
```

### GET /evaluate/tasks
列出所有评测任务

**响应**:
```json
{
  "tasks": [
    {
      "task_id": "uuid",
      "status": "completed",
      "progress": 1.0,
      "result": {...}
    }
  ],
  "count": 10
}
```

## 健康检查

### GET /health
健康检查

**响应**:
```json
{
  "status": "healthy"
}
```

### GET /
API根路径

**响应**:
```json
{
  "message": "RAG Benchmark API",
  "version": "0.3.0",
  "docs": "/docs"
}
```

## 错误响应

所有API在出错时返回标准错误格式：

```json
{
  "detail": "错误信息描述"
}
```

**常见HTTP状态码**:
- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

## 使用示例

### Python客户端

```python
import requests

BASE_URL = "http://localhost:8000"

# 注册模型
response = requests.post(f"{BASE_URL}/models/register", json={
    "model_id": "gpt-3.5-turbo-default",
    "model_name": "gpt-3.5-turbo",
    "model_type": "llm",
    "api_key": "sk-..."
})

# 创建RAG
response = requests.post(f"{BASE_URL}/rag/create", json={
    "name": "my_rag",
    "model_config": {
        "llm_model_id": "gpt-3.5-turbo-default",
        "embedding_model_id": "text-embedding-3-small-default"
    }
})

# 启动评测
response = requests.post(f"{BASE_URL}/evaluate/start", json={
    "dataset_name": "xquad",
    "rag_name": "my_rag",
    "eval_type": "e2e",
    "sample_size": 10,
    "model_config": {
        "llm_model_id": "gpt-3.5-turbo-default",
        "embedding_model_id": "text-embedding-3-small-default"
    }
})

task_id = response.json()["task_id"]

# 查询状态
response = requests.get(f"{BASE_URL}/evaluate/status/{task_id}")
print(response.json())
```

### cURL示例

```bash
# 注册模型
curl -X POST http://localhost:8000/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-3.5-turbo-default",
    "model_name": "gpt-3.5-turbo",
    "model_type": "llm",
    "api_key": "sk-..."
  }'

# 列出数据集
curl http://localhost:8000/datasets

# 创建RAG
curl -X POST http://localhost:8000/rag/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_rag",
    "model_config": {
      "llm_model_id": "gpt-3.5-turbo-default",
      "embedding_model_id": "text-embedding-3-small-default"
    }
  }'

# 启动评测
curl -X POST http://localhost:8000/evaluate/start \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "xquad",
    "rag_name": "my_rag",
    "eval_type": "e2e",
    "sample_size": 10,
    "model_config": {
      "llm_model_id": "gpt-3.5-turbo-default",
      "embedding_model_id": "text-embedding-3-small-default"
    }
  }'
```
