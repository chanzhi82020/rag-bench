# RAG实例持久化

## 问题

之前RAG实例只存储在内存中（`rag_instances`字典），后端重启后所有RAG实例都会丢失。

## 解决方案

添加RAG实例的持久化存储，将配置保存到JSON文件。

## 实现

### 1. 存储结构

```
data/
├── tasks/          # 评测任务状态
│   └── {task_id}.json
├── models.json     # 模型仓库
└── rags/           # RAG实例配置（新增）
    ├── baseline.json
    ├── my_rag.json
    └── ...
```

### 2. RAG配置文件格式

```json
{
  "name": "baseline",
  "model_info": {
    "llm_model_id": "deepseek-ai/deepseek-v3.1",
    "embedding_model_id": "embedding-3"
  },
  "rag_config": {
    "top_k": 5,
    "temperature": 0.7,
    "max_length": 512,
    "similarity_threshold": 0.0,
    "retrieval_mode": "dense",
    "top_p": 0.9,
    "batch_size": 1,
    "timeout": 30,
    "extra_params": {}
  },
  "created_at": "2025-11-21T16:30:00.000000"
}
```

### 3. 核心函数

#### 保存RAG实例

```python
def save_rag_instance(rag_name: str):
    """保存RAG实例配置到磁盘"""
    rag_file = RAGS_DIR / f"{rag_name}.json"
    rag_data = {
        "name": rag_name,
        "model_info": rag_instances[rag_name]["model_info"],
        "rag_config": rag_instances[rag_name]["rag_config"],
        "created_at": rag_instances[rag_name].get("created_at", datetime.now().isoformat()),
    }
    with open(rag_file, 'w') as f:
        json.dump(rag_data, f, indent=2)
```

#### 加载单个RAG实例

```python
def load_rag_instance(rag_name: str) -> Optional[Dict]:
    """从磁盘加载RAG实例配置"""
    rag_file = RAGS_DIR / f"{rag_name}.json"
    if rag_file.exists():
        with open(rag_file, 'r') as f:
            return json.load(f)
    return None
```

#### 加载所有RAG实例

```python
def load_all_rag_instances():
    """从磁盘加载所有RAG实例"""
    global rag_instances
    for rag_file in RAGS_DIR.glob("*.json"):
        try:
            with open(rag_file, 'r') as f:
                rag_data = json.load(f)
            
            rag_name = rag_data["name"]
            model_info = rag_data["model_info"]
            rag_config_dict = rag_data["rag_config"]
            
            # 重建RAG实例
            rag_config = RAGConfig(
                top_k=rag_config_dict.get("top_k", 5),
                temperature=rag_config_dict.get("temperature", 0.7),
                max_length=rag_config_dict.get("max_length", 512)
            )
            
            llm = get_model_client(model_info["llm_model_id"])
            embedding = get_model_client(model_info["embedding_model_id"])
            
            rag = BaselineRAG(
                embedding_model=embedding,
                llm=llm,
                config=rag_config
            )
            
            rag_instances[rag_name] = {
                "rag": rag,
                "model_info": model_info,
                "rag_config": rag_config_dict,
                "created_at": rag_data.get("created_at"),
            }
            
            logger.info(f"RAG实例 '{rag_name}' 已从磁盘加载")
        except Exception as e:
            logger.error(f"加载RAG实例失败: {e}")
```

#### 删除RAG实例文件

```python
def delete_rag_instance_file(rag_name: str):
    """删除RAG实例配置文件"""
    rag_file = RAGS_DIR / f"{rag_name}.json"
    if rag_file.exists():
        rag_file.unlink()
```

### 4. 集成到API

#### 启动时加载

```python
# 启动时加载模型仓库和RAG实例
load_model_registry()
load_all_rag_instances()
```

#### 创建时保存

```python
@app.post("/rag/create")
async def create_rag(request: CreateRAGRequest):
    # ... 创建RAG实例
    
    rag_instances[request.name] = {
        "rag": rag,
        "model_info": request.model_info.model_dump(),
        "rag_config": rag_config.to_dict(),
        "created_at": datetime.now().isoformat()
    }
    
    # 保存到磁盘
    save_rag_instance(request.name)
    
    return {...}
```

#### 删除时清理

```python
@app.delete("/rag/{rag_name}")
async def delete_rag(rag_name: str):
    # ... 删除内存中的实例
    
    del rag_instances[rag_name]
    
    # 删除持久化文件
    delete_rag_instance_file(rag_name)
    
    return {...}
```

## 工作流程

### 创建RAG实例

1. 用户在前端创建RAG实例
2. 后端创建RAG对象并存储到`rag_instances`
3. 调用`save_rag_instance()`保存配置到JSON文件
4. 返回成功响应

### 后端重启

1. 后端启动时调用`load_all_rag_instances()`
2. 遍历`data/rags/`目录下的所有JSON文件
3. 读取配置并重建RAG实例
4. 存储到`rag_instances`字典
5. RAG实例恢复完成

### 删除RAG实例

1. 用户在前端删除RAG实例
2. 后端从`rag_instances`删除
3. 调用`delete_rag_instance_file()`删除JSON文件
4. 返回成功响应

## 注意事项

### 1. 向量存储不持久化

**当前实现**: 只保存RAG配置，不保存向量存储（索引的文档）

**原因**:
- 向量存储可能很大（数GB）
- 序列化/反序列化复杂
- 不同向量存储实现不同

**影响**:
- 重启后RAG实例存在，但没有索引的文档
- 需要重新索引文档

**未来改进**:
- 使用持久化向量存储（如Chroma、Qdrant）
- 保存向量存储路径到配置
- 启动时重新加载向量存储

### 2. 模型客户端依赖

RAG实例依赖模型仓库中的模型配置：

```python
llm = get_model_client(model_info["llm_model_id"])
embedding = get_model_client(model_info["embedding_model_id"])
```

**要求**:
- 模型必须在模型仓库中注册
- 模型配置（API key等）必须有效

**错误处理**:
- 如果模型不存在，加载会失败
- 失败的RAG实例会被跳过
- 日志中会记录错误

### 3. 配置变更

如果RAG配置类（`RAGConfig`）增加新字段：

**兼容性**:
- 使用`.get()`方法提供默认值
- 旧配置文件仍然可以加载

```python
rag_config = RAGConfig(
    top_k=rag_config_dict.get("top_k", 5),  # 默认值
    temperature=rag_config_dict.get("temperature", 0.7),
    new_field=rag_config_dict.get("new_field", "default")  # 新字段
)
```

## 测试

### 1. 创建并重启

```bash
# 1. 启动后端
./start_api.sh

# 2. 创建RAG实例（通过前端或API）
curl -X POST http://localhost:8000/rag/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_rag",
    "model_info": {
      "llm_model_id": "deepseek-ai/deepseek-v3.1",
      "embedding_model_id": "embedding-3"
    }
  }'

# 3. 验证文件已创建
ls -la data/rags/test_rag.json

# 4. 重启后端
# Ctrl+C 停止
./start_api.sh

# 5. 验证RAG实例已恢复
curl http://localhost:8000/rag/list
```

### 2. 删除验证

```bash
# 1. 删除RAG实例
curl -X DELETE http://localhost:8000/rag/test_rag

# 2. 验证文件已删除
ls -la data/rags/test_rag.json  # 应该不存在
```

## 优势

1. **持久化**: RAG实例配置在重启后保留
2. **简单**: 使用JSON文件，易于查看和编辑
3. **可靠**: 创建/删除操作同步到磁盘
4. **可扩展**: 易于添加新的配置字段

## 限制

1. **向量存储**: 不持久化，重启后需要重新索引
2. **并发**: 没有文件锁，不适合多进程部署
3. **规模**: 适合小规模部署（<1000个RAG实例）

## 未来改进

1. **向量存储持久化**: 使用持久化向量数据库
2. **数据库存储**: 使用SQLite/PostgreSQL替代JSON文件
3. **备份恢复**: 支持导出/导入RAG配置
4. **版本控制**: 跟踪配置变更历史
5. **批量操作**: 支持批量导入/导出RAG实例

## 总结

通过添加持久化功能，RAG实例现在可以在后端重启后自动恢复，提升了系统的可用性和用户体验。

**修改的文件**:
- `src/rag_benchmark/api/main.py` - 添加持久化函数和集成

**新增目录**:
- `data/rags/` - 存储RAG实例配置

**效果**:
- ✅ 创建RAG后自动保存
- ✅ 重启后自动加载
- ✅ 删除时清理文件
- ⚠️ 向量存储需要重新索引
