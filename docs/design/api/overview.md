# API Module Design

## Overview

The API module provides a FastAPI-based web service that exposes RAG Benchmark's core functionality through RESTful endpoints, enabling remote evaluation, model management, and task orchestration.

## Architecture

```
api/
├── main.py        # FastAPI application and endpoints
├── test_api.py    # Basic API tests
└── README.md      # API documentation
```

## Core Components

### 1. FastAPI Application

```python
app = FastAPI(
    title="RAG Benchmark API",
    description="API服务用于RAG系统评测",
    version="0.1.0"
)
```

**CORS Configuration**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Design Rationale**:
- FastAPI for modern async support
- CORS enabled for frontend integration
- OpenAPI documentation auto-generation

### 2. State Management

**Global State**:
```python
tasks_status = {}        # Task tracking
rag_instances = {}       # RAG system instances
model_registry = {}      # Model repository
```

**Persistence**:
```python
TASKS_DIR = Path("data/tasks")           # Task status files
MODELS_FILE = Path("data/models.json")   # Model registry file
```

**Design Rationale**:
- In-memory state for fast access
- Disk persistence for durability
- Separate storage for different concerns

### 3. Data Models (Pydantic)

**ModelInfo**: Model registration
```python
class ModelInfo(BaseModel):
    model_id: str                    # Unique identifier
    model_name: str                  # Model name (e.g., gpt-3.5-turbo)
    model_type: str                  # "llm" or "embedding"
    base_url: Optional[str]          # Custom API endpoint
    api_key: str                     # API key
    description: Optional[str]       # Description
```

**ModelConfig**: Model reference for RAG/evaluation
```python
class ModelConfig(BaseModel):
    llm_model_id: str                # Reference to LLM in registry
    embedding_model_id: str          # Reference to embedding in registry
```

**RAGConfigModel**: RAG configuration
```python
class RAGConfigModel(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_length: int = Field(default=512, ge=1, le=2048)
```

**EvaluateRequest**: Evaluation request
```python
class EvaluateRequest(BaseModel):
    dataset_name: str
    subset: Optional[str]
    rag_name: str
    eval_type: str = "e2e"           # e2e, retrieval, generation
    sample_size: Optional[int]
    model_info: ModelConfig          # Evaluation models
```

**TaskStatus**: Task status tracking
```python
class TaskStatus(BaseModel):
    task_id: str
    status: str                      # pending, running, completed, failed
    progress: float                  # 0.0 - 1.0
    current_stage: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: str
    updated_at: str
```

**Design Rationale**:
- Pydantic for automatic validation
- Type safety and documentation
- Clear separation of concerns

## API Endpoints

### Model Registry APIs

**POST /models/register**: Register a model
```python
@app.post("/models/register")
async def register_model(model: ModelInfo):
    """
    Register LLM or embedding model to repository
    
    Returns: {"message": "...", "model_id": "..."}
    """
```

**GET /models/list**: List all models
```python
@app.get("/models/list")
async def list_models():
    """
    Returns: {
        "llm_models": [...],
        "embedding_models": [...],
        "total": n
    }
    """
```

**GET /models/{model_id}**: Get model info
```python
@app.get("/models/{model_id:path}")
async def get_model(model_id: str):
    """Returns model information"""
```

**PUT /models/{model_id}**: Update model
```python
@app.put("/models/{model_id:path}")
async def update_model(model_id: str, model: ModelInfo):
    """Update model information"""
```

**DELETE /models/{model_id}**: Delete model
```python
@app.delete("/models/{model_id:path}")
async def delete_model(model_id: str):
    """Delete model from registry"""
```

**Design Rationale**:
- Centralized model management
- Reusable model configurations
- Secure API key storage

### Dataset APIs

**GET /datasets**: List datasets
```python
@app.get("/datasets", response_model=List[str])
async def list_datasets():
    """Returns list of available dataset names"""
```

**POST /datasets/stats**: Get dataset statistics
```python
@app.post("/datasets/stats", response_model=DatasetStats)
async def get_dataset_stats(dataset_info: DatasetInfo):
    """
    Returns statistics:
    - record_count
    - avg_input_length
    - avg_reference_length
    - avg_contexts_per_record
    - corpus_count
    """
```

**POST /datasets/sample**: Get dataset samples
```python
@app.post("/datasets/sample")
async def sample_dataset(dataset_info: DatasetInfo, n: int = 5):
    """Returns n sample records from dataset"""
```

**Design Rationale**:
- Read-only dataset access
- Statistics for dataset selection
- Sampling for preview

### RAG APIs

**POST /rag/create**: Create RAG instance
```python
@app.post("/rag/create")
async def create_rag(request: CreateRAGRequest):
    """
    Create RAG instance with specified models
    
    Request:
    - name: RAG instance name
    - model_info: ModelConfig (references to registry)
    - rag_config: RAGConfigModel
    
    Returns: {"message": "...", "name": "...", ...}
    """
```

**GET /rag/list**: List RAG instances
```python
@app.get("/rag/list")
async def list_rags():
    """
    Returns: {
        "rags": [{"name": "...", "model_info": {...}, ...}],
        "count": n
    }
    """
```

**POST /rag/index**: Index documents
```python
@app.post("/rag/index")
async def index_documents(request: IndexDocumentsRequest):
    """
    Index documents into RAG system
    
    Request:
    - rag_name: RAG instance name
    - documents: List of document texts
    """
```

**POST /rag/query**: Query RAG system
```python
@app.post("/rag/query")
async def query_rag(request: QueryRequest):
    """
    Query RAG system
    
    Returns: {
        "query": "...",
        "answer": "...",
        "contexts": [...],
        "scores": [...]
    }
    """
```

**Design Rationale**:
- Stateful RAG instances
- Separate indexing and querying
- Support multiple RAG systems

### Evaluation APIs

**POST /evaluate/start**: Start evaluation task
```python
@app.post("/evaluate/start")
async def start_evaluation(
    request: EvaluateRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start async evaluation task
    
    Returns: {
        "task_id": "...",
        "message": "...",
        "status_url": "/evaluate/status/{task_id}"
    }
    """
```

**GET /evaluate/status/{task_id}**: Get task status
```python
@app.get("/evaluate/status/{task_id}", response_model=TaskStatus)
async def get_evaluation_status(task_id: str):
    """
    Get evaluation task status
    
    Returns: TaskStatus with progress and results
    """
```

**GET /evaluate/tasks**: List all tasks
```python
@app.get("/evaluate/tasks")
async def list_evaluation_tasks():
    """
    Returns: {
        "tasks": [...],
        "count": n
    }
    """
```

**Design Rationale**:
- Async evaluation for long-running tasks
- Progress tracking
- Task persistence for recovery

## Task Execution Flow

### Evaluation Task Flow
```
POST /evaluate/start
    ↓
Create task_id
    ↓
Save initial task status
    ↓
Add background task
    ↓
Return task_id immediately
    ↓
Background: run_evaluation_task()
    ↓
Stage 1: Load dataset (progress: 0.1-0.2)
    ↓
Stage 2: Prepare experiment dataset (progress: 0.3-0.5)
    ↓
Stage 3: Run evaluation (progress: 0.6-0.8)
    ↓
Stage 4: Process results (progress: 0.8-1.0)
    ↓
Update task status: completed
    ↓
Save final results
```

### Task Status Updates
```python
def update_task_status(task_id: str, **kwargs):
    """Update task status and persist to disk"""
    tasks_status[task_id].update(kwargs)
    tasks_status[task_id]["updated_at"] = datetime.now().isoformat()
    save_task_status(task_id)
```

**Design Rationale**:
- Non-blocking API responses
- Fine-grained progress tracking
- Automatic persistence

## Persistence Strategy

### Task Persistence
```python
def save_task_status(task_id: str):
    """Save task status to data/tasks/{task_id}.json"""
    task_file = TASKS_DIR / f"{task_id}.json"
    with open(task_file, 'w') as f:
        json.dump(tasks_status[task_id], f, indent=2)

def load_task_status(task_id: str) -> Optional[Dict]:
    """Load task status from disk"""
    task_file = TASKS_DIR / f"{task_id}.json"
    if task_file.exists():
        with open(task_file, 'r') as f:
            return json.load(f)
    return None
```

### Model Registry Persistence
```python
def save_model_registry():
    """Save model registry to data/models.json"""
    with open(MODELS_FILE, 'w') as f:
        json.dump(model_registry, f, indent=2)

def load_model_registry():
    """Load model registry on startup"""
    global model_registry
    if MODELS_FILE.exists():
        with open(MODELS_FILE, 'r') as f:
            model_registry = json.load(f)
```

**Design Rationale**:
- Survive server restarts
- Enable task recovery
- Simple JSON format for debugging

## Model Management

### Model Registry Pattern

**Registration**:
```python
# Register models once
POST /models/register
{
    "model_id": "gpt-3.5-turbo-default",
    "model_name": "gpt-3.5-turbo",
    "model_type": "llm",
    "api_key": "sk-...",
    "description": "Default GPT-3.5 Turbo"
}
```

**Usage**:
```python
# Reference by ID in RAG creation
POST /rag/create
{
    "name": "my_rag",
    "model_info": {
        "llm_model_id": "gpt-3.5-turbo-default",
        "embedding_model_id": "text-embedding-3-small-default"
    }
}

# Reference by ID in evaluation
POST /evaluate/start
{
    "rag_name": "my_rag",
    "model_info": {
        "llm_model_id": "gpt-4-eval",
        "embedding_model_id": "text-embedding-3-large-eval"
    }
}
```

**Design Rationale**:
- Centralized credential management
- Reusable configurations
- Separate RAG models from evaluation models

### Model Client Creation
```python
def get_model_client(model_id: str):
    """Create model client from registry"""
    model_info = model_registry[model_id]
    
    kwargs = {
        "model": model_info["model_name"],
        "api_key": model_info["api_key"]
    }
    
    if model_info.get("base_url"):
        kwargs["base_url"] = model_info["base_url"]
    
    if model_info["model_type"] == "llm":
        return ChatOpenAI(**kwargs)
    elif model_info["model_type"] == "embedding":
        return OpenAIEmbeddings(**kwargs)
```

**Design Rationale**:
- Lazy client creation
- Support custom endpoints
- Type-based instantiation

## Error Handling

### API Error Responses
```python
# Not found
raise HTTPException(status_code=404, detail="Resource not found")

# Bad request
raise HTTPException(status_code=400, detail="Invalid input")

# Internal error
try:
    # operation
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Task Error Handling
```python
def run_evaluation_task(task_id: str, request: EvaluateRequest):
    try:
        # evaluation stages
        update_task_status(task_id, status="completed", result=...)
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        update_task_status(task_id, status="failed", error=str(e))
```

**Design Rationale**:
- Standard HTTP status codes
- Detailed error messages
- Task-level error isolation

## Design Principles

1. **RESTful Design**:
   - Resource-based URLs
   - Standard HTTP methods
   - Stateless requests (except RAG instances)

2. **Async Processing**:
   - Background tasks for long operations
   - Non-blocking API responses
   - Progress tracking

3. **Persistence**:
   - Task status survives restarts
   - Model registry persisted
   - Simple JSON format

4. **Separation of Concerns**:
   - Model registry separate from RAG instances
   - RAG models separate from evaluation models
   - Dataset access separate from evaluation

5. **Type Safety**:
   - Pydantic models for validation
   - Type hints throughout
   - Auto-generated OpenAPI docs

## Usage Patterns

### Pattern 1: Complete Evaluation Workflow
```python
# 1. Register models
POST /models/register
{
    "model_id": "my-llm",
    "model_name": "gpt-3.5-turbo",
    "model_type": "llm",
    "api_key": "sk-..."
}

# 2. Create RAG
POST /rag/create
{
    "name": "my_rag",
    "model_info": {
        "llm_model_id": "my-llm",
        "embedding_model_id": "my-embedding"
    }
}

# 3. Start evaluation
POST /evaluate/start
{
    "dataset_name": "hotpotqa",
    "rag_name": "my_rag",
    "eval_type": "e2e",
    "model_info": {...}
}

# 4. Check status
GET /evaluate/status/{task_id}

# 5. Get results when completed
GET /evaluate/status/{task_id}
```

### Pattern 2: Interactive RAG Testing
```python
# Create RAG
POST /rag/create {...}

# Index documents
POST /rag/index
{
    "rag_name": "my_rag",
    "documents": ["doc1", "doc2", ...]
}

# Query
POST /rag/query
{
    "rag_name": "my_rag",
    "query": "What is...?",
    "top_k": 3
}
```

### Pattern 3: Dataset Exploration
```python
# List datasets
GET /datasets

# Get statistics
POST /datasets/stats
{
    "name": "hotpotqa",
    "subset": "distractor"
}

# Get samples
POST /datasets/sample?n=5
{
    "name": "hotpotqa",
    "subset": "distractor"
}
```

## Security Considerations

### Current Implementation
- API keys stored in model registry
- No authentication on API endpoints
- CORS allows all origins

### Production Recommendations
1. **Authentication**: Add API key or OAuth
2. **Authorization**: Role-based access control
3. **Encryption**: HTTPS only
4. **Rate Limiting**: Prevent abuse
5. **Input Validation**: Strict validation
6. **Secret Management**: Use environment variables or secret manager

## Performance Considerations

### Scalability
- In-memory state limits horizontal scaling
- Consider Redis for shared state
- Background tasks run in same process

### Optimization
- Batch evaluation for efficiency
- Cache model clients
- Lazy loading of datasets

### Resource Management
- RAG instances consume memory
- FAISS indices can be large
- Consider cleanup of old tasks

## Future Enhancements

1. **Authentication**: JWT-based auth
2. **WebSocket**: Real-time progress updates
3. **Distributed Tasks**: Celery for task queue
4. **Caching**: Redis for results caching
5. **Monitoring**: Prometheus metrics
6. **Logging**: Structured logging
7. **Rate Limiting**: Per-user rate limits
8. **Batch APIs**: Batch evaluation requests
9. **Result Storage**: Database for results
10. **API Versioning**: /v1/, /v2/ endpoints
