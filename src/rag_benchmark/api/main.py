"""RAG Benchmark API Server

FastAPI服务，暴露RAG Benchmark的核心功能
"""
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uuid
import json
import math
from pathlib import Path

from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.cache import DiskCacheBackend
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, BaselineRAG, RAGConfig, RAGInterface
from rag_benchmark.evaluate import evaluate_e2e, evaluate_retrieval, evaluate_generation
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rag_benchmark.api.checkpoint_manager import CheckpointManager
from rag_benchmark.api.sample_selector import SampleSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Benchmark API",
    description="API服务用于RAG系统评测",
    version="0.1.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态管理
tasks_status = {}
rag_instances = {}
model_registry = {}  # 模型仓库

# 持久化目录
TASKS_DIR = Path("data/tasks")
TASKS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_FILE = Path("data/models.json")
MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)


cache = DiskCacheBackend()

def sanitize_float_values(obj: Any) -> Any:
    """Sanitize float values to be JSON compliant
    
    Converts inf, -inf, and nan to None to prevent JSON serialization errors.
    
    Args:
        obj: Object to sanitize (can be dict, list, float, or any other type)
        
    Returns:
        Sanitized object with invalid floats replaced by None
    """
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
RAGS_DIR = Path("data/rags")
RAGS_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR = Path("data/indices")
INDICES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize CheckpointManager
checkpoint_manager = CheckpointManager(TASKS_DIR)


# ============ Pydantic Models ============

class DatasetInfo(BaseModel):
    name: str
    subset: Optional[str] = None


class DatasetStats(BaseModel):
    dataset_name: str
    subset: Optional[str]
    record_count: int
    avg_input_length: float
    avg_reference_length: float
    avg_contexts_per_record: float
    corpus_count: int


class RAGConfigModel(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_length: int = Field(default=512, ge=1, le=2048)


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str  # 唯一标识
    model_name: str  # 模型名称，如 gpt-3.5-turbo
    model_type: str  # llm 或 embedding
    base_url: Optional[str] = None
    api_key: str
    description: Optional[str] = None


class ModelConfig(BaseModel):
    """模型配置（引用模型仓库）"""
    llm_model_id: str  # 引用模型仓库中的LLM模型ID
    embedding_model_id: str  # 引用模型仓库中的Embedding模型ID


class CreateRAGRequest(BaseModel):
    name: str
    rag_type: str = "baseline"  # "baseline" or "api"
    model_info: ModelConfig
    rag_config: Optional[RAGConfigModel] = None
    api_config: Optional[Dict[str, str]] = None  # For API-based RAG


class IndexDocumentsRequest(BaseModel):
    rag_name: str
    dataset_name: str  # 从数据集加载corpus
    subset: Optional[str] = None


class QueryRequest(BaseModel):
    rag_name: str
    query: str
    top_k: Optional[int] = None


class SampleSelection(BaseModel):
    """Sample selection strategy for evaluation
    
    Attributes:
        strategy: Selection strategy - "specific_ids", "random", or "all"
        sample_ids: List of specific sample IDs to evaluate (for "specific_ids" strategy)
        sample_size: Number of random samples to select (for "random" strategy)
        random_seed: Random seed for reproducible random sampling (optional)
    """
    strategy: str = Field(..., description="Selection strategy: 'specific_ids', 'random', or 'all'")
    sample_ids: Optional[List[str]] = Field(None, description="Specific sample IDs to evaluate")
    sample_size: Optional[int] = Field(None, ge=1, description="Number of random samples to select")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducible sampling")


class EvaluateRequest(BaseModel):
    """Evaluation request with sample selection support
    
    Attributes:
        dataset_name: Name of the dataset to evaluate
        subset: Optional dataset subset
        rag_name: Name of the RAG instance to evaluate
        eval_type: Type of evaluation - "e2e", "retrieval", or "generation"
        model_info: Model configuration for evaluation
        sample_selection: Sample selection strategy (optional)
    """
    dataset_name: str
    subset: Optional[str] = None
    rag_name: str
    eval_type: str = "e2e"  # e2e, retrieval, generation
    model_info: ModelConfig  # 评测时的模型配置
    sample_selection: Optional[SampleSelection] = Field(
        None,
        description="Sample selection strategy. If not provided, evaluates all samples or uses sample_size for backward compatibility."
    )


class IndexStatus(BaseModel):
    """Index status information for a RAG instance"""
    has_index: bool
    document_count: Optional[int] = None
    embedding_dimension: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    total_size_bytes: Optional[int] = None


class SampleInfo(BaseModel):
    """Sample information for evaluation tasks
    
    Attributes:
        selection_strategy: Strategy used for sample selection ("specific_ids", "random", "all")
        total_samples: Total number of samples in the dataset
        selected_samples: Number of samples selected for evaluation
        sample_ids: List of sample IDs selected (optional, for specific_ids strategy)
        completed_samples: Number of samples that completed evaluation successfully
        failed_samples: Number of samples that failed during evaluation
    """
    selection_strategy: str = Field(..., description="Sample selection strategy used")
    total_samples: int = Field(..., ge=0, description="Total samples in dataset")
    selected_samples: int = Field(..., ge=0, description="Number of samples selected")
    sample_ids: Optional[List[str]] = Field(None, description="List of selected sample IDs")
    completed_samples: int = Field(default=0, ge=0, description="Samples completed successfully")
    failed_samples: int = Field(default=0, ge=0, description="Samples that failed")


class CheckpointInfo(BaseModel):
    """Checkpoint progress information for evaluation tasks
    
    Attributes:
        has_checkpoint: Whether checkpoint data exists for this task
        completed_stages: List of stages that have been completed
        current_stage: Current stage being executed (if running)
        last_checkpoint_at: Timestamp of last checkpoint save
    """
    has_checkpoint: bool = Field(..., description="Whether checkpoint data exists")
    completed_stages: List[str] = Field(default_factory=list, description="List of completed stages")
    current_stage: Optional[str] = Field(None, description="Current stage being executed")
    last_checkpoint_at: Optional[str] = Field(None, description="Timestamp of last checkpoint")


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float
    current_stage: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    sample_info: Optional[SampleInfo] = Field(None, description="Sample tracking information")
    checkpoint_info: Optional[CheckpointInfo] = Field(None, description="Checkpoint progress information")


# ============ Helper Functions ============

def save_task_status(task_id: str):
    """保存任务状态到磁盘（使用目录结构）"""
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    status_file = task_dir / "status.json"
    # Sanitize the status data before saving to prevent JSON serialization errors
    sanitized_status = sanitize_float_values(tasks_status[task_id])
    with open(status_file, 'w') as f:
        json.dump(sanitized_status, f, indent=2)


def load_task_status(task_id: str) -> Optional[Dict]:
    """从磁盘加载任务状态（从目录结构）"""
    task_dir = TASKS_DIR / task_id
    status_file = task_dir / "status.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    
    return None


def save_model_registry():
    """保存模型仓库到磁盘"""
    with open(MODELS_FILE, 'w') as f:
        json.dump(model_registry, f, indent=2)


def load_model_registry():
    """从磁盘加载模型仓库"""
    global model_registry
    if MODELS_FILE.exists():
        with open(MODELS_FILE, 'r') as f:
            model_registry = json.load(f)


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
    logger.info(f"RAG实例 '{rag_name}' 已保存到 {rag_file}")


def load_rag_registry():
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
            
            if hasattr(llm, 'temperature'):
                llm.temperature = rag_config.temperature
            
            rag = BaselineRAG(
                embedding_model=embedding,
                llm=llm,
                config=rag_config
            )
            
            # Load persisted index data if available
            index_path = INDICES_DIR / rag_name
            if index_path.exists():
                index_loaded = rag.load_from_disk(index_path)
                if index_loaded:
                    logger.info(f"RAG实例 '{rag_name}' 已加载持久化索引数据")
            
            rag_instances[rag_name] = {
                "rag": rag,
                "model_info": model_info,
                "rag_config": rag_config_dict,
                "created_at": rag_data.get("created_at"),
            }
            
            logger.info(f"RAG实例 '{rag_name}' 已从磁盘加载")
        except Exception as e:
            logger.error(f"加载RAG实例 {rag_file.stem} 失败: {e}")


def delete_rag_instance_file(rag_name: str):
    """删除RAG实例配置文件"""
    rag_file = RAGS_DIR / f"{rag_name}.json"
    if rag_file.exists():
        rag_file.unlink()
        logger.info(f"RAG实例文件 '{rag_name}' 已删除")


def get_model_client(model_id: str):
    """根据模型ID获取模型客户端"""
    if model_id not in model_registry:
        raise ValueError(f"模型 '{model_id}' 不存在")
    
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
    else:
        raise ValueError(f"不支持的模型类型: {model_info['model_type']}")


def update_task_status(task_id: str, **kwargs):
    """更新任务状态并保存"""
    tasks_status[task_id].update(kwargs)
    tasks_status[task_id]["updated_at"] = datetime.now().isoformat()
    save_task_status(task_id)


def get_rag_index_status_info(rag_name: str) -> IndexStatus:
    """Get index status information for a RAG instance
    
    Args:
        rag_name: Name of the RAG instance
        
    Returns:
        IndexStatus object with index information
    """
    from ..persistence import get_index_metadata
    
    # Check if RAG instance exists
    if rag_name not in rag_instances:
        return IndexStatus(has_index=False)
    
    rag_info = rag_instances[rag_name]
    rag = rag_info["rag"]
    
    # Get in-memory index stats
    stats = rag.get_index_stats()
    
    if not stats["has_index"]:
        return IndexStatus(has_index=False)
    
    # Try to get metadata from disk for additional info
    metadata = get_index_metadata(rag_name, INDICES_DIR)
    
    # Build IndexStatus response
    index_status = IndexStatus(
        has_index=True,
        document_count=stats["document_count"],
        embedding_dimension=stats["embedding_dimension"]
    )
    
    # Add metadata info if available
    if metadata:
        index_status.created_at = metadata.get("created_at")
        index_status.updated_at = metadata.get("updated_at")
        
        # Calculate total size from file_sizes
        file_sizes = metadata.get("file_sizes", {})
        total_size = sum(file_sizes.values())
        index_status.total_size_bytes = total_size if total_size > 0 else None
    
    return index_status


# ============ API Endpoints ============

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "RAG Benchmark API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# 启动时加载模型仓库和RAG实例
load_model_registry()
load_rag_registry()


# ============ Model Registry APIs ============

@app.post("/models/register")
async def register_model(model: ModelInfo):
    """注册模型到仓库"""
    try:
        model_registry[model.model_id] = model.model_dump()
        save_model_registry()
        
        return {
            "message": f"模型 '{model.model_id}' 注册成功",
            "model_id": model.model_id
        }
    except Exception as e:
        logger.error(f"注册模型失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/list")
async def list_models():
    """列出所有已注册的模型"""
    llm_models = [m for m in model_registry.values() if m["model_type"] == "llm"]
    embedding_models = [m for m in model_registry.values() if m["model_type"] == "embedding"]
    
    return {
        "llm_models": llm_models,
        "embedding_models": embedding_models,
        "total": len(model_registry)
    }


@app.get("/models/{model_id:path}")
async def get_model(model_id: str):
    """获取模型信息"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
    
    return model_registry[model_id]


@app.delete("/models/{model_id:path}")
async def delete_model(model_id: str):
    """删除模型"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
    
    del model_registry[model_id]
    save_model_registry()
    
    return {"message": f"模型 '{model_id}' 已删除"}


@app.put("/models/{model_id:path}")
async def update_model(model_id: str, model: ModelInfo):
    """更新模型信息"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
    
    model_registry[model_id] = model.model_dump()
    save_model_registry()
    
    return {
        "message": f"模型 '{model_id}' 更新成功",
        "model": model_registry[model_id]
    }


# ============ Dataset APIs ============

@app.get("/datasets", response_model=List[str])
async def list_datasets():
    """列出所有可用的数据集"""
    from rag_benchmark.datasets.registry import DATASET_REGISTRY
    return list(DATASET_REGISTRY._datasets.keys())


@app.post("/datasets/stats", response_model=DatasetStats)
async def get_dataset_stats(dataset_info: DatasetInfo):
    """获取数据集统计信息"""
    try:
        dataset = GoldenDataset(dataset_info.name, dataset_info.subset)
        stats = dataset.stats()
        return DatasetStats(**stats)
    except Exception as e:
        logger.error(f"获取数据集统计失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/datasets/sample")
async def sample_dataset(
    dataset_info: DatasetInfo,
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None
):
    """获取数据集样本（支持分页和搜索）
    
    Args:
        dataset_info: Dataset name and subset
        page: Page number (1-indexed, default: 1)
        page_size: Number of samples per page (default: 20, allowed: 10, 20, 50, 100)
        search: Optional search query for user_input field
        
    Returns:
        {
            "dataset_name": str,
            "subset": str,
            "total_count": int,
            "page": int,
            "page_size": int,
            "total_pages": int,
            "samples": List[dict]
        }
    """
    try:
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pagination parameters",
                    "details": "page must be >= 1",
                    "action": "Please provide a valid page number"
                }
            )
        
        allowed_page_sizes = [10, 20, 50, 100]
        if page_size not in allowed_page_sizes:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pagination parameters",
                    "details": f"page_size must be one of {allowed_page_sizes}",
                    "action": f"Please use one of the allowed page sizes: {allowed_page_sizes}"
                }
            )
        
        # Load dataset
        dataset = GoldenDataset(dataset_info.name, dataset_info.subset)
        
        # Apply search filter if provided
        if search:
            dataset = dataset.search(search, case_sensitive=False)
        
        # Get paginated samples
        samples, total_count = dataset.paginate(page, page_size)
        
        # Calculate total pages
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0
        
        # Validate page number is within range (after getting total_count)
        if total_count > 0 and page > total_pages:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pagination parameters",
                    "details": f"page {page} exceeds total pages {total_pages}",
                    "action": f"Please provide a page number between 1 and {total_pages}",
                    "total_pages": total_pages
                }
            )
        
        return {
            "dataset_name": dataset_info.name,
            "subset": dataset_info.subset,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "samples": [
                {
                    "id": s.id,
                    "user_input": s.user_input,
                    "reference": s.reference,
                    "reference_contexts": s.reference_contexts[:2],
                    "reference_context_ids": s.reference_context_ids
                }
                for s in samples
            ]
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"获取数据集样本失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============ RAG APIs ============

@app.post("/rag/create")
async def create_rag(request: CreateRAGRequest):
    """创建RAG实例"""
    try:
        rag_config = RAGConfig(
            top_k=request.rag_config.top_k if request.rag_config else 5,
            temperature=request.rag_config.temperature if request.rag_config else 0.7,
            max_length=request.rag_config.max_length if request.rag_config else 512
        )
        
        # 从模型仓库获取模型客户端
        llm = get_model_client(request.model_info.llm_model_id)
        embedding = get_model_client(request.model_info.embedding_model_id)
        
        # 设置temperature
        if hasattr(llm, 'temperature'):
            llm.temperature = rag_config.temperature
        
        rag = BaselineRAG(
            embedding_model=embedding,
            llm=llm,
            config=rag_config
        )
        
        rag_instances[request.name] = {
            "rag": rag,
            "rag_type": request.rag_type,
            "model_info": request.model_info.model_dump(),
            "rag_config": rag_config.to_dict(),
            "api_config": request.api_config,
            "created_at": datetime.now().isoformat()
        }
        
        # 保存到磁盘
        save_rag_instance(request.name)
        
        return {
            "message": f"RAG实例 '{request.name}' 创建成功",
            "name": request.name,
            "model_info": request.model_info.model_dump(),
            "rag_config": rag_config.to_dict()
        }
    except Exception as e:
        logger.error(f"创建RAG实例失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/rag/list")
async def list_rags():
    """列出所有RAG实例"""
    rags_list = []
    for name, info in rag_instances.items():
        # Get index status for this RAG
        index_status = get_rag_index_status_info(name)
        
        rags_list.append({
            "name": name,
            "rag_type": info.get("rag_type", "baseline"),
            "model_info": info["model_info"],
            "rag_config": info["rag_config"],
            "api_config": info.get("api_config"),
            "index_status": index_status.model_dump()
        })
    
    return {
        "rags": rags_list,
        "count": len(rag_instances)
    }


@app.delete("/rag/{rag_name}")
async def delete_rag(rag_name: str):
    """删除RAG实例"""
    if rag_name not in rag_instances:
        raise HTTPException(status_code=404, detail=f"RAG实例 '{rag_name}' 不存在")
    
    try:
        # Get RAG instance before deleting from memory
        rag_info = rag_instances[rag_name]
        rag: BaselineRAG = rag_info["rag"]
        
        # Delete from memory
        del rag_instances[rag_name]
        
        # Delete persisted config file
        delete_rag_instance_file(rag_name)
        
        # Delete persisted index data using RAG instance method
        index_path = INDICES_DIR / rag_name
        rag.delete_from_disk(index_path)
        
        logger.info(f"RAG实例 '{rag_name}' 已删除")
        return {
            "message": f"RAG实例 '{rag_name}' 删除成功",
            "name": rag_name
        }
    except Exception as e:
        logger.error(f"删除RAG实例失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/{rag_name}/index/status", response_model=IndexStatus)
async def get_rag_index_status(rag_name: str):
    """获取RAG实例的详细索引状态信息
    
    Args:
        rag_name: RAG实例名称
        
    Returns:
        IndexStatus对象，包含索引的详细信息
        
    Raises:
        HTTPException: 如果RAG实例不存在
    """
    if rag_name not in rag_instances:
        raise HTTPException(status_code=404, detail=f"RAG实例 '{rag_name}' 不存在")
    
    try:
        index_status = get_rag_index_status_info(rag_name)
        return index_status
    except Exception as e:
        logger.error(f"获取索引状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/corpus/preview")
async def preview_corpus(
    dataset_info: DatasetInfo,
    page: int = 1,
    page_size: int = 20
):
    """预览数据集的corpus文档（支持分页）
    
    Args:
        dataset_info: Dataset name and subset
        page: Page number (1-indexed, default: 1)
        page_size: Number of documents per page (default: 20, allowed: 10, 20, 50, 100)
        
    Returns:
        {
            "dataset_name": str,
            "subset": str,
            "total_count": int,
            "page": int,
            "page_size": int,
            "total_pages": int,
            "documents": List[dict]
        }
    """
    try:
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pagination parameters",
                    "details": "page must be >= 1",
                    "action": "Please provide a valid page number"
                }
            )
        
        allowed_page_sizes = [10, 20, 50, 100]
        if page_size not in allowed_page_sizes:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pagination parameters",
                    "details": f"page_size must be one of {allowed_page_sizes}",
                    "action": f"Please use one of the allowed page sizes: {allowed_page_sizes}"
                }
            )
        
        # Load dataset
        dataset = GoldenDataset(dataset_info.name, dataset_info.subset)
        corpus_records = list(dataset.iter_corpus())
        total_count = len(corpus_records)
        
        if total_count == 0:
            return {
                "dataset_name": dataset_info.name,
                "subset": dataset_info.subset,
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "documents": []
            }
        
        # Calculate total pages
        total_pages = (total_count + page_size - 1) // page_size
        
        # Validate page number is within range
        if page > total_pages:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pagination parameters",
                    "details": f"page {page} exceeds total pages {total_pages}",
                    "action": f"Please provide a page number between 1 and {total_pages}",
                    "total_pages": total_pages
                }
            )
        
        # Calculate pagination indices
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated records
        paginated_records = corpus_records[start_idx:end_idx]
        
        documents = [
            {
                "id": record.reference_context_id,
                "doc_id": record.reference_context_id,  # Alias for frontend compatibility
                "content": record.reference_context,
                "title": record.title,
                "metadata": record.metadata,
                "length": len(record.reference_context)  # For backward compatibility
            }
            for record in paginated_records
        ]
        
        return {
            "dataset_name": dataset_info.name,
            "subset": dataset_info.subset,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "documents": documents
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"预览corpus失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class CorpusByIdRequest(BaseModel):
    """Request model for getting corpus documents by IDs"""
    dataset_info: DatasetInfo
    document_ids: List[str]


@app.post("/datasets/corpus/by-id")
async def get_corpus_by_id(request: CorpusByIdRequest):
    """获取指定ID的corpus文档
    
    Args:
        request: Request containing dataset_info and document_ids
        
    Returns:
        {
            "dataset_name": str,
            "subset": str,
            "documents": List[dict]
        }
    """
    try:
        # Validate document_ids
        if not request.document_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid request parameters",
                    "details": "document_ids cannot be empty",
                    "action": "Please provide at least one document ID"
                }
            )
        
        # Load dataset
        dataset = GoldenDataset(request.dataset_info.name, request.dataset_info.subset)
        
        # Get corpus documents by IDs
        corpus_records = dataset.get_corpus_by_ids(request.document_ids)
        
        # Check if all requested IDs were found
        found_ids = {record.reference_context_id for record in corpus_records}
        missing_ids = [doc_id for doc_id in request.document_ids if doc_id not in found_ids]
        
        if missing_ids:
            logger.warning(f"Some document IDs not found: {missing_ids}")
        
        documents = [
            {
                "id": record.reference_context_id,
                "doc_id": record.reference_context_id,  # Alias for frontend compatibility
                "content": record.reference_context,
                "title": record.title,
                "metadata": record.metadata,
                "length": len(record.reference_context)  # For backward compatibility
            }
            for record in corpus_records
        ]
        
        return {
            "dataset_name": request.dataset_info.name,
            "subset": request.dataset_info.subset,
            "documents": documents,
            "requested_count": len(request.document_ids),
            "found_count": len(documents),
            "missing_ids": missing_ids if missing_ids else None
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"获取corpus文档失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class IndexDocumentsWithSelectionRequest(BaseModel):
    rag_name: str
    dataset_name: str
    subset: Optional[str] = None
    document_ids: Optional[List[str]] = None  # 如果为None，索引全部文档


@app.post("/rag/index")
async def index_documents(request: IndexDocumentsWithSelectionRequest):
    """为RAG实例索引文档（从数据集corpus加载，支持选择）"""
    if request.rag_name not in rag_instances:
        raise HTTPException(status_code=404, detail=f"RAG实例 '{request.rag_name}' 不存在")
    
    rag_info = rag_instances[request.rag_name]
    
    # 只有baseline类型的RAG才能索引文档
    if rag_info.get("rag_type", "baseline") != "baseline":
        raise HTTPException(
            status_code=400, 
            detail="只有baseline类型的RAG实例支持索引文档"
        )
    
    try:
        # 从数据集加载corpus
        dataset = GoldenDataset(request.dataset_name, request.subset)
        corpus_records = list(dataset.iter_corpus())
        
        if not corpus_records:
            raise ValueError(f"数据集 '{request.dataset_name}' 没有corpus数据")
        
        # 如果指定了document_ids，验证并只索引选中的文档
        if request.document_ids is not None:
            # Validate that all document IDs exist
            valid_doc_ids = {record.reference_context_id for record in corpus_records}
            invalid_doc_ids = [doc_id for doc_id in request.document_ids if doc_id not in valid_doc_ids]
            
            if invalid_doc_ids:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid document IDs provided",
                        "details": f"The following document IDs do not exist in the corpus: {invalid_doc_ids}",
                        "action": "Please verify document IDs and try again. Use POST /datasets/corpus/preview to see valid document IDs."
                    }
                )
            
            selected_records = [
                record for record in corpus_records 
                if record.reference_context_id in request.document_ids
            ]
            documents = [record.reference_context for record in selected_records]
            logger.info(f"从数据集 '{request.dataset_name}' 选择了 {len(documents)}/{len(corpus_records)} 个文档")
        else:
            # 索引全部文档
            documents = [record.reference_context for record in corpus_records]
            logger.info(f"从数据集 '{request.dataset_name}' 加载了 {len(documents)} 个文档")
        
        # 索引文档（使用批量处理）
        rag: BaselineRAG = rag_info["rag"]
        rag.index_documents(documents, batch_size=50)
        
        # Persist index data to disk using RAG instance method
        try:
            index_path = INDICES_DIR / request.rag_name
            rag.save_to_disk(index_path)
            logger.info(f"索引数据已持久化到磁盘: {request.rag_name}")
        except RuntimeError as e:
            # Log error but don't fail the request - RAG is still functional in memory
            logger.error(f"持久化索引数据失败: {e}")
        
        return {
            "message": f"成功从数据集 '{request.dataset_name}' 索引 {len(documents)} 个文档",
            "rag_name": request.rag_name,
            "dataset_name": request.dataset_name,
            "document_count": len(documents),
            "total_corpus_count": len(corpus_records)
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"索引文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rag/query")
async def query_rag(request: QueryRequest):
    """查询RAG系统"""
    if request.rag_name not in rag_instances:
        raise HTTPException(status_code=404, detail=f"RAG实例 '{request.rag_name}' 不存在")
    
    try:
        rag: RAGInterface = rag_instances[request.rag_name]["rag"]
        retrieve_result, generate_result = await rag.retrieve_and_generate(request.query, request.top_k)
        
        return {
            "query": request.query,
            "answer": generate_result.response,
            "contexts": retrieve_result.contexts,
            "scores": retrieve_result.scores
        }
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============ Evaluation APIs ============

@app.post("/evaluate/start")
async def start_evaluation(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """启动评测任务（异步）
    
    Validates sample selection parameters before creating the task.
    Returns immediate error for invalid sample IDs.
    """
    # Validate sample selection parameters before creating task
    if request.sample_selection:
        selection = request.sample_selection
        
        # Validate strategy
        valid_strategies = ["specific_ids", "random", "all"]
        if selection.strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid sample selection strategy",
                    "details": f"Strategy '{selection.strategy}' is not valid. Must be one of: {valid_strategies}",
                    "action": "Please use a valid strategy: 'specific_ids', 'random', or 'all'"
                }
            )
        
        # Validate specific_ids strategy
        if selection.strategy == "specific_ids":
            if not selection.sample_ids or len(selection.sample_ids) == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid sample selection parameters",
                        "details": "sample_ids is required and must not be empty for 'specific_ids' strategy",
                        "action": "Please provide a list of sample IDs to evaluate"
                    }
                )
            
            # Load dataset to validate sample IDs
            try:
                dataset = GoldenDataset(request.dataset_name, request.subset)
                invalid_ids = SampleSelector.validate_sample_ids(dataset, selection.sample_ids)
                
                if invalid_ids:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Invalid sample IDs provided",
                            "details": f"The following sample IDs do not exist in the dataset: {invalid_ids}",
                            "action": "Please verify sample IDs and try again. Use GET /datasets/sample to see valid sample IDs."
                        }
                    )
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Failed to validate sample IDs",
                        "details": str(e),
                        "action": "Please check that the dataset name and subset are correct"
                    }
                )
        
        # Validate random strategy
        elif selection.strategy == "random":
            if not selection.sample_size or selection.sample_size <= 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid sample selection parameters",
                        "details": "sample_size is required and must be greater than 0 for 'random' strategy",
                        "action": "Please provide a valid sample_size"
                    }
                )
            
            # Validate that sample_size doesn't exceed dataset size
            try:
                dataset = GoldenDataset(request.dataset_name, request.subset)
                total_samples = dataset.count()
                
                if selection.sample_size > total_samples:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Invalid sample selection parameters",
                            "details": f"sample_size ({selection.sample_size}) exceeds total dataset size ({total_samples})",
                            "action": f"Please provide a sample_size <= {total_samples}"
                        }
                    )
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Failed to validate sample_size",
                        "details": str(e),
                        "action": "Please check that the dataset name and subset are correct"
                    }
                )
        
        # Validate all strategy (no additional parameters needed)
        elif selection.strategy == "all":
            # No additional validation needed for "all" strategy
            pass
    
    task_id = str(uuid.uuid4())
    
    # Create task directory
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save task configuration to config.json
    config_data = {
        "dataset_name": request.dataset_name,
        "subset": request.subset,
        "rag_name": request.rag_name,
        "eval_type": request.eval_type,
        "model_info": request.model_info.model_dump(),
        "sample_selection": request.sample_selection.model_dump() if request.sample_selection else None
    }
    config_file = task_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Initialize sample_info (will be populated during execution)
    sample_info = None
    if request.sample_selection:
        # Pre-initialize with known information
        sample_info = {
            "selection_strategy": request.sample_selection.strategy,
            "total_samples": 0,  # Will be updated during execution
            "selected_samples": 0,  # Will be updated during execution
            "sample_ids": request.sample_selection.sample_ids if request.sample_selection.strategy == "specific_ids" else None,
            "completed_samples": 0,
            "failed_samples": 0
        }
    
    # Initialize task status
    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "current_stage": "初始化",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sample_info": sample_info
    }
    save_task_status(task_id)

    def run_evaluation_task_sync(task_id: str, request: EvaluateRequest):
        """简化版：直接用asyncio.run执行异步任务（自动管理事件循环）"""
        asyncio.run(run_evaluation_task(task_id, request))

    background_tasks.add_task(
        run_evaluation_task_sync,
        task_id,
        request
    )
    
    return {
        "task_id": task_id,
        "message": "评测任务已启动",
        "status_url": f"/evaluate/status/{task_id}"
    }


async def run_evaluation_task(task_id: str, request: EvaluateRequest):
    """运行评测任务（支持断点续传）"""
    try:
        # 检查是否有已保存的状态
        saved_status = load_task_status(task_id)
        if saved_status and saved_status.get("status") == "completed":
            logger.info(f"任务 {task_id} 已完成，跳过")
            return
        
        # Load checkpoint data if exists
        checkpoint = checkpoint_manager.load_checkpoint(task_id)
        completed_stages = checkpoint.get("completed_stages", []) if checkpoint else []
        
        # Handle corrupted checkpoint data
        if checkpoint and not isinstance(completed_stages, list):
            logger.warning(f"任务 {task_id}: 检查点数据损坏，从头开始")
            checkpoint_manager.clear_checkpoints(task_id)
            completed_stages = []
        
        logger.info(f"任务 {task_id}: 已完成阶段: {completed_stages}")
        
        # 阶段1: 加载数据集和样本选择
        if "load_dataset" not in completed_stages:
            update_task_status(
                task_id,
                status="running",
                progress=0.1,
                current_stage="加载数据集"
            )
            
            # Load full dataset
            dataset = GoldenDataset(request.dataset_name, request.subset)
            total_samples = dataset.count()
            
            # Determine selection strategy
            if request.sample_selection:
                selection = request.sample_selection
                selection_strategy = selection.strategy
            else:
                # Backward compatibility: no sample_selection means select all
                selection_strategy = "all"
                selection = None
            
            # Apply sample selection using SampleSelector
            # Note: Validation already done in start_evaluation endpoint
            if selection and selection.strategy == "specific_ids":
                dataset = SampleSelector.select_by_ids(dataset, selection.sample_ids)
                selected_sample_ids = selection.sample_ids
                logger.info(f"任务 {task_id}: 选择了 {len(selection.sample_ids)} 个特定样本")
                
            elif selection and selection.strategy == "random":
                dataset = SampleSelector.select_random(
                    dataset, 
                    selection.sample_size,
                    selection.random_seed
                )
                selected_sample_ids = None  # Random selection doesn't track specific IDs
                logger.info(f"任务 {task_id}: 随机选择了 {selection.sample_size} 个样本")
                
            elif selection and selection.strategy == "all":
                dataset = SampleSelector.select_all(dataset)
                selected_sample_ids = None  # All samples, no need to track IDs
                logger.info(f"任务 {task_id}: 选择了所有 {total_samples} 个样本")
                
            else:
                # Backward compatibility: no sample_selection means select all
                dataset = SampleSelector.select_all(dataset)
                selected_sample_ids = None
                logger.info(f"任务 {task_id}: 未指定样本选择策略，使用所有样本")
            
            selected_samples = dataset.count()
            
            # Initialize sample_info in task status
            sample_info = {
                "selection_strategy": selection_strategy,
                "total_samples": total_samples,
                "selected_samples": selected_samples,
                "sample_ids": selected_sample_ids,
                "completed_samples": 0,
                "failed_samples": 0
            }
            
            # Persist selected dataset to disk
            try:
                checkpoint_manager.save_golden_dataset(task_id, dataset)
            except IOError as e:
                logger.warning(f"任务 {task_id}: 持久化数据集失败: {e}")
                # Continue - we can still complete the task
            
            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                task_id,
                "load_dataset",
                {
                    "dataset_name": request.dataset_name,
                    "subset": request.subset,
                    "total_samples": total_samples,
                    "selected_samples": selected_samples,
                    "selection_strategy": selection_strategy,
                    "sample_ids": selected_sample_ids
                }
            )
            
            update_task_status(
                task_id, 
                progress=0.2, 
                current_stage="数据集加载完成",
                sample_info=sample_info
            )
        else:
            logger.info(f"任务 {task_id}: 跳过阶段 'load_dataset'（已完成）")
            # Load dataset directly from disk
            dataset = checkpoint_manager.load_golden_dataset(task_id)
            
            # Restore sample_info from checkpoint
            checkpoint_data = checkpoint.get("stage_data", {}).get("load_dataset", {})
            sample_info = {
                "selection_strategy": checkpoint_data.get("selection_strategy", "all"),
                "total_samples": checkpoint_data.get("total_samples", dataset.count()),
                "selected_samples": checkpoint_data.get("selected_samples", dataset.count()),
                "sample_ids": checkpoint_data.get("sample_ids"),
                "completed_samples": 0,
                "failed_samples": 0
            }
            
            update_task_status(
                task_id, 
                progress=0.2, 
                current_stage="数据集加载完成（从检查点恢复）",
                sample_info=sample_info
            )
        
        # 阶段2: 准备实验数据集
        if "prepare_experiment" not in completed_stages:
            update_task_status(task_id, progress=0.3, current_stage="准备实验数据集")
            
            # Get or create RAG instance
            if request.rag_name in rag_instances:
                rag = rag_instances[request.rag_name]["rag"]
            else:
                # 使用请求中的模型配置创建临时RAG
                rag_config = RAGConfig()
                llm = get_model_client(request.model_info.llm_model_id)
                embedding = get_model_client(request.model_info.embedding_model_id)
                
                rag = BaselineRAG(
                    embedding_model=embedding,
                    llm=llm,
                    config=rag_config
                )
            
            exp_ds = await prepare_experiment_dataset(dataset, rag)
            
            # Persist experiment dataset to disk
            try:
                checkpoint_manager.save_experiment_dataset(task_id, exp_ds)
            except IOError as e:
                logger.warning(f"任务 {task_id}: 持久化实验数据集失败: {e}")
                # Continue - we can still complete the task
            
            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                task_id,
                "prepare_experiment",
                {
                    "experiment_dataset_saved": True,
                    "experiment_size": len(exp_ds.samples)
                }
            )
            
            update_task_status(task_id, progress=0.5, current_stage="实验数据集准备完成")
        else:
            logger.info(f"任务 {task_id}: 跳过阶段 'prepare_experiment'（已完成）")
            # Load experiment dataset directly from disk
            exp_ds = checkpoint_manager.load_experiment_dataset(task_id)
            update_task_status(task_id, progress=0.5, current_stage="实验数据集准备完成（从检查点恢复）")
        
        # 阶段3: 运行评测
        if "run_evaluation" not in completed_stages:
            update_task_status(task_id, progress=0.6, current_stage=f"运行{request.eval_type}评测")
            
            # 使用评测请求中的模型配置
            llm = get_model_client(request.model_info.llm_model_id)
            embeddings = get_model_client(request.model_info.embedding_model_id)
            eval_llm = LangchainLLMWrapper(langchain_llm=llm, cache=cache)
            eval_embeddings = LangchainEmbeddingsWrapper(embeddings=embeddings, cache=cache)

            if request.eval_type == "e2e":
                result = await evaluate_e2e(
                    exp_ds,
                    experiment_name=f"{request.rag_name}_e2e",
                    llm=eval_llm,
                    embeddings=eval_embeddings
                )
            elif request.eval_type == "retrieval":
                result = await evaluate_retrieval(
                    exp_ds,
                    experiment_name=f"{request.rag_name}_retrieval",
                    llm=eval_llm,
                    embeddings=eval_embeddings
                )
            elif request.eval_type == "generation":
                result = await evaluate_generation(
                    exp_ds,
                    experiment_name=f"{request.rag_name}_generation",
                    llm=eval_llm,
                    embeddings=eval_embeddings
                )
            else:
                raise ValueError(f"不支持的评测类型: {request.eval_type}")
            
            # Save checkpoint with evaluation result
            checkpoint_manager.save_checkpoint(
                task_id,
                "run_evaluation",
                {
                    "eval_type": request.eval_type,
                    "evaluation_completed": True
                }
            )
            
            update_task_status(task_id, progress=0.8, current_stage="评测完成，处理结果")
        else:
            logger.info(f"任务 {task_id}: 跳过阶段 'run_evaluation'（已完成）")
            # This shouldn't happen in normal flow, but handle it
            raise ValueError("Cannot resume from completed evaluation - result not persisted")
        
        # 阶段4: 处理结果
        logger.info(f"任务 {task_id}: 开始处理评测结果")
        
        try:
            # 使用ragas内部计算好的指标平均值
            metrics = result._repr_dict
            # 获取详细结果（DataFrame格式，用于前端展开显示）
            df = result.to_pandas()
            sample_count = len(df)
            
            # 将DataFrame转换为记录列表（用于前端表格显示）
            detailed_results = df.to_dict('records')
            
            # Count completed and failed samples
            # In ragas with raise_exceptions=False (default), failed samples return np.nan for all metrics
            # A sample is considered failed only if ALL metric values are NaN/None
            completed_samples = 0
            failed_samples = 0
            
            # Get metric names from the evaluation result
            # These are the actual metrics computed by ragas for this evaluation
            metric_names = set(metrics.keys()) if metrics else set()
            
            for record in detailed_results:
                # Extract metric values using the actual metric names from the evaluation
                # This is more robust than hardcoding excluded columns
                metric_values = []
                for key, value in record.items():
                    if key in metric_names:
                        metric_values.append(value)
                
                # Check if all metrics are NaN/None (complete failure)
                # or if at least one metric has a valid value (partial or complete success)
                if not metric_values:
                    # No metrics found, consider as failed
                    failed_samples += 1
                else:
                    all_invalid = all(
                        value is None or (isinstance(value, float) and math.isnan(value))
                        for value in metric_values
                    )
                    if all_invalid:
                        failed_samples += 1
                    else:
                        completed_samples += 1
            
            logger.info(f"任务 {task_id}: 成功提取 {len(metrics)} 个指标")
            logger.info(f"任务 {task_id}: 样本数: {sample_count}, 成功: {completed_samples}, 失败: {failed_samples}")
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 提取指标失败: {e}", exc_info=True)
            metrics = {}
            detailed_results = []
            sample_count = 0
            completed_samples = 0
            failed_samples = 0
        
        logger.info(f"任务 {task_id}: 最终指标: {metrics}")
        
        # Update sample_info with final counts
        current_status = tasks_status.get(task_id, {})
        sample_info = current_status.get("sample_info", {})
        if sample_info:
            sample_info["completed_samples"] = completed_samples
            sample_info["failed_samples"] = failed_samples
        
        # Sanitize metrics and detailed_results to remove invalid float values
        sanitized_metrics = sanitize_float_values(metrics)
        sanitized_detailed_results = sanitize_float_values(detailed_results)
        
        update_task_status(
            task_id,
            status="completed",
            progress=1.0,
            current_stage="完成",
            result={
                "metrics": sanitized_metrics,  # 指标平均值字典
                "detailed_results": sanitized_detailed_results,  # 每个样本的详细结果
                "sample_count": sample_count,
                "eval_type": request.eval_type,
                "model_info": request.model_info.model_dump()
            },
            sample_info=sample_info
        )
        
        logger.info(f"评测任务 {task_id} 完成")
        
    except Exception as e:
        logger.error(f"评测任务失败: {e}", exc_info=True)
        update_task_status(
            task_id,
            status="failed",
            error=str(e)
        )


@app.get("/evaluate/status/{task_id}", response_model=TaskStatus)
async def get_evaluation_status(task_id: str):
    """获取评测任务状态
    
    Returns task status including:
    - Basic status information (status, progress, current_stage)
    - Sample tracking information (sample_info)
    - Checkpoint progress information (checkpoint_info)
    
    Args:
        task_id: Task identifier
        
    Returns:
        TaskStatus object with complete task information
        
    Raises:
        HTTPException: If task does not exist
    """
    # 先从内存查找
    if task_id not in tasks_status:
        # 再从磁盘加载
        saved_status = load_task_status(task_id)
        if saved_status:
            tasks_status[task_id] = saved_status
        else:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # Get task status
    task_status = tasks_status[task_id]
    
    # Load checkpoint information
    checkpoint = checkpoint_manager.load_checkpoint(task_id)
    checkpoint_info = None
    
    if checkpoint:
        checkpoint_info = CheckpointInfo(
            has_checkpoint=True,
            completed_stages=checkpoint.get("completed_stages", []),
            current_stage=checkpoint.get("current_stage"),
            last_checkpoint_at=checkpoint.get("last_checkpoint_at")
        )
    else:
        checkpoint_info = CheckpointInfo(
            has_checkpoint=False,
            completed_stages=[],
            current_stage=None,
            last_checkpoint_at=None
        )
    
    # Add checkpoint_info to task status
    task_status_with_checkpoint = {
        **task_status,
        "checkpoint_info": checkpoint_info.model_dump()
    }
    
    return TaskStatus(**task_status_with_checkpoint)


@app.get("/evaluate/tasks")
async def list_evaluation_tasks(status: Optional[str] = None):
    """列出所有评测任务
    
    支持按状态过滤任务列表
    
    Args:
        status: 可选的状态过滤器 (pending, running, completed, failed)
        
    Returns:
        包含任务列表和计数的字典：
        - tasks: 任务状态列表（包含sample_info）
        - count: 任务总数
        - filter: 应用的过滤器（如果有）
        
    Raises:
        HTTPException: 如果提供了无效的状态过滤器
    """
    # Validate status filter if provided
    valid_statuses = ["pending", "running", "completed", "failed"]
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid status filter",
                "details": f"Status '{status}' is not valid. Must be one of: {valid_statuses}",
                "action": f"Please use a valid status: {', '.join(valid_statuses)}"
            }
        )
    
    # Load all saved tasks from directory structure
    for task_dir in TASKS_DIR.iterdir():
        if task_dir.is_dir():
            task_id = task_dir.name
            if task_id not in tasks_status:
                status_file = task_dir / "status.json"
                if status_file.exists():
                    try:
                        with open(status_file, 'r') as f:
                            tasks_status[task_id] = json.load(f)
                    except Exception as e:
                        logger.error(f"加载任务状态失败 {task_id}: {e}")
                        continue
    
    # Get all tasks
    all_tasks = list(tasks_status.values())
    
    # Apply status filter if provided
    if status:
        filtered_tasks = [task for task in all_tasks if task.get("status") == status]
    else:
        filtered_tasks = all_tasks
    
    # Sanitize tasks to ensure JSON compliance
    sanitized_tasks = sanitize_float_values(filtered_tasks)
    
    response = {
        "tasks": sanitized_tasks,
        "count": len(sanitized_tasks)
    }
    
    # Include filter info if applied
    if status:
        response["filter"] = status
        response["total_count"] = len(all_tasks)
    
    return response


@app.get("/evaluate/{task_id}/samples")
async def get_task_samples(task_id: str):
    """获取评测任务的样本详情
    
    返回任务的样本选择策略、样本ID列表和每个样本的详细结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        包含样本信息和详细结果的字典：
        - sample_info: 样本选择信息（策略、总数、选中数等）
        - detailed_results: 每个样本的详细评测结果
        - task_status: 任务状态（pending, running, completed, failed）
        
    Raises:
        HTTPException: 如果任务不存在
    """
    # First check memory
    if task_id not in tasks_status:
        # Try loading from disk
        saved_status = load_task_status(task_id)
        if saved_status:
            tasks_status[task_id] = saved_status
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Evaluation task not found",
                    "details": f"Task ID '{task_id}' does not exist",
                    "action": "Use GET /evaluate/tasks to list available tasks"
                }
            )
    
    task_status = tasks_status[task_id]
    
    # Extract sample_info
    sample_info = task_status.get("sample_info")
    
    # Extract detailed results if task is completed
    detailed_results = []
    if task_status.get("status") == "completed" and task_status.get("result"):
        detailed_results = task_status["result"].get("detailed_results", [])
    
    # Build response
    response = {
        "task_id": task_id,
        "task_status": task_status.get("status"),
        "sample_info": sample_info,
        "detailed_results": detailed_results,
        "result_count": len(detailed_results)
    }
    
    return response


@app.delete("/evaluate/delete/{task_id}")
async def delete_evaluation_task(task_id: str):
    """删除评测任务记录
    
    删除任务的所有相关文件和数据，包括：
    - 任务目录及其所有文件（status.json, config.json, checkpoints等）
    - 内存中的任务状态
    
    注意：此操作不会停止正在运行的任务进程，仅删除记录
    
    Args:
        task_id: 任务ID
        
    Returns:
        成功确认消息
        
    Raises:
        HTTPException: 如果任务不存在
    """
    import shutil
    
    # Check if task exists in memory or on disk
    task_dir = TASKS_DIR / task_id
    task_exists_in_memory = task_id in tasks_status
    task_exists_on_disk = task_dir.exists()
    
    if not task_exists_in_memory and not task_exists_on_disk:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Evaluation task not found",
                "details": f"Task ID '{task_id}' does not exist",
                "action": "Use GET /evaluate/tasks to list available tasks"
            }
        )
    
    try:
        # Remove from in-memory tasks_status dict
        if task_exists_in_memory:
            del tasks_status[task_id]
            logger.info(f"任务 '{task_id}' 已从内存中删除")
        
        # Remove task directory and all files
        if task_exists_on_disk:
            shutil.rmtree(task_dir)
            logger.info(f"任务目录 '{task_dir}' 已删除")
        
        return {
            "message": f"Evaluation task '{task_id}' deleted successfully",
            "task_id": task_id,
            "deleted_from_memory": task_exists_in_memory,
            "deleted_from_disk": task_exists_on_disk
        }
        
    except Exception as e:
        logger.error(f"删除任务 '{task_id}' 失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to delete evaluation task",
                "details": str(e),
                "task_id": task_id
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
