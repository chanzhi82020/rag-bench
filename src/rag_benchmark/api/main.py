"""RAG Benchmark API Server

FastAPI服务，暴露RAG Benchmark的核心功能
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uuid
import json
from pathlib import Path

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, BaselineRAG, RAGConfig
from rag_benchmark.evaluate import evaluate_e2e, evaluate_retrieval, evaluate_generation
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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
RAGS_DIR = Path("data/rags")
RAGS_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR = Path("data/indices")
INDICES_DIR.mkdir(parents=True, exist_ok=True)


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


class EvaluateRequest(BaseModel):
    dataset_name: str
    subset: Optional[str] = None
    rag_name: str
    eval_type: str = "e2e"  # e2e, retrieval, generation
    sample_size: Optional[int] = None
    model_info: ModelConfig  # 评测时的模型配置


class IndexStatus(BaseModel):
    """Index status information for a RAG instance"""
    has_index: bool
    document_count: Optional[int] = None
    embedding_dimension: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    total_size_bytes: Optional[int] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float
    current_stage: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ============ Helper Functions ============

def save_task_status(task_id: str):
    """保存任务状态到磁盘"""
    task_file = TASKS_DIR / f"{task_id}.json"
    with open(task_file, 'w') as f:
        json.dump(tasks_status[task_id], f, indent=2)


def load_task_status(task_id: str) -> Optional[Dict]:
    """从磁盘加载任务状态"""
    task_file = TASKS_DIR / f"{task_id}.json"
    if task_file.exists():
        with open(task_file, 'r') as f:
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
async def sample_dataset(dataset_info: DatasetInfo, n: int = 5):
    """获取数据集样本"""
    try:
        dataset = GoldenDataset(dataset_info.name, dataset_info.subset)
        samples = dataset.head(n)
        return {
            "dataset_name": dataset_info.name,
            "subset": dataset_info.subset,
            "count": len(samples),
            "samples": [
                {
                    "user_input": s.user_input,
                    "reference": s.reference,
                    "reference_contexts": s.reference_contexts[:2]
                }
                for s in samples
            ]
        }
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
async def preview_corpus(dataset_info: DatasetInfo, limit: int = 100):
    """预览数据集的corpus文档"""
    try:
        dataset = GoldenDataset(dataset_info.name, dataset_info.subset)
        corpus_records = list(dataset.iter_corpus())
        
        if not corpus_records:
            return {
                "dataset_name": dataset_info.name,
                "subset": dataset_info.subset,
                "total_count": 0,
                "preview_count": 0,
                "documents": []
            }
        
        # 限制预览数量
        preview_records = corpus_records[:limit]
        
        documents = [
            {
                "id": record.reference_context_id,
                "content": record.reference_context,
                "length": len(record.reference_context)
            }
            for record in preview_records
        ]
        
        return {
            "dataset_name": dataset_info.name,
            "subset": dataset_info.subset,
            "total_count": len(corpus_records),
            "preview_count": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"预览corpus失败: {e}")
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
        
        # 如果指定了document_ids，只索引选中的文档
        if request.document_ids:
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
    except Exception as e:
        logger.error(f"索引文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rag/query")
async def query_rag(request: QueryRequest):
    """查询RAG系统"""
    if request.rag_name not in rag_instances:
        raise HTTPException(status_code=404, detail=f"RAG实例 '{request.rag_name}' 不存在")
    
    try:
        rag = rag_instances[request.rag_name]["rag"]
        result = rag.query(request.query, request.top_k)
        
        return {
            "query": result["query"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "scores": result.get("scores", [])
        }
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============ Evaluation APIs ============

@app.post("/evaluate/start")
async def start_evaluation(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """启动评测任务（异步）"""
    task_id = str(uuid.uuid4())
    
    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "current_stage": "初始化",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "request": {
            "dataset_name": request.dataset_name,
            "subset": request.subset,
            "rag_name": request.rag_name,
            "eval_type": request.eval_type,
            "sample_size": request.sample_size,
            "model_info": request.model_info.model_dump()
        }
    }
    save_task_status(task_id)
    
    background_tasks.add_task(
        run_evaluation_task,
        task_id,
        request
    )
    
    return {
        "task_id": task_id,
        "message": "评测任务已启动",
        "status_url": f"/evaluate/status/{task_id}"
    }


def run_evaluation_task(task_id: str, request: EvaluateRequest):
    """运行评测任务（支持断点续传）"""
    try:
        # 检查是否有已保存的状态
        saved_status = load_task_status(task_id)
        if saved_status and saved_status.get("status") == "completed":
            logger.info(f"任务 {task_id} 已完成，跳过")
            return
        
        # 阶段1: 加载数据集
        update_task_status(
            task_id,
            status="running",
            progress=0.1,
            current_stage="加载数据集"
        )
        
        dataset = GoldenDataset(request.dataset_name, request.subset)
        if request.sample_size:
            # 使用create_subset方法创建子数据集
            dataset = dataset.create_subset(request.sample_size)
        
        update_task_status(task_id, progress=0.2, current_stage="数据集加载完成")
        
        # 阶段2: 准备实验数据集
        update_task_status(task_id, progress=0.3, current_stage="准备实验数据集")
        
        # 使用评测请求中的模型配置创建临时RAG
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
        
        exp_ds = prepare_experiment_dataset(dataset, rag)
        update_task_status(task_id, progress=0.5, current_stage="实验数据集准备完成")
        
        # 阶段3: 运行评测
        update_task_status(task_id, progress=0.6, current_stage=f"运行{request.eval_type}评测")
        
        # 使用评测请求中的模型配置
        eval_llm = get_model_client(request.model_info.llm_model_id)
        eval_embeddings = get_model_client(request.model_info.embedding_model_id)
        
        if request.eval_type == "e2e":
            result = evaluate_e2e(
                exp_ds,
                experiment_name=f"{request.rag_name}_e2e",
                llm=eval_llm,
                embeddings=eval_embeddings
            )
        elif request.eval_type == "retrieval":
            result = evaluate_retrieval(
                exp_ds,
                experiment_name=f"{request.rag_name}_retrieval",
                llm=eval_llm,
                embeddings=eval_embeddings
            )
        elif request.eval_type == "generation":
            result = evaluate_generation(
                exp_ds,
                experiment_name=f"{request.rag_name}_generation",
                llm=eval_llm,
                embeddings=eval_embeddings
            )
        else:
            raise ValueError(f"不支持的评测类型: {request.eval_type}")
        
        update_task_status(task_id, progress=0.8, current_stage="评测完成，处理结果")
        
        # 阶段4: 处理结果
        # ragas的EvaluationResult内部已经计算好了指标平均值
        # result._repr_dict 包含所有指标的平均值
        # result.scores 包含每个样本的详细评测结果
        logger.info(f"任务 {task_id}: 开始处理评测结果")
        
        try:
            # 方法1: 使用ragas内部计算好的指标平均值
            metrics = result._repr_dict
            # 获取详细结果（DataFrame格式，用于前端展开显示）
            df = result.to_pandas()
            sample_count = len(df)
            
            # 将DataFrame转换为记录列表（用于前端表格显示）
            detailed_results = df.to_dict('records')
            
            logger.info(f"任务 {task_id}: 成功提取 {len(metrics)} 个指标")
            logger.info(f"任务 {task_id}: 样本数: {sample_count}")
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 提取指标失败: {e}", exc_info=True)
            metrics = {}
            detailed_results = []
            sample_count = 0
        
        logger.info(f"任务 {task_id}: 最终指标: {metrics}")
        
        update_task_status(
            task_id,
            status="completed",
            progress=1.0,
            current_stage="完成",
            result={
                "metrics": metrics,  # 指标平均值字典
                "detailed_results": detailed_results,  # 每个样本的详细结果
                "sample_count": sample_count,
                "eval_type": request.eval_type,
                "model_info": request.model_info.model_dump()
            }
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
    """获取评测任务状态"""
    # 先从内存查找
    if task_id in tasks_status:
        return TaskStatus(**tasks_status[task_id])
    
    # 再从磁盘加载
    saved_status = load_task_status(task_id)
    if saved_status:
        tasks_status[task_id] = saved_status
        return TaskStatus(**saved_status)
    
    raise HTTPException(status_code=404, detail="任务不存在")


@app.get("/evaluate/tasks")
async def list_evaluation_tasks():
    """列出所有评测任务"""
    # 加载所有已保存的任务
    for task_file in TASKS_DIR.glob("*.json"):
        task_id = task_file.stem
        if task_id not in tasks_status:
            with open(task_file, 'r') as f:
                tasks_status[task_id] = json.load(f)
    
    return {
        "tasks": list(tasks_status.values()),
        "count": len(tasks_status)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
