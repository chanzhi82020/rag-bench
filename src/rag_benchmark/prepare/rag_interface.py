"""RAG系统抽象接口定义"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from watchfiles import awatch

from rag_benchmark.common.executor import task_wrapper


@dataclass
class RetrievalResult:
    """检索结果
    
    Attributes:
        contexts: 检索到的上下文文本列表
        context_ids: 上下文ID列表（可选）
        scores: 相似度分数列表（可选）
        metadata: 额外元数据（可选）
    """
    contexts: List[str]
    context_ids: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """验证数据一致性"""
        if self.context_ids is not None and len(self.context_ids) != len(self.contexts):
            raise ValueError("context_ids length must match contexts length")
        if self.scores is not None and len(self.scores) != len(self.contexts):
            raise ValueError("scores length must match contexts length")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "contexts": self.contexts,
            "context_ids": self.context_ids,
            "scores": self.scores,
            "metadata": self.metadata,
        }


@dataclass
class GenerationResult:
    """生成结果
    
    Attributes:
        response: 主要答案
        multi_responses: 多个候选答案（可选）
        confidence: 置信度（可选，范围0-1）
        metadata: 额外元数据（可选）
    """
    response: str
    multi_responses: Optional[List[str]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """验证数据"""
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "response": self.response,
            "multi_responses": self.multi_responses,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class RAGConfig:
    """RAG系统配置

    Attributes:
        # 检索配置
        top_k: 检索返回的top-k个结果
        similarity_threshold: 相似度阈值，低于此值的结果将被过滤
        retrieval_mode: 检索模式（dense, sparse, hybrid）

        # 生成配置
        max_length: 生成答案的最大长度
        temperature: 生成温度，控制随机性
        top_p: nucleus sampling参数

        # 其他配置
        batch_size: 批处理大小
        timeout: 超时时间（秒）
        extra_params: 额外的自定义参数
    """

    # 检索配置
    top_k: int = 5
    similarity_threshold: float = 0.0
    retrieval_mode: str = "dense"  # dense, sparse, hybrid

    # 生成配置
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # 其他配置
    batch_size: int = 10
    timeout: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_mode": self.retrieval_mode,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "extra_params": self.extra_params,
        }


class RAGInterface(ABC):
    """RAG系统抽象接口

    用户需要实现此接口来集成自己的RAG系统。

    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """初始化RAG系统

        Args:
            config: RAG配置，如果为None则使用默认配置
        """
        self.config = config or RAGConfig()
        self.semaphore = asyncio.Semaphore(self.config.batch_size)

    @abstractmethod
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """检索相关上下文

        Args:
            query: 用户查询
            top_k: 返回top-k个结果，如果为None则使用config中的值

        Returns:
            RetrievalResult实例，包含contexts和可选的元数据

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclass must implement retrieve()")

    @abstractmethod
    async def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """基于上下文生成答案

        Args:
            query: 用户查询
            contexts: 检索到的上下文列表

        Returns:
            GenerationResult实例，包含response和可选的元数据

        Raises:
            NotImplementedError: 子类必须实现此方法
            
        """
        raise NotImplementedError("Subclass must implement generate()")

    async def retrieve_and_generate(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[RetrievalResult, GenerationResult]:
        """检索并生成答案（便捷方法）

        Args:
            query: 用户查询
            top_k: 返回top-k个结果

        Returns:
            (RetrievalResult, GenerationResult)元组
        """
        retrieval_result = await self.retrieve(query, top_k)
        generation_result = await self.generate(query, retrieval_result.contexts)
        return retrieval_result, generation_result

    async def batch_retrieve(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """批量检索（默认实现，子类可以重写以优化性能）

        Args:
            queries: 查询列表
            top_k: 返回top-k个结果

        Returns:
            每个查询对应的RetrievalResult列表
        """
        tasks = [
            task_wrapper(self.semaphore, self.retrieve, query=query, top_k=top_k)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

    async def batch_generate(
        self, queries: List[str], contexts_list: List[List[str]]
    ) -> List[GenerationResult]:
        """批量生成（默认实现，子类可以重写以优化性能）

        Args:
            queries: 查询列表
            contexts_list: 每个查询对应的上下文列表

        Returns:
            每个查询对应的GenerationResult列表
        """
        tasks = [
            task_wrapper(self.semaphore, self.generate, query=query, contexts=contexts)
            for query, contexts in zip(queries, contexts_list)
        ]
        return await asyncio.gather(*tasks)

    async def batch_retrieve_and_generate(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[RetrievalResult, GenerationResult]]:
        """批量检索并生成（默认实现，子类可以重写以优化性能）

        Args:
            queries: 查询列表
            top_k: 返回top-k个结果

        Returns:
            每个查询对应的(RetrievalResult, GenerationResult)元组列表
        """
        retrieval_results = await self.batch_retrieve(queries, top_k)
        generation_results = await self.batch_generate(
            queries, [r.contexts for r in retrieval_results]
        )
        return list(zip(retrieval_results, generation_results))
