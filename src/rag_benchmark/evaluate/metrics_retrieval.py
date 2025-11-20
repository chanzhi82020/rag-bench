"""检索阶段专用评测指标

实现传统信息检索指标，用于评测RAG系统的检索性能。
这些指标需要retrieved_context_ids和reference_context_ids。
"""

import logging
from typing import List, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """计算Recall@K
    
    Recall@K = |retrieved_ids[:k] ∩ reference_ids| / |reference_ids|
    
    Args:
        retrieved_ids: 检索到的文档ID列表（按相关性排序）
        reference_ids: 参考答案的文档ID列表
        k: 只考虑前k个检索结果，None表示考虑全部
        
    Returns:
        Recall@K分数 (0.0 - 1.0)
        
    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4"]
        >>> reference = ["doc2", "doc5"]
        >>> recall_at_k(retrieved, reference, k=3)
        0.5  # 找到了doc2，但没找到doc5
    """
    if not reference_ids:
        logger.warning("reference_ids is empty, returning 0.0")
        return 0.0
    
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    retrieved_set = set(retrieved_ids)
    reference_set = set(reference_ids)
    
    hits = len(retrieved_set & reference_set)
    return hits / len(reference_set)


def precision_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """计算Precision@K
    
    Precision@K = |retrieved_ids[:k] ∩ reference_ids| / k
    
    Args:
        retrieved_ids: 检索到的文档ID列表（按相关性排序）
        reference_ids: 参考答案的文档ID列表
        k: 只考虑前k个检索结果，None表示考虑全部
        
    Returns:
        Precision@K分数 (0.0 - 1.0)
        
    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4"]
        >>> reference = ["doc2", "doc5"]
        >>> precision_at_k(retrieved, reference, k=3)
        0.333  # 3个结果中有1个相关
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    if not retrieved_ids:
        logger.warning("retrieved_ids is empty, returning 0.0")
        return 0.0
    
    retrieved_set = set(retrieved_ids)
    reference_set = set(reference_ids)
    
    hits = len(retrieved_set & reference_set)
    return hits / len(retrieved_ids)


def f1_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """计算F1@K
    
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
    
    Args:
        retrieved_ids: 检索到的文档ID列表（按相关性排序）
        reference_ids: 参考答案的文档ID列表
        k: 只考虑前k个检索结果，None表示考虑全部
        
    Returns:
        F1@K分数 (0.0 - 1.0)
    """
    precision = precision_at_k(retrieved_ids, reference_ids, k)
    recall = recall_at_k(retrieved_ids, reference_ids, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    reference_ids: List[str]
) -> float:
    """计算Mean Reciprocal Rank (MRR)
    
    MRR = 1 / rank_of_first_relevant_item
    
    Args:
        retrieved_ids: 检索到的文档ID列表（按相关性排序）
        reference_ids: 参考答案的文档ID列表
        
    Returns:
        MRR分数 (0.0 - 1.0)
        
    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4"]
        >>> reference = ["doc2", "doc5"]
        >>> mean_reciprocal_rank(retrieved, reference)
        0.5  # doc2在第2位，所以MRR = 1/2
    """
    reference_set = set(reference_ids)
    
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in reference_set:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """计算Normalized Discounted Cumulative Gain (NDCG@K)
    
    NDCG@K = DCG@K / IDCG@K
    DCG@K = Σ(rel_i / log2(i+1)) for i in 1..k
    
    Args:
        retrieved_ids: 检索到的文档ID列表（按相关性排序）
        reference_ids: 参考答案的文档ID列表
        k: 只考虑前k个检索结果，None表示考虑全部
        
    Returns:
        NDCG@K分数 (0.0 - 1.0)
        
    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4"]
        >>> reference = ["doc2", "doc3"]
        >>> ndcg_at_k(retrieved, reference, k=4)
        0.786  # 考虑位置的相关性得分
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    if not retrieved_ids:
        return 0.0
    
    reference_set = set(reference_ids)
    
    # 计算DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids, start=1):
        relevance = 1.0 if doc_id in reference_set else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # 计算IDCG (理想情况下的DCG)
    # 理想情况是所有相关文档都在最前面
    ideal_retrieved = list(reference_ids) + [
        doc_id for doc_id in retrieved_ids if doc_id not in reference_set
    ]
    ideal_retrieved = ideal_retrieved[:len(retrieved_ids)]
    
    idcg = 0.0
    for i, doc_id in enumerate(ideal_retrieved, start=1):
        relevance = 1.0 if doc_id in reference_set else 0.0
        idcg += relevance / np.log2(i + 1)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def average_precision(
    retrieved_ids: List[str],
    reference_ids: List[str]
) -> float:
    """计算Average Precision (AP)
    
    AP = (Σ(P@k * rel_k)) / |reference_ids|
    
    Args:
        retrieved_ids: 检索到的文档ID列表（按相关性排序）
        reference_ids: 参考答案的文档ID列表
        
    Returns:
        AP分数 (0.0 - 1.0)
    """
    if not reference_ids:
        return 0.0
    
    reference_set = set(reference_ids)
    
    precision_sum = 0.0
    hits = 0
    
    for k, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in reference_set:
            hits += 1
            precision_at_this_k = hits / k
            precision_sum += precision_at_this_k
    
    return precision_sum / len(reference_ids)


# 批量计算函数
def compute_retrieval_metrics(
    retrieved_ids_list: List[List[str]],
    reference_ids_list: List[List[str]],
    k_values: List[int] = [1, 3, 5, 10]
) -> dict:
    """批量计算检索指标
    
    Args:
        retrieved_ids_list: 多个查询的检索结果ID列表
        reference_ids_list: 多个查询的参考答案ID列表
        k_values: 要计算的k值列表
        
    Returns:
        包含各种指标平均值的字典
        
    Example:
        >>> retrieved_list = [
        ...     ["doc1", "doc2", "doc3"],
        ...     ["doc4", "doc5", "doc6"]
        ... ]
        >>> reference_list = [
        ...     ["doc2", "doc7"],
        ...     ["doc4", "doc8"]
        ... ]
        >>> metrics = compute_retrieval_metrics(retrieved_list, reference_list)
        >>> print(metrics["recall@5"])
        0.5
    """
    if len(retrieved_ids_list) != len(reference_ids_list):
        raise ValueError("retrieved_ids_list and reference_ids_list must have same length")
    
    n_samples = len(retrieved_ids_list)
    results = {}
    
    # 计算各种k值的指标
    for k in k_values:
        recall_scores = []
        precision_scores = []
        f1_scores = []
        ndcg_scores = []
        
        for retrieved, reference in zip(retrieved_ids_list, reference_ids_list):
            recall_scores.append(recall_at_k(retrieved, reference, k))
            precision_scores.append(precision_at_k(retrieved, reference, k))
            f1_scores.append(f1_at_k(retrieved, reference, k))
            ndcg_scores.append(ndcg_at_k(retrieved, reference, k))
        
        results[f"recall@{k}"] = np.mean(recall_scores)
        results[f"precision@{k}"] = np.mean(precision_scores)
        results[f"f1@{k}"] = np.mean(f1_scores)
        results[f"ndcg@{k}"] = np.mean(ndcg_scores)
    
    # 计算MRR和MAP
    mrr_scores = []
    map_scores = []
    
    for retrieved, reference in zip(retrieved_ids_list, reference_ids_list):
        mrr_scores.append(mean_reciprocal_rank(retrieved, reference))
        map_scores.append(average_precision(retrieved, reference))
    
    results["mrr"] = np.mean(mrr_scores)
    results["map"] = np.mean(map_scores)
    
    return results



# ============================================================================
# RAGAS Metric Integration Classes
# ============================================================================
# 以下类将传统IR指标包装为RAGAS Metric，使其可以与evaluate()函数集成

from dataclasses import dataclass, field
from ragas.metrics.base import SingleTurnMetric, MetricType
from ragas.dataset_schema import SingleTurnSample
from langchain_core.callbacks import Callbacks


@dataclass
class RecallAtK(SingleTurnMetric):
    """Recall@K metric for RAGAS integration
    
    计算检索结果的召回率。需要数据集包含retrieved_context_ids和reference_context_ids字段。
    
    Args:
        k: 只考虑前k个检索结果
        
    Example:
        >>> from rag_benchmark.evaluate import evaluate
        >>> from rag_benchmark.evaluate.metrics_retrieval import RecallAtK
        >>> 
        >>> result = evaluate(
        ...     dataset=exp_ds,
        ...     metrics=[RecallAtK(k=5)],
        ...     name="retrieval_eval"
        ... )
    """
    
    k: int = 5
    name: str = field(default=f"recall@{k}", repr=True)

    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )

    def init(self, run_config):
        """Initialize the metric (required by RAGAS)"""
        pass
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """异步计算单个样本的得分"""
        retrieved_ids = getattr(sample, "retrieved_context_ids", [])
        reference_ids = getattr(sample, "reference_context_ids", [])
        
        if not retrieved_ids or not reference_ids:
            logger.warning(f"Missing context_ids for {self.name}, returning 0.0")
            return 0.0
        
        return recall_at_k(retrieved_ids, reference_ids, self.k)


@dataclass
class PrecisionAtK(SingleTurnMetric):
    """Precision@K metric for RAGAS integration
    
    计算检索结果的精确率。需要数据集包含retrieved_context_ids和reference_context_ids字段。
    
    Args:
        k: 只考虑前k个检索结果
    """
    
    k: int = 5
    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )

    name: str = field(default=f"precision@{k}", repr=True)

    def init(self, run_config):
        """Initialize the metric (required by RAGAS)"""
        pass
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_ids = getattr(sample, "retrieved_context_ids", [])
        reference_ids = getattr(sample, "reference_context_ids", [])
        
        if not retrieved_ids or not reference_ids:
            logger.warning(f"Missing context_ids for {self.name}, returning 0.0")
            return 0.0
        
        return precision_at_k(retrieved_ids, reference_ids, self.k)


@dataclass
class F1AtK(SingleTurnMetric):
    """F1@K metric for RAGAS integration
    
    计算检索结果的F1分数。需要数据集包含retrieved_context_ids和reference_context_ids字段。
    
    Args:
        k: 只考虑前k个检索结果
    """
    
    k: int = 5
    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )

    name: str = field(default=f"f1@{k}", repr=True)

    def init(self, run_config):
        """Initialize the metric (required by RAGAS)"""
        pass
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_ids = getattr(sample, "retrieved_context_ids", [])
        reference_ids = getattr(sample, "reference_context_ids", [])
        
        if not retrieved_ids or not reference_ids:
            logger.warning(f"Missing context_ids for {self.name}, returning 0.0")
            return 0.0
        
        return f1_at_k(retrieved_ids, reference_ids, self.k)


@dataclass
class NDCGAtK(SingleTurnMetric):
    """NDCG@K metric for RAGAS integration
    
    计算归一化折损累积增益。需要数据集包含retrieved_context_ids和reference_context_ids字段。
    
    Args:
        k: 只考虑前k个检索结果
    """
    
    k: int = 10
    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )
    name: str = field(default=f"ndcg@{k}", repr=True)

    def init(self, run_config):
        """Initialize the metric (required by RAGAS)"""
        pass
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_ids = getattr(sample, "retrieved_context_ids", [])
        reference_ids = getattr(sample, "reference_context_ids", [])
        
        if not retrieved_ids or not reference_ids:
            logger.warning(f"Missing context_ids for {self.name}, returning 0.0")
            return 0.0
        
        return ndcg_at_k(retrieved_ids, reference_ids, self.k)


@dataclass
class MRRMetric(SingleTurnMetric):
    """Mean Reciprocal Rank (MRR) metric for RAGAS integration
    
    计算平均倒数排名。需要数据集包含retrieved_context_ids和reference_context_ids字段。
    """

    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )
    name: str = field(default="mrr", repr=True)


    def init(self, run_config):
        """Initialize the metric (required by RAGAS)"""
        pass
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_ids = getattr(sample, "retrieved_context_ids", [])
        reference_ids = getattr(sample, "reference_context_ids", [])
        
        if not retrieved_ids or not reference_ids:
            logger.warning(f"Missing context_ids for {self.name}, returning 0.0")
            return 0.0
        
        return mean_reciprocal_rank(retrieved_ids, reference_ids)


@dataclass
class MAPMetric(SingleTurnMetric):
    """Mean Average Precision (MAP) metric for RAGAS integration
    
    计算平均精确率。需要数据集包含retrieved_context_ids和reference_context_ids字段。
    """

    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )
    name: str = field(default="map", repr=True)

    def init(self, run_config):
        """Initialize the metric (required by RAGAS)"""
        pass
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_ids = getattr(sample, "retrieved_context_ids", [])
        reference_ids = getattr(sample, "reference_context_ids", [])
        
        if not retrieved_ids or not reference_ids:
            logger.warning(f"Missing context_ids for {self.name}, returning 0.0")
            return 0.0
        
        return average_precision(retrieved_ids, reference_ids)
