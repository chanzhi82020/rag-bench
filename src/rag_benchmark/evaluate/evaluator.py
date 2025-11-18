"""核心评估器实现"""

import logging
import typing as t
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from ragas import evaluate as ragas_evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.executor import Executor
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig
from ragas.cost import TokenUsageParser
from tqdm import tqdm

from .results import EvaluationResult, MetricResult

logger = logging.getLogger(__name__)

# 预定义的指标组合
METRIC_GROUPS = {
    "retrieval": [context_recall, context_precision],
    "generation": [faithfulness, answer_correctness],
    "e2e": [faithfulness, answer_relevancy, context_precision, context_recall],
    "all": [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    ],
}


class EvaluationError(Exception):
    """评测过程中的错误"""

    pass


def evaluate(
    dataset: EvaluationDataset,
    metrics: Union[List[Metric], str],
    name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    callbacks: Optional[Callbacks] = None,
    run_config: Optional[RunConfig] = None,
    token_usage_parser: Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    return_executor: bool = False,
    allow_nest_asyncio: bool = True,
    **kwargs,
) -> Union[EvaluationResult, Executor]:
    """评测RAG系统

    Args:
        dataset: RAGAS EvaluationDataset实例
        metrics: 评测指标列表或预定义组合名称（"retrieval", "generation", "e2e", "all"）
        name: 评测名称（对应RAGAS的experiment_name）
        llm: 用于评测的LLM模型（可选，如果不提供则使用RAGAS默认）
        embeddings: 用于评测的Embedding模型（可选，如果不提供则使用RAGAS默认）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置，如超时和重试（可选）
        token_usage_parser: Token使用解析器（可选）
        raise_exceptions: 是否抛出异常，False时失败的行返回np.nan（默认False）
        column_map: 数据集列名映射（可选）
        show_progress: 是否显示进度条（默认True）
        batch_size: 批处理大小（可选）
        return_executor: 是否返回Executor实例而不是运行评测（默认False）
        allow_nest_asyncio: 是否允许nest_asyncio补丁（默认True）
        **kwargs: 传递给RAGAS的其他参数

    Returns:
        EvaluationResult实例（如果return_executor=False）
        或Executor实例（如果return_executor=True）

    Raises:
        EvaluationError: 评测失败时
        ValueError: 参数错误时

    Example:
        >>> from ragas.metrics import faithfulness, answer_relevancy
        >>> 
        >>> # 使用默认模型
        >>> result = evaluate(
        ...     dataset=exp_ds,
        ...     metrics=[faithfulness, answer_relevancy],
        ...     name="my_rag_system"
        ... )
        >>> 
        >>> # 使用自定义模型
        >>> from ragas.llms import llm_factory
        >>> from ragas.embeddings import embedding_factory
        >>> 
        >>> eval_llm = llm_factory(
        ...     model="deepseek-ai/deepseek-v3.1",
        ...     base_url="https://integrate.api.nvidia.com/v1",
        ...     api_key="your-api-key"
        ... )
        >>> eval_emb = embedding_factory(
        ...     model="embedding-3",
        ...     base_url="https://open.bigmodel.cn/api/paas/v4/",
        ...     api_key="your-api-key"
        ... )
        >>> 
        >>> result = evaluate(
        ...     dataset=exp_ds,
        ...     metrics=[faithfulness, answer_relevancy],
        ...     llm=eval_llm,
        ...     embeddings=eval_emb,
        ...     name="my_rag_system"
        ... )
        >>> print(result.summary())
    """
    # 验证输入
    if len(dataset) == 0:
        raise ValueError("Dataset cannot be empty")

    # 处理指标参数
    if isinstance(metrics, str):
        if metrics not in METRIC_GROUPS:
            raise ValueError(
                f"Unknown metric group: {metrics}. "
                f"Available groups: {list(METRIC_GROUPS.keys())}"
            )
        metrics = METRIC_GROUPS[metrics]
    elif not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("Metrics must be a non-empty list or a valid group name")

    # 设置默认名称
    if name is None:
        name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(
        f"Starting evaluation '{name}' with {len(metrics)} metrics on {len(dataset)} samples"
    )

    try:
        # 调用RAGAS进行评测
        if show_progress:
            logger.info("Running RAGAS evaluation...")
            if llm is not None:
                logger.info(f"Using custom LLM: {type(llm).__name__}")
            if embeddings is not None:
                logger.info(f"Using custom embeddings: {type(embeddings).__name__}")

        # 直接透传所有参数给RAGAS
        ragas_result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            experiment_name=name,
            callbacks=callbacks,
            run_config=run_config,
            token_usage_parser=token_usage_parser,
            raise_exceptions=raise_exceptions,
            column_map=column_map,
            show_progress=show_progress,
            batch_size=batch_size,
            return_executor=return_executor,
            allow_nest_asyncio=allow_nest_asyncio,
            **kwargs,
        )
        
        # 如果返回Executor，直接返回
        if return_executor:
            return ragas_result

        # 转换为我们的结果格式
        metric_results = {}
        for metric in metrics:
            metric_name = metric.name
            if metric_name in ragas_result:
                # RAGAS返回的是DataFrame，需要提取分数
                scores_series = ragas_result[metric_name]
                
                # 转换为列表
                if hasattr(scores_series, 'tolist'):
                    scores = scores_series.tolist()
                elif isinstance(scores_series, list):
                    scores = scores_series
                else:
                    # 如果是单个值，转换为列表
                    scores = [float(scores_series)] * len(dataset)
                
                # 计算平均分
                avg_score = sum(scores) / len(scores) if scores else 0.0

                metric_results[metric_name] = MetricResult(
                    name=metric_name,
                    score=avg_score,
                    scores=scores,
                    metadata={"metric_type": type(metric).__name__},
                )
            else:
                logger.warning(f"Metric '{metric_name}' not found in RAGAS results")

        if not metric_results:
            raise EvaluationError("No valid metric results obtained")

        result = EvaluationResult(
            name=name,
            metrics=metric_results,
            dataset_size=len(dataset),
            metadata={
                "metric_names": [m.name for m in metrics],
            },
        )

        logger.info(f"Evaluation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise EvaluationError(f"Evaluation failed: {e}") from e


def evaluate_retrieval(
    dataset: EvaluationDataset,
    name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    callbacks: Optional[Callbacks] = None,
    run_config: Optional[RunConfig] = None,
    show_progress: bool = True,
    **kwargs,
) -> Union[EvaluationResult, Executor]:
    """评测检索阶段

    使用检索相关的指标评测RAG系统的检索性能。

    Args:
        dataset: RAGAS EvaluationDataset实例
        name: 评测名称
        llm: 用于评测的LLM模型（可选）
        embeddings: 用于评测的Embedding模型（可选）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置（可选）
        show_progress: 是否显示进度条
        **kwargs: 传递给evaluate()的其他参数

    Returns:
        EvaluationResult实例或Executor实例

    Example:
        >>> result = evaluate_retrieval(exp_ds, name="retrieval_test")
        >>> print(f"Context Recall: {result.get_score('context_recall')}")
    """
    if name is None:
        name = f"retrieval_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return evaluate(
        dataset=dataset,
        metrics="retrieval",
        name=name,
        llm=llm,
        embeddings=embeddings,
        callbacks=callbacks,
        run_config=run_config,
        show_progress=show_progress,
        **kwargs,
    )


def evaluate_generation(
    dataset: EvaluationDataset,
    name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    callbacks: Optional[Callbacks] = None,
    run_config: Optional[RunConfig] = None,
    show_progress: bool = True,
    **kwargs,
) -> Union[EvaluationResult, Executor]:
    """评测生成阶段

    使用生成相关的指标评测RAG系统的生成性能。

    Args:
        dataset: RAGAS EvaluationDataset实例
        name: 评测名称
        llm: 用于评测的LLM模型（可选）
        embeddings: 用于评测的Embedding模型（可选）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置（可选）
        show_progress: 是否显示进度条
        **kwargs: 传递给evaluate()的其他参数

    Returns:
        EvaluationResult实例或Executor实例

    Example:
        >>> result = evaluate_generation(exp_ds, name="generation_test")
        >>> print(f"Faithfulness: {result.get_score('faithfulness')}")
    """
    if name is None:
        name = f"generation_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return evaluate(
        dataset=dataset,
        metrics="generation",
        name=name,
        llm=llm,
        embeddings=embeddings,
        callbacks=callbacks,
        run_config=run_config,
        show_progress=show_progress,
        **kwargs,
    )


def evaluate_e2e(
    dataset: EvaluationDataset,
    name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    callbacks: Optional[Callbacks] = None,
    run_config: Optional[RunConfig] = None,
    show_progress: bool = True,
    **kwargs,
) -> Union[EvaluationResult, Executor]:
    """端到端评测

    使用端到端指标评测RAG系统的整体性能。

    Args:
        dataset: RAGAS EvaluationDataset实例
        name: 评测名称
        llm: 用于评测的LLM模型（可选）
        embeddings: 用于评测的Embedding模型（可选）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置（可选）
        show_progress: 是否显示进度条
        **kwargs: 传递给evaluate()的其他参数

    Returns:
        EvaluationResult实例或Executor实例

    Example:
        >>> result = evaluate_e2e(exp_ds, name="e2e_test")
        >>> print(result.summary())
    """
    if name is None:
        name = f"e2e_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return evaluate(
        dataset=dataset,
        metrics="e2e",
        name=name,
        llm=llm,
        embeddings=embeddings,
        callbacks=callbacks,
        run_config=run_config,
        show_progress=show_progress,
        **kwargs,
    )


def get_available_metrics() -> Dict[str, List[str]]:
    """获取可用的指标和指标组合

    Returns:
        包含指标组合的字典
    """
    return {
        group_name: [metric.name for metric in metrics]
        for group_name, metrics in METRIC_GROUPS.items()
    }


def validate_dataset(dataset: EvaluationDataset) -> Dict[str, Any]:
    """验证数据集是否适合评测

    Args:
        dataset: RAGAS EvaluationDataset实例

    Returns:
        验证结果字典
    """
    if len(dataset) == 0:
        return {
            "is_valid": False,
            "errors": ["Dataset is empty"],
            "warnings": [],
        }

    errors = []
    warnings = []

    # 检查必需字段
    for i, sample in enumerate(dataset):
        if not hasattr(sample, "user_input") or not sample.user_input:
            errors.append(f"Sample {i}: missing or empty user_input")

        if not hasattr(sample, "response") or not sample.response:
            warnings.append(f"Sample {i}: missing or empty response")

        if not hasattr(sample, "retrieved_contexts") or not sample.retrieved_contexts:
            warnings.append(f"Sample {i}: missing or empty retrieved_contexts")

        if not hasattr(sample, "reference") or not sample.reference:
            warnings.append(f"Sample {i}: missing or empty reference")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "sample_count": len(dataset),
    }
