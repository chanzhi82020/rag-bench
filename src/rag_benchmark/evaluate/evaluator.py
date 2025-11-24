"""核心评估器实现"""

import logging
from typing import Optional, Union

from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from ragas import aevaluate
from ragas.dataset_schema import EvaluationDataset, EvaluationResult
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
from ragas.run_config import RunConfig

from rag_benchmark.evaluate.metrics_retrieval import RecallAtK, PrecisionAtK, F1AtK, NDCGAtK

logger = logging.getLogger(__name__)


async def evaluate_retrieval(
    dataset: EvaluationDataset,
    experiment_name: Optional[str] = None,
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
        experiment_name: 评测名称
        llm: 用于评测的LLM模型（可选）
        embeddings: 用于评测的Embedding模型（可选）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置（可选）
        show_progress: 是否显示进度条
        **kwargs: 传递给evaluate()的其他参数

    Returns:
        EvaluationResult实例或Executor实例

    """

    retrieval_metrics = [context_recall, context_precision]
    sample = dataset[0]
    if sample.reference_context_ids and sample.retrieved_context_ids:
        retrieval_metrics.extend([
            RecallAtK(),
            PrecisionAtK(),
            F1AtK(),
            NDCGAtK()
        ])

    return await aevaluate(
        dataset=dataset,
        metrics=retrieval_metrics,
        experiment_name=experiment_name,
        llm=llm,
        embeddings=embeddings,
        callbacks=callbacks,
        run_config=run_config,
        show_progress=show_progress,
        **kwargs,
    )


async def evaluate_generation(
    dataset: EvaluationDataset,
    experiment_name: Optional[str] = None,
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
        experiment_name: 评测名称
        llm: 用于评测的LLM模型（可选）
        embeddings: 用于评测的Embedding模型（可选）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置（可选）
        show_progress: 是否显示进度条
        **kwargs: 传递给evaluate()的其他参数

    Returns:
        EvaluationResult实例或Executor实例
    """

    generate_metrics = [faithfulness, answer_correctness]
    return await aevaluate(
        dataset=dataset,
        metrics=generate_metrics,
        experiment_name=experiment_name,
        llm=llm,
        embeddings=embeddings,
        callbacks=callbacks,
        run_config=run_config,
        show_progress=show_progress,
        **kwargs,
    )


async def evaluate_e2e(
    dataset: EvaluationDataset,
    experiment_name: Optional[str] = None,
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
        experiment_name: 评测名称
        llm: 用于评测的LLM模型（可选）
        embeddings: 用于评测的Embedding模型（可选）
        callbacks: Langchain回调函数（可选）
        run_config: 运行时配置（可选）
        show_progress: 是否显示进度条
        **kwargs: 传递给evaluate()的其他参数

    Returns:
        EvaluationResult实例或Executor实例

    """
    e2e_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    sample = dataset[0]
    if sample.reference_context_ids and sample.retrieved_context_ids:
        e2e_metrics.extend([
            RecallAtK(),
            PrecisionAtK(),
            F1AtK(),
            NDCGAtK()
        ])
    return await aevaluate(
        dataset=dataset,
        metrics=e2e_metrics,
        experiment_name=experiment_name,
        llm=llm,
        embeddings=embeddings,
        callbacks=callbacks,
        run_config=run_config,
        show_progress=show_progress,
        **kwargs,
    )

