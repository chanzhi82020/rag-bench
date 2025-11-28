"""核心prepare函数实现"""
import asyncio
import logging
from typing import List, Optional

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from rag_benchmark.datasets import GoldenDataset, GoldenRecord
from .rag_interface import RAGInterface

logger = logging.getLogger(__name__)


async def prepare_experiment_dataset(
        golden_dataset: GoldenDataset,
        rag_system: RAGInterface,
        top_k: Optional[int] = None,
) -> EvaluationDataset:
    """准备实验数据集

    从Golden Dataset生成Experiment Dataset，通过调用RAG系统填充
    retrieved_contexts和response字段。

    Args:
        golden_dataset: Golden数据集
        rag_system: RAG系统实例
        top_k: 检索返回的top-k个结果，如果为None则使用RAG系统配置

    Returns:
        RAGAS EvaluationDataset实例
    """
    golden_records = list(golden_dataset)
    total_records = len(golden_records)

    if total_records == 0:
        logger.warning("Golden dataset is empty")
        return EvaluationDataset(samples=[])

    logger.info(f"Preparing experiment dataset from {total_records} golden records")


    experiment_records: List[SingleTurnSample] = []

    queries = [r.user_input for r in golden_records]
    # 批量调用RAG系统

    results = await rag_system.batch_retrieve_and_generate(
        queries=queries,
        top_k=top_k,
    )

    # 创建RAGAS的SingleTurnSample
    for j, (golden_record, (retrieval_result, generation_result)) in enumerate(
            zip(golden_records, results)
    ):
        exp_record = SingleTurnSample(
            user_input=golden_record.user_input,
            reference=golden_record.reference,
            reference_contexts=golden_record.reference_contexts,
            retrieved_contexts=retrieval_result.contexts,
            response=generation_result.response,
            reference_context_ids=golden_record.reference_context_ids,
            # 使用检索结果的context_ids（如果有）
            retrieved_context_ids=retrieval_result.context_ids,
            # 使用生成结果的multi_responses（如果有）
            multi_responses=generation_result.multi_responses,
        )

        experiment_records.append(exp_record)

    return EvaluationDataset(samples=experiment_records)



def _process_single_record(
        golden_record: GoldenRecord,
        rag_system: RAGInterface,
        top_k: Optional[int] = None,
) -> SingleTurnSample:
    """处理单条记录

    Args:
        golden_record: Golden记录
        rag_system: RAG系统
        top_k: 检索top-k

    Returns:
        SingleTurnSample (RAGAS格式)
    """
    # 调用RAG系统检索和生成
    retrieval_result, generation_result = rag_system.retrieve_and_generate(
        query=golden_record.user_input,
        top_k=top_k,
    )

    # 创建RAGAS的SingleTurnSample，使用新的元数据
    exp_record = SingleTurnSample(
        user_input=golden_record.user_input,
        reference=golden_record.reference,
        reference_contexts=golden_record.reference_contexts,
        retrieved_contexts=retrieval_result.contexts,
        response=generation_result.response,
        reference_context_ids=golden_record.reference_context_ids,
        # 使用检索结果的context_ids（如果有）
        retrieved_context_ids=retrieval_result.context_ids,
        # 使用生成结果的multi_responses（如果有）
        multi_responses=generation_result.multi_responses,
    )

    return exp_record


