"""核心prepare函数实现"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from tqdm import tqdm

from rag_benchmark.datasets import GoldenDataset, GoldenRecord

from .rag_interface import RAGInterface

logger = logging.getLogger(__name__)


class PrepareError(Exception):
    """Prepare过程中的错误"""

    pass


def prepare_experiment_dataset(
    golden_dataset: GoldenDataset,
    rag_system: RAGInterface,
    top_k: Optional[int] = None,
    batch_size: int = 1,
    show_progress: bool = True,
    skip_on_error: bool = True,
) -> EvaluationDataset:
    """准备实验数据集

    从Golden Dataset生成Experiment Dataset，通过调用RAG系统填充
    retrieved_contexts和response字段。

    Args:
        golden_dataset: Golden数据集
        rag_system: RAG系统实例
        top_k: 检索返回的top-k个结果，如果为None则使用RAG系统配置
        batch_size: 批处理大小，默认为1（逐条处理）
        show_progress: 是否显示进度条
        skip_on_error: 遇到错误时是否跳过该记录继续处理

    Returns:
        RAGAS EvaluationDataset实例

    Raises:
        PrepareError: 当skip_on_error=False且处理失败时

    Example:
        >>> from rag_benchmark.datasets import GoldenDataset
        >>> from rag_benchmark.prepare import prepare_experiment_dataset
        >>>
        >>> golden_ds = GoldenDataset("hotpotqa")
        >>> rag = MyRAGSystem()
        >>> exp_ds = prepare_experiment_dataset(golden_ds, rag)
        >>> print(len(exp_ds))
    """
    golden_records = list(golden_dataset)
    total_records = len(golden_records)

    if total_records == 0:
        logger.warning("Golden dataset is empty")
        return EvaluationDataset(samples=[])

    logger.info(f"Preparing experiment dataset from {total_records} golden records")

    experiment_records: List[SingleTurnSample] = []
    failed_records: List[Dict[str, Any]] = []

    # 创建进度条
    iterator = tqdm(
        golden_records,
        desc="Preparing experiment dataset",
        disable=not show_progress,
        unit="record",
    )

    if batch_size > 1:
        # 批量处理模式
        experiment_records, failed_records = _batch_process(
            golden_records=golden_records,
            rag_system=rag_system,
            top_k=top_k,
            batch_size=batch_size,
            skip_on_error=skip_on_error,
            iterator=iterator,
        )
    else:
        # 逐条处理模式
        for idx, golden_record in enumerate(iterator):
            try:
                exp_record = _process_single_record(
                    golden_record=golden_record,
                    rag_system=rag_system,
                    top_k=top_k,
                )
                experiment_records.append(exp_record)

            except Exception as e:
                error_info = {
                    "index": idx,
                    "user_input": golden_record.user_input,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                failed_records.append(error_info)

                if skip_on_error:
                    logger.warning(f"Failed to process record {idx}: {e}")
                    continue
                else:
                    raise PrepareError(f"Failed to process record {idx}: {e}") from e

    # 输出处理结果摘要
    success_count = len(experiment_records)
    failed_count = len(failed_records)

    logger.info(
        f"Preparation complete: {success_count} succeeded, {failed_count} failed"
    )

    if failed_records:
        logger.warning(f"Failed records summary:")
        for fail in failed_records[:5]:  # 只显示前5个
            logger.warning(f"  - Record {fail['index']}: {fail['error']}")
        if len(failed_records) > 5:
            logger.warning(f"  ... and {len(failed_records) - 5} more")

    # 返回RAGAS的EvaluationDataset
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


def _batch_process(
    golden_records: List[GoldenRecord],
    rag_system: RAGInterface,
    top_k: Optional[int],
    batch_size: int,
    skip_on_error: bool,
    iterator: tqdm,
) -> tuple[List[SingleTurnSample], List[Dict[str, Any]]]:
    """批量处理记录

    Args:
        golden_records: Golden记录列表
        rag_system: RAG系统
        top_k: 检索top-k
        batch_size: 批次大小
        skip_on_error: 是否跳过错误
        iterator: 进度条

    Returns:
        (成功的实验记录列表, 失败记录信息列表)
    """
    experiment_records: List[SingleTurnSample] = []
    failed_records: List[Dict[str, Any]] = []

    # 分批处理
    for i in range(0, len(golden_records), batch_size):
        batch = golden_records[i : i + batch_size]
        batch_queries = [r.user_input for r in batch]

        try:
            # 批量调用RAG系统
            results = rag_system.batch_retrieve_and_generate(
                queries=batch_queries,
                top_k=top_k,
            )

            # 创建RAGAS的SingleTurnSample
            for j, (golden_record, (retrieval_result, generation_result)) in enumerate(
                zip(batch, results)
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
                iterator.update(1)

        except Exception as e:
            # 批量处理失败，回退到逐条处理
            logger.warning(
                f"Batch processing failed, falling back to single processing: {e}"
            )

            for j, golden_record in enumerate(batch):
                idx = i + j
                try:
                    exp_record = _process_single_record(
                        golden_record=golden_record,
                        rag_system=rag_system,
                        top_k=top_k,
                    )
                    experiment_records.append(exp_record)

                except Exception as single_error:
                    error_info = {
                        "index": idx,
                        "user_input": golden_record.user_input,
                        "error": str(single_error),
                        "error_type": type(single_error).__name__,
                    }
                    failed_records.append(error_info)

                    if not skip_on_error:
                        raise PrepareError(
                            f"Failed to process record {idx}: {single_error}"
                        ) from single_error

                iterator.update(1)

    return experiment_records, failed_records