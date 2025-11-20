"""测试Analysis模块"""

import os

import pytest
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from rag_benchmark.analysis import compare_results
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.evaluate import evaluate_e2e
from rag_benchmark.prepare import DummyRAG, SimpleRAG, prepare_experiment_dataset

# Skip all tests if no OpenAI API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OpenAI API key not set"
)


@pytest.fixture
def sample_golden_dataset():
    """创建测试用的Golden Dataset"""
    golden_ds = GoldenDataset("xquad", subset="zh")
    # 只使用前5条记录
    records = golden_ds.head(5)
    
    # 创建一个简化的数据集用于测试
    from rag_benchmark.datasets.loaders.base import BaseLoader
    from rag_benchmark.datasets.schemas.golden import GoldenRecord
    
    class TestLoader(BaseLoader):
        def __init__(self, records):
            self.records = records
        
        def load_golden_records(self):
            return iter(self.records)
        
        def load_corpus_records(self):
            return iter([])
    
    # 创建新的数据集实例
    test_ds = GoldenDataset("test", loader=TestLoader(records))
    return test_ds


@pytest.fixture
def evaluation_results(sample_golden_dataset):
    """创建两个评测结果用于对比"""
    # 准备两个RAG系统
    rag1 = DummyRAG()
    
    # 收集corpus用于SimpleRAG
    corpus = []
    for record in sample_golden_dataset:
        if record.reference_contexts:
            corpus.extend(record.reference_contexts)
    
    rag2 = SimpleRAG(corpus=corpus)
    
    # 生成实验数据集
    exp_ds1 = prepare_experiment_dataset(sample_golden_dataset, rag1)
    exp_ds2 = prepare_experiment_dataset(sample_golden_dataset, rag2)
    
    # 评测
    result1 = evaluate_e2e(exp_ds1, experiment_name="test_rag1")
    result2 = evaluate_e2e(exp_ds2, experiment_name="test_rag2")
    
    return [result1, result2]


def test_compare_results(evaluation_results):
    """测试结果对比功能"""
    comparison = compare_results(
        results=evaluation_results,
        names=["RAG1", "RAG2"]
    )
    
    # 验证基本属性
    assert len(comparison.names) == 2
    assert comparison.names == ["RAG1", "RAG2"]
    assert len(comparison.results) == 2
    assert len(comparison.metrics) > 0
    
    # 验证comparison_df
    assert comparison.comparison_df is not None
    assert len(comparison.comparison_df) == 2
    assert "name" in comparison.comparison_df.columns


def test_comparison_summary(evaluation_results):
    """测试对比摘要"""
    comparison = compare_results(
        results=evaluation_results,
        names=["RAG1", "RAG2"]
    )
    
    summary = comparison.summary()
    
    # 验证摘要格式
    assert "Model/System" in summary.columns
    assert len(summary) == 2
    assert summary["Model/System"].tolist() == ["RAG1", "RAG2"]


def test_get_best(evaluation_results):
    """测试获取最佳模型"""
    comparison = compare_results(
        results=evaluation_results,
        names=["RAG1", "RAG2"]
    )
    
    # 获取faithfulness最高的模型
    best = comparison.get_best("faithfulness", higher_is_better=True)
    
    assert "name" in best
    assert "score" in best
    assert best["name"] in ["RAG1", "RAG2"]
    assert isinstance(best["score"], (int, float))


def test_get_worst_cases(evaluation_results):
    """测试获取最差样本"""
    comparison = compare_results(
        results=evaluation_results,
        names=["RAG1", "RAG2"]
    )
    
    # 获取最差的3个样本
    worst = comparison.get_worst_cases("faithfulness", n=3, model_idx=0)
    
    assert len(worst) <= 3
    assert "faithfulness" in worst.columns
    assert "user_input" in worst.columns


def test_comparison_save(evaluation_results, tmp_path):
    """测试保存对比结果"""
    comparison = compare_results(
        results=evaluation_results,
        names=["RAG1", "RAG2"]
    )
    
    # 保存到临时文件
    save_path = tmp_path / "comparison.csv"
    comparison.save(str(save_path))
    
    # 验证文件存在
    assert save_path.exists()
    
    # 验证可以读取
    import pandas as pd
    df = pd.read_csv(save_path)
    assert len(df) == 2
    assert "name" in df.columns


def test_compare_with_custom_metrics(evaluation_results):
    """测试指定特定指标进行对比"""
    comparison = compare_results(
        results=evaluation_results,
        names=["RAG1", "RAG2"],
        metrics=["faithfulness", "answer_relevancy"]
    )
    
    # 验证只包含指定的指标
    summary = comparison.summary()
    assert "faithfulness" in summary.columns
    assert "answer_relevancy" in summary.columns


def test_compare_default_names(evaluation_results):
    """测试默认名称生成"""
    comparison = compare_results(results=evaluation_results)
    
    # 验证默认名称
    assert comparison.names == ["Model 1", "Model 2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
