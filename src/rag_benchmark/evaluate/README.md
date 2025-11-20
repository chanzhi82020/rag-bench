from rag_benchmark.evaluate import evaluate_e2efrom rag_benchmark.evaluate import evaluate_e2e

# RAG Benchmark Evaluate Module

评测模块（Evaluate Module）是RAG评测流程的核心阶段，负责使用RAGAS指标和传统IR指标对RAG系统进行全面评测。

## 功能概述

- ✅ **端到端评测**: 使用RAGAS的完整指标集评测RAG系统
- ✅ **分阶段评测**: 单独评测检索阶段或生成阶段
- ✅ **传统IR指标**: recall@k, precision@k, MRR, NDCG, MAP
- ✅ **结果管理**: 结构化的评测结果，支持保存、加载和对比
- ✅ **自定义模型**: 支持使用自定义LLM和Embedding模型
- ✅ **批量评测**: 支持大规模数据集的高效评测

## 快速开始

### 1. 端到端评测

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import DummyRAG, prepare_experiment_dataset
from rag_benchmark.evaluate import evaluate, evaluate_e2e
from ragas.metrics import faithfulness, answer_relevancy

# 1. 准备评测数据
golden_ds = GoldenDataset("xquad", subset="zh")
rag = DummyRAG()
exp_ds = prepare_experiment_dataset(golden_ds, rag)

# 2. 进行评测
result = evaluate_e2e(
   dataset=exp_ds,
   experiment_name="my_rag_system_v1"
)

# 3. 查看结果
print(result.to_pandas())

# 4. 保存结果
result.to_pandas().to_csv("results/evaluation.csv")
```

### 2. 检索阶段评测

```python
from rag_benchmark.evaluate import evaluate_retrieval

# 使用RAGAS检索指标
result = evaluate_retrieval(exp_ds, experiment_name="retrieval_test")
print(f"Context Recall: {np.mean(result['context_recall']):.4f}")
print(f"Context Precision: {np.mean(result['context_precision']):.4f}")
```

### 3. 使用自定义模型

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 配置自定义评测模型
eval_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="your-api-key",
    temperature=0.0
)

eval_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-api-key"
)

# 使用自定义模型评测
result = evaluate_e2e(
    dataset=exp_ds,
    llm=eval_llm,
    embeddings=eval_embeddings,
    experiment_name="custom_model_evaluation"
)
```

## 支持的指标

### RAGAS指标（需要LLM）

| 指标 | 类型 | 说明 | 范围 |
|------|------|------|------|
| faithfulness | 生成 | 答案对上下文的忠实度 | 0-1 |
| answer_relevancy | 端到端 | 答案与问题的相关性 | 0-1 |
| answer_correctness | 生成 | 答案的正确性 | 0-1 |
| context_recall | 检索 | 上下文召回率 | 0-1 |
| context_precision | 检索 | 上下文精确率 | 0-1 |

### 传统IR指标（不需要LLM）

| 指标 | 说明 | 需要数据 |
|------|------|----------|
| recall@k | 前k个结果中相关文档的召回率 | context_ids |
| precision@k | 前k个结果中相关文档的精确率 | context_ids |
| f1@k | Recall和Precision的调和平均 | context_ids |
| ndcg@k | 归一化折损累积增益 | context_ids |
| mrr | 平均倒数排名 | context_ids |
| map | 平均精确率 | context_ids |

### 预定义指标组合

```python
METRIC_GROUPS = {
    "retrieval": [context_recall, context_precision],
    "generation": [faithfulness, answer_correctness],
    "e2e": [faithfulness, answer_relevancy, context_precision, context_recall],
    "all": [所有可用指标]
}
```

## 评测结果

### EvaluationResult
复用RAGAS的EvaluationResult

## 示例代码

完整的示例代码请查看：

- `examples/evaluate_rag_system.py` - 完整端到端评测流程
- `examples/evaluate_retrieval.py` - 检索阶段评测（含传统IR指标）
- `examples/evaluate_generation.py` - 生成阶段评测
- `examples/evaluate_with_custom_models.py` - 使用自定义模型

## API参考

### 核心函数

#### evaluate_retrieval()

专门用于检索阶段评测的便捷函数。

#### evaluate_generation()

专门用于生成阶段评测的便捷函数。

#### evaluate_e2e()

专门用于端到端评测的便捷函数。

### 传统IR指标函数

#### 单个指标函数

- `recall_at_k()` - 计算Recall@K
- `precision_at_k()` - 计算Precision@K
- `f1_at_k()` - 计算F1@K
- `mean_reciprocal_rank()` - 计算MRR
- `ndcg_at_k()` - 计算NDCG@K
- `average_precision()` - 计算AP

## 注意事项

1. **API密钥**: RAGAS指标需要LLM，因此需要配置API密钥
2. **数据要求**: 
   - 检索评测需要 `retrieved_contexts` 和 `reference_contexts`
   - 生成评测需要 `response` 和 `reference`
   - 传统IR指标需要 `retrieved_context_ids` 和 `reference_context_ids`
3. **评测时间**: LLM评测可能需要较长时间，建议先用小数据集测试
4. **成本**: 使用API会产生费用，注意控制评测规模

## 最佳实践

1. **先小后大**: 先用小数据集（5-10条）测试，确认配置正确后再扩大规模
2. **保存结果**: 及时保存评测结果，避免重复评测
3. **对比分析**: 使用相同的评测LLM对比不同RAG系统
4. **多维度评测**: 结合RAGAS指标和传统IR指标，全面评估系统性能
5. **关注趋势**: 定期评测，跟踪系统性能变化

## 故障排除

### 问题：评测失败，提示API错误

**解决方案**:
1. 检查API密钥是否正确
2. 确认网络连接正常
3. 检查API配额是否充足
4. 尝试使用其他LLM服务

### 问题：评测速度很慢

**解决方案**:
1. 减少数据集大小
2. 使用更快的LLM模型
3. 调整 `batch_size` 参数
4. 考虑使用本地LLM

### 问题：某些指标返回NaN

**解决方案**:
1. 检查数据集是否包含必需字段
2. 确认LLM能够正常访问
3. 查看日志了解具体错误信息

## 与其他模块集成

### 与prepare模块集成

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, DummyRAG
from rag_benchmark.evaluate import evaluate, evaluate_e2e

# 完整流程
golden_ds = GoldenDataset("hotpotqa")
rag = DummyRAG()
exp_ds = prepare_experiment_dataset(golden_ds, rag)
result = evaluate_e2e(exp_ds)
print(result)
```

## 下一步

完成评测后，可以：

1. 使用`analysis`模块进行深入分析（计划中）
2. 对比不同RAG系统的性能
3. 识别系统的优势和不足
4. 指导RAG系统的优化方向
