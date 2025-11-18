# RAG Benchmark Evaluate Module

评测模块（Evaluate Module）是RAG评测流程的核心阶段，负责使用RAGAS指标对RAG系统进行全面评测。

## 功能概述

- **端到端评测**: 使用RAGAS的完整指标集评测RAG系统
- **分阶段评测**: 单独评测检索阶段或生成阶段
- **结果管理**: 结构化的评测结果，支持保存、加载和对比
- **指标集成**: 完整集成RAGAS的所有评测指标
- **批量评测**: 支持大规模数据集的高效评测

## 快速开始

### 基本使用

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import DummyRAG, prepare_experiment_dataset
from rag_benchmark.evaluate import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# 1. 准备评测数据（来自prepare模块）
golden_ds = GoldenDataset("xquad", subset="zh")
rag = DummyRAG()
exp_ds = prepare_experiment_dataset(golden_ds, rag)

# 2. 进行评测
result = evaluate(
    dataset=exp_ds,
    metrics=[faithfulness, answer_relevancy],
    name="my_rag_system_v1"
)

# 3. 查看结果
print(result.summary())
print(f"Faithfulness: {result.get_score('faithfulness')}")
print(f"Answer Relevancy: {result.get_score('answer_relevancy')}")

# 4. 保存结果
result.save("results/evaluation.json")
```

### 使用预定义指标组合

```python
# 检索阶段评测
result = evaluate(exp_ds, metrics="retrieval", name="retrieval_test")

# 生成阶段评测
result = evaluate(exp_ds, metrics="generation", name="generation_test")

# 端到端评测
result = evaluate(exp_ds, metrics="e2e", name="e2e_test")

# 所有指标
result = evaluate(exp_ds, metrics="all", name="full_evaluation")
```

## 核心概念

### 评测指标

本模块直接使用RAGAS的评测指标：

#### 端到端指标
- **faithfulness**: 答案对检索上下文的忠实度
- **answer_relevancy**: 答案与问题的相关性
- **answer_correctness**: 答案的正确性

#### 检索阶段指标
- **context_recall**: 上下文召回率
- **context_precision**: 上下文精确率

#### 预定义组合
```python
METRIC_GROUPS = {
    "retrieval": [context_recall, context_precision],
    "generation": [faithfulness, answer_correctness],
    "e2e": [faithfulness, answer_relevancy, context_precision, context_recall],
    "all": [所有可用指标]
}
```

### 评测结果

#### EvaluationResult
评测结果容器，包含：
- 所有指标的结果
- 评测元数据
- 统计信息
- 保存/加载功能

```python
# 查看摘要
summary = result.summary()
print(f"平均分数: {summary['average_score']}")
print(f"指标数量: {summary['metrics_count']}")

# 获取特定指标
faithfulness_score = result.get_score('faithfulness')
faithfulness_detail = result.get_metric('faithfulness')

# 对比两次评测
comparison = result1.compare_with(result2)
print(comparison)
```

#### MetricResult
单个指标的详细结果：
- 平均分数
- 每个样本的分数
- 统计信息（最小值、最大值、标准差）

## 分阶段评测

### 检索阶段评测

```python
from rag_benchmark.evaluate import evaluate_retrieval

# 专门评测检索性能
result = evaluate_retrieval(
    dataset=exp_ds,
    name="retrieval_evaluation"
)

print(f"Context Recall: {result.get_score('context_recall')}")
print(f"Context Precision: {result.get_score('context_precision')}")
```

### 生成阶段评测

```python
from rag_benchmark.evaluate import evaluate_generation

# 专门评测生成性能
result = evaluate_generation(
    dataset=exp_ds,
    name="generation_evaluation"
)

print(f"Faithfulness: {result.get_score('faithfulness')}")
print(f"Answer Correctness: {result.get_score('answer_correctness')}")
```

### 端到端评测

```python
from rag_benchmark.evaluate import evaluate_e2e

# 完整的端到端评测
result = evaluate_e2e(
    dataset=exp_ds,
    name="e2e_evaluation"
)

# 查看完整摘要
print(result.summary())
```

## 高级功能

### 自定义指标组合

```python
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# 自定义指标组合
custom_metrics = [faithfulness, answer_relevancy, context_recall]
result = evaluate(
    dataset=exp_ds,
    metrics=custom_metrics,
    name="custom_evaluation"
)
```

### 数据验证

```python
from rag_benchmark.evaluate import validate_dataset

# 验证数据集是否适合评测
validation = validate_dataset(exp_ds)
if not validation['is_valid']:
    print("数据集验证失败:")
    for error in validation['errors']:
        print(f"  - {error}")
else:
    print("数据集验证通过")
```

## 结果管理

### 保存和加载

```python
# 保存为JSON
result.save("results/evaluation.json", format="json")

# 保存为CSV（用于数据分析）
result.save("results/evaluation.csv", format="csv")

# 加载已保存的结果
from rag_benchmark.evaluate import EvaluationResult
loaded_result = EvaluationResult.load("results/evaluation.json")
```

### 结果对比

```python
# 对比两个版本的RAG系统
result_v1 = evaluate(exp_ds_v1, metrics="e2e", name="rag_v1")
result_v2 = evaluate(exp_ds_v2, metrics="e2e", name="rag_v2")

comparison = result_v1.compare_with(result_v2)
print(f"改进情况:")
for metric, data in comparison['comparison'].items():
    improvement = data['improvement']
    print(f"  {metric}: {improvement:.2f}%")
```

## 可用指标

### 查看所有可用指标

```python
from rag_benchmark.evaluate import get_available_metrics

metrics = get_available_metrics()
for group, metric_list in metrics.items():
    print(f"{group}: {metric_list}")
```

### RAGAS指标说明

| 指标 | 类型 | 说明 | 范围 |
|------|------|------|------|
| faithfulness | 生成 | 答案对上下文的忠实度 | 0-1 |
| answer_relevancy | 生成 | 答案与问题的相关性 | 0-1 |
| answer_correctness | 生成 | 答案的正确性 | 0-1 |
| context_recall | 检索 | 上下文召回率 | 0-1 |
| context_precision | 检索 | 上下文精确率 | 0-1 |

## 环境要求

### OpenAI API密钥

RAGAS需要OpenAI API来运行评测。请设置环境变量：

```bash
export OPENAI_API_KEY='your-api-key'
```

或在Python代码中设置：

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## 错误处理

### 常见错误

```python
from rag_benchmark.evaluate import EvaluationError

try:
    result = evaluate(exp_ds, metrics="e2e")
except EvaluationError as e:
    print(f"评测失败: {e}")
except ValueError as e:
    print(f"参数错误: {e}")
```

## 与其他模块集成

### 与prepare模块集成

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, DummyRAG
from rag_benchmark.evaluate import evaluate

# 完整流程
golden_ds = GoldenDataset("hotpotqa")
rag = DummyRAG()
exp_ds = prepare_experiment_dataset(golden_ds, rag)
result = evaluate(exp_ds, metrics="e2e")
print(result.summary())
```

### 为analysis模块准备数据

```python
# 评测多个RAG系统
results = []
for rag_name, rag_system in rag_systems.items():
    exp_ds = prepare_experiment_dataset(golden_ds, rag_system)
    result = evaluate(exp_ds, metrics="e2e", name=rag_name)
    results.append(result)
    
# 保存所有结果供analysis模块使用
for result in results:
    result.save(f"results/{result.name}.json")
```

## API参考

### 核心函数

#### evaluate
```python
def evaluate(
    dataset: EvaluationDataset,
    metrics: Union[List[Metric], str],
    name: Optional[str] = None,
    show_progress: bool = True,
    **kwargs
) -> EvaluationResult
```

#### evaluate_retrieval
```python
def evaluate_retrieval(
    dataset: EvaluationDataset,
    name: Optional[str] = None,
    show_progress: bool = True,
    **kwargs
) -> EvaluationResult
```

#### evaluate_generation
```python
def evaluate_generation(
    dataset: EvaluationDataset,
    name: Optional[str] = None,
    show_progress: bool = True,
    **kwargs
) -> EvaluationResult
```

#### evaluate_e2e
```python
def evaluate_e2e(
    dataset: EvaluationDataset,
    name: Optional[str] = None,
    show_progress: bool = True,
    **kwargs
) -> EvaluationResult
```

### 工具函数

#### get_available_metrics
```python
def get_available_metrics() -> Dict[str, List[str]]
```

#### validate_dataset
```python
def validate_dataset(dataset: EvaluationDataset) -> Dict[str, Any]
```

## 最佳实践

1. **数据验证**: 评测前使用`validate_dataset()`验证数据
2. **指标选择**: 根据需求选择合适的指标组合
3. **结果保存**: 及时保存评测结果，避免重复计算
4. **版本管理**: 为不同版本的RAG系统使用不同的评测名称
5. **API配额**: 注意OpenAI API的使用配额

## 故障排除

### 常见问题

**Q: 评测速度很慢怎么办？**

A: RAGAS评测需要调用OpenAI API，速度取决于网络和API响应时间。可以：
1. 减少数据集大小进行快速测试
2. 选择计算成本较低的指标

**Q: 如何解读评测结果？**

A: 
1. 查看RAGAS官方文档了解指标含义
2. 对比多个系统的结果
3. 关注分数的分布和标准差

**Q: OpenAI API密钥错误？**

A:
1. 确认环境变量设置正确
2. 检查API密钥是否有效
3. 确认API配额未超限

## 示例代码

完整的示例代码请参考：
- `examples/evaluate_rag_system.py` - 基本评测示例
- `test_evaluate_basic.py` - 基础功能测试

## 下一步

完成评测后，可以：

1. 使用`analysis`模块进行深入分析（待实现）
2. 对比不同RAG系统的性能
3. 识别系统的优势和不足
4. 指导RAG系统的优化方向
