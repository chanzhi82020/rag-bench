# Prepare模块简化说明

## 简化原则

**核心原则**: 不要为RAGAS已有的功能创建薄包装层

## 删除的冗余封装

### 1. `save_experiment_dataset()` ❌ 删除

**原因**: 只是简单封装了 `dataset.to_jsonl()`

**之前**:
```python
from rag_benchmark.prepare import save_experiment_dataset
save_experiment_dataset(exp_ds, "output.jsonl", overwrite=True)
```

**之后**:
```python
# 直接使用RAGAS API
exp_ds.to_jsonl("output.jsonl")
```

### 2. `load_experiment_dataset()` ❌ 删除

**原因**: 只是简单封装了 `EvaluationDataset.from_jsonl()`

**之前**:
```python
from rag_benchmark.prepare import load_experiment_dataset
loaded = load_experiment_dataset("output.jsonl")
```

**之后**:
```python
# 直接使用RAGAS API
from ragas.dataset_schema import EvaluationDataset
loaded = EvaluationDataset.from_jsonl("output.jsonl")
```

## 保留的核心功能

### ✅ `prepare_experiment_dataset()` - 保留

**原因**: 这是真正的业务逻辑，提供了：
- Golden Dataset → Experiment Dataset 的转换
- RAG系统集成
- 进度显示
- 错误处理
- 批量处理
- 失败记录追踪

这不是简单封装，而是核心功能。

### ✅ `RAGInterface` - 保留

**原因**: 提供统一的RAG系统接口，这是框架的核心抽象。

### ✅ `DummyRAG` / `SimpleRAG` - 保留

**原因**: 示例实现，帮助用户理解如何集成RAG系统。

## 简化后的API

### 完整使用流程

```python
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, DummyRAG

# 1. 加载Golden Dataset
golden_ds = GoldenDataset("hotpotqa")

# 2. 创建RAG系统
rag = DummyRAG()

# 3. 准备实验数据（核心功能）
exp_ds = prepare_experiment_dataset(golden_ds, rag)

# 4. 保存（直接使用RAGAS）
exp_ds.to_jsonl("output.jsonl")

# 5. 加载（直接使用RAGAS）
loaded = EvaluationDataset.from_jsonl("output.jsonl")

# 6. 评测（直接使用RAGAS）
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
results = evaluate(loaded, metrics=[faithfulness, answer_relevance])
```

## 优势

1. **更清晰**: 用户直接使用RAGAS API，无需学习额外的封装
2. **更简洁**: 减少了不必要的代码层
3. **更一致**: 与RAGAS生态保持一致
4. **更易维护**: 更少的代码意味着更少的维护负担

## 模块职责

Prepare模块现在只专注于它的核心职责：

**核心职责**: 将Golden Dataset转换为Experiment Dataset

**非职责**: 
- ❌ 数据保存/加载（由RAGAS处理）
- ❌ 数据验证（由RAGAS处理）
- ❌ 数据格式转换（已经是RAGAS格式）

## 代码统计

**简化前**:
- `prepare.py`: ~320 lines
- 导出函数: 3个（prepare, save, load）

**简化后**:
- `prepare.py`: ~250 lines
- 导出函数: 1个（prepare）
- 减少: ~70 lines

## 总结

通过删除不必要的封装，prepare模块变得更加简洁和专注。用户可以直接使用RAGAS的强大功能，而我们只提供真正有价值的核心功能：**Golden Dataset到Experiment Dataset的转换**。

这符合"不要重复造轮子"的原则，让代码更易理解和维护。
