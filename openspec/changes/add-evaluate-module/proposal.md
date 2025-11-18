# Proposal: Add Evaluate Module for RAG Evaluation

## Why

Prepare模块已完成，可以生成Experiment Dataset。但缺少评测能力，用户无法：
1. 评估RAG系统的检索质量（recall, precision, MRR, NDCG）
2. 评估RAG系统的生成质量（faithfulness, answer_relevance, correctness）
3. 进行端到端的综合评测
4. 获取结构化的评测结果

Evaluate模块是RAG Benchmark框架的核心价值所在，是MVP的关键组件。

## What Changes

### 核心功能

1. **直接使用RAGAS评测** - 不重复造轮子
   - 使用 `ragas.evaluate()` 进行评测
   - 使用 `ragas.metrics` 中的所有指标
   - 直接返回RAGAS的评测结果

2. **提供便捷的评测函数** - 简化常见场景
   - `evaluate_retrieval()`: 只评测检索阶段
   - `evaluate_generation()`: 只评测生成阶段
   - `evaluate_end_to_end()`: 端到端评测

3. **扩展检索指标** - RAGAS未提供的指标
   - recall@k, precision@k
   - MRR (Mean Reciprocal Rank)
   - NDCG@k (Normalized Discounted Cumulative Gain)

### 设计原则

**最小化封装**: 
- ❌ 不创建自定义的Result类（直接使用RAGAS的Result）
- ❌ 不重新实现RAGAS已有的指标
- ✅ 只提供便捷函数简化常见使用场景
- ✅ 只实现RAGAS缺失的检索指标

### 架构

```
src/rag_benchmark/evaluate/
├── __init__.py              # 模块导出
├── evaluate.py              # 便捷评测函数
├── metrics_retrieval.py     # 扩展的检索指标
└── README.md                # 模块文档
```

**不创建**:
- ❌ `evaluator.py` - 不需要，直接用 `ragas.evaluate()`
- ❌ `results.py` - 不需要，直接用RAGAS的Result
- ❌ `metrics_ragas.py` - 不需要，直接用 `ragas.metrics`
- ❌ `metrics_generation.py` - 不需要，RAGAS已提供

## Impact

### Affected Specs
- 新增 `evaluate` capability

### Affected Code
- 新增 `src/rag_benchmark/evaluate/` 目录
- 新增示例 `examples/evaluate_*.py`
- 更新主README

### Dependencies
- 依赖 `ragas` 包（已安装）
- 依赖 `prepare` 模块（已完成）
- 需要LLM API（用于某些RAGAS指标）

### Benefits
- 完成RAG评测的核心功能
- 用户可以评估RAG系统性能
- 支持多种评测场景
- 直接使用RAGAS生态

### Non-Goals
- ❌ 不实现自定义的评测框架
- ❌ 不重新实现RAGAS已有的指标
- ❌ 不创建复杂的结果分析（留给analysis模块）
