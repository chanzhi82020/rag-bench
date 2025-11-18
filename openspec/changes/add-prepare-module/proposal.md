# Proposal: Add Prepare Module for Experiment Dataset Preparation

## Why

根据MVP计划，prepare模块是RAG评测流程的第一个核心阶段。当前项目已完成datasets模块（Golden Dataset管理），但缺少将Golden Dataset转换为Experiment Dataset的能力。

Experiment Dataset是评测的基础，需要在Golden Dataset基础上填充：
- `retrieved_contexts`: 由RAG系统检索得到的上下文
- `response`: 由RAG系统生成的答案

没有prepare模块，用户无法：
1. 使用自己的RAG系统生成实验数据
2. 使用框架内置的baseline RAG快速生成基准数据
3. 进行后续的评测工作

## What Changes

### 新增功能
- 创建 `ExperimentDataset` schema，兼容RAGAS的数据格式
- 实现 `prepare_experiment_dataset()` 函数，支持两种模式：
  - 用户提供RAG系统接口
  - 使用内置baseline RAG（可选）
- 提供实验数据集的保存和加载功能
- 支持批量处理和进度显示
- 提供数据验证确保实验数据集格式正确

### 架构设计
```
src/rag_benchmark/prepare/
├── __init__.py           # 模块导出
├── schema.py             # ExperimentDataset schema定义
├── prepare.py            # 核心prepare函数
├── rag_interface.py      # RAG系统抽象接口
└── baseline_rag/         # 内置baseline RAG（可选，后续实现）
    ├── __init__.py
    ├── retriever.py      # 基于FAISS的检索器
    └── generator.py      # 基于开源LLM的生成器
```

### 数据流程
```
Golden Dataset (user_input, reference, reference_contexts, corpus)
    ↓
prepare_experiment_dataset(golden_ds, rag_system)
    ↓
Experiment Dataset (+ retrieved_contexts, + response)
    ↓
保存为JSONL格式，供evaluate模块使用
```

## Impact

### Affected Specs
- 新增 `prepare` capability - 实验数据集准备功能

### Affected Code
- 新增 `src/rag_benchmark/prepare/` 目录及所有文件
- 新增 `src/rag_benchmark/examples/prepare_experiment_dataset.py` 示例
- 更新 `pyproject.toml` 添加可选依赖（如果实现baseline RAG）

### Dependencies
- 依赖已有的 `datasets` 模块
- 与 `ragas` 的数据格式保持兼容
- 为后续的 `evaluate` 模块提供输入数据

### Benefits
- 完成RAG评测流程的第一个关键阶段
- 用户可以使用自己的RAG系统生成实验数据
- 为evaluate模块的实现奠定基础
- 提供清晰的RAG系统接入接口

### Non-Goals (本次不包含)
- baseline RAG的完整实现（可作为后续独立任务）
- 分布式处理大规模数据集
- 实验数据集的自动质量评估
