# 001-build-dataset-module

## Summary
构建RAG Benchmark的数据集模块，支持Golden Dataset的管理、加载和验证，以及将公开数据集（HotpotQA、NQ）转换为标准格式。

## Why
数据集模块是RAG Benchmark框架的基础组件，为整个评测流程提供标准化的数据支持。当前缺少统一的数据格式和管理机制，需要实现：
1. 标准化的Golden Dataset格式定义
2. 高效的数据加载和验证机制
3. 公开数据集到标准格式的转换工具
4. 可扩展的数据集注册和管理系统

## Dates
- Created: 2025-11-18
- Status: Draft

## Related
- RAG-Bench-plan.md#L1-294
- Project MVP Phase A (数据模块)

## Capabilities
- [golden-dataset-management](specs/golden-dataset-management/spec.md)
- [dataset-conversion](specs/dataset-conversion/spec.md)

## Change Impact
- **ADDED**: 完整的数据集模块 `rag_benchmark/datasets/`
- **ADDED**: Golden Dataset标准格式和加载接口
- **ADDED**: 公开数据集转换工具
- **ADDED**: 数据质量验证机制

## Architectural Impact
新增数据集模块作为RAG Benchmark的基础组件，为后续的prepare和evaluate阶段提供标准化数据支持。