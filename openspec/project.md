# Project Context

## Purpose
构建一个RAG（Retrieval-Augmented Generation）Benchmark框架，用于评测RAG系统的性能。该框架参考ARES、BIER等benchmark框架，重点集成RAGAS评估开源框架，支持端到端和分阶段的RAG评测。

主要功能包括：
- 维护golden数据集（包含问题、答案、上下文和语料库）
- 准备实验数据集（填充检索到的上下文和生成的响应）
- 执行评测（基于RAGAS和扩展的评估指标）
- 提供结果对比分析能力
- 内置baseline RAG系统用于快速基准测试

## Tech Stack
- **Python**: >=3.11（主要开发语言）
- **datasets**: >=4.4.1（用于加载和处理数据集，如HotpotQA）
- **ragas**: >=0.3.9（RAG评估框架，提供核心评测指标）
- **包管理工具**: uv（使用清华大学镜像源）
- **数据集格式**: JSONL（用于golden dataset和experiment dataset）

## Project Conventions

### Code Style
- 使用Python编写，遵循PEP 8代码规范
- 代码文件使用中文注释，必要时保留英文专业名词
- 单个文件不超过1000行，超过则进行功能拆分
- 保持代码简单直观，避免过度设计

### Architecture Patterns
- **模块化设计**: 按功能划分模块（datasets、prepare、evaluate、analysis、examples）
- **两阶段流程**: 
  1. Prepare阶段：准备实验数据集
  2. Evaluate阶段：执行评测
- **数据驱动**: 所有评测基于标准化的数据格式
- **可扩展性**: 支持自定义评测指标和RAG系统接入

### Testing Strategy
- 所有变更必须通过lint检查
- 使用pytest进行单元测试
- 提供完整的示例和文档
- 支持集成测试验证端到端流程

### Git Workflow
- 主分支: master
- 提交信息使用中文，清晰描述变更内容
- 功能开发使用feature分支
- 重要变更需要通过OpenSpec流程进行规划

## Domain Context

### RAG评测核心概念
- **Golden Dataset**: 包含user_input（问题）、reference（正确答案）、reference_contexts（相关上下文）、corpus（文档语料库）
- **Experiment Dataset**: Golden Dataset + retrieved_contexts（检索到的上下文） + response（生成的响应）
- **评测维度**:
  - 检索阶段：context_recall、context_precision、recall@k、precision@k、MRR、NDCG@k
  - 生成阶段：faithfulness、answer_relevance、grounding、coherence、correctness
  - 端到端：综合上述所有指标

### 数据质量标准
- 小型数据集：100-500条
- 中型数据集：1000-5000条
- 大型数据集：10000+条
- QA质量：问题清晰无歧义，答案准确完整
- 上下文质量：与问题高度相关，信息密度适中

## Important Constraints
- 必须使用固定的数据格式（JSONL）
- Golden Dataset由框架维护，不支持从文档自动构建
- 专注于评测功能，不包含RAG系统本身的实现
- MVP版本不包含prompt-based data generation
- 暂不提供Web UI（未来版本考虑）

## External Dependencies
- **Hugging Face Datasets**: 用于加载HotpotQA、NQ等公开数据集
- **RAGAS**: 核心评测框架，提供faithfulness、context_relevance等指标
- **向量数据库（可选）**: baseline RAG需要FAISS或Chroma作为检索器
- **开源LLM（可选）**: baseline RAG生成器，如Qwen 1.5等小型模型
