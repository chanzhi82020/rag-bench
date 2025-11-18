# Implementation Tasks

## 1. 创建便捷评测函数
- [ ] 1.1 实现 `evaluate_end_to_end()` 函数
  - 封装 `ragas.evaluate()` 用于端到端评测
  - 支持常用指标组合
  - 返回RAGAS的Result对象
  - _Requirements: 端到端评测_

- [ ] 1.2 实现 `evaluate_retrieval()` 函数
  - 只评测检索阶段指标
  - 使用context_recall, context_precision等
  - _Requirements: 检索阶段评测_

- [ ] 1.3 实现 `evaluate_generation()` 函数
  - 只评测生成阶段指标
  - 使用faithfulness, answer_relevance等
  - _Requirements: 生成阶段评测_

## 2. 实现扩展检索指标
- [ ] 2.1 实现 `recall_at_k()` 指标
  - 计算top-k召回率
  - 支持不同的k值
  - _Requirements: 检索指标扩展_

- [ ] 2.2 实现 `precision_at_k()` 指标
  - 计算top-k精确率
  - 支持不同的k值
  - _Requirements: 检索指标扩展_

- [ ] 2.3 实现 `mrr()` 指标
  - 计算平均倒数排名
  - _Requirements: 检索指标扩展_

- [ ] 2.4 实现 `ndcg_at_k()` 指标
  - 计算归一化折损累积增益
  - 支持不同的k值
  - _Requirements: 检索指标扩展_

## 3. 创建示例代码
- [ ] 3.1 创建 `examples/evaluate_end_to_end.py`
  - 演示端到端评测
  - 展示如何使用RAGAS指标
  - _Requirements: 使用示例_

- [ ] 3.2 创建 `examples/evaluate_retrieval.py`
  - 演示检索阶段评测
  - 展示扩展指标的使用
  - _Requirements: 使用示例_

- [ ] 3.3 创建 `examples/evaluate_generation.py`
  - 演示生成阶段评测
  - _Requirements: 使用示例_

## 4. 编写文档
- [ ] 4.1 创建 `src/rag_benchmark/evaluate/README.md`
  - 说明evaluate模块的功能
  - 提供快速开始指南
  - 列出所有可用指标
  - 提供API文档
  - _Requirements: 模块文档_

- [ ] 4.2 更新项目主README
  - 添加evaluate模块介绍
  - 更新完整的使用流程
  - _Requirements: 项目文档_

## 5. 单元测试
- [ ]* 5.1 测试便捷评测函数
  - 测试evaluate_end_to_end
  - 测试evaluate_retrieval
  - 测试evaluate_generation
  - _Requirements: 功能测试_

- [ ]* 5.2 测试扩展检索指标
  - 测试recall@k计算
  - 测试precision@k计算
  - 测试MRR计算
  - 测试NDCG@k计算
  - _Requirements: 指标测试_

## 6. 最终检查
- [ ] 6.1 代码质量检查
  - 运行black格式化
  - 运行isort排序
  - _Requirements: 代码质量_

- [ ] 6.2 验证示例可运行
  - 运行所有示例脚本
  - 确保输出正确
  - _Requirements: 示例验证_
