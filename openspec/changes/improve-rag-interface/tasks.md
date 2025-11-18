# Implementation Tasks

## 1. 定义新的数据结构
- [ ] 1.1 创建 `RetrievalResult` 数据类
  - 包含contexts、context_ids、scores、metadata字段
  - 提供便捷方法（如to_dict()）
  - 添加验证逻辑
  - _Requirements: 检索结果扩展_

- [ ] 1.2 创建 `GenerationResult` 数据类
  - 包含response、multi_responses、confidence、metadata字段
  - 提供便捷方法
  - 添加验证逻辑
  - _Requirements: 生成结果扩展_

## 2. 更新RAGInterface接口
- [ ] 2.1 添加 `retrieve_with_metadata()` 方法
  - 定义新的检索接口
  - 提供默认实现（调用旧接口）
  - 添加详细文档说明
  - _Requirements: 扩展检索接口_

- [ ] 2.2 添加 `generate_with_metadata()` 方法
  - 定义新的生成接口
  - 提供默认实现（调用旧接口）
  - 添加详细文档说明
  - _Requirements: 扩展生成接口_

- [ ] 2.3 更新旧接口的默认实现
  - 让旧接口调用新接口
  - 确保向后兼容
  - 添加兼容性测试
  - _Requirements: 向后兼容_

- [ ] 2.4 添加批量处理的新接口
  - `batch_retrieve_with_metadata()`
  - `batch_generate_with_metadata()`
  - 提供默认实现
  - _Requirements: 批量处理扩展_

## 3. 更新prepare模块
- [ ] 3.1 更新 `_process_single_record()` 函数
  - 使用新的`retrieve_with_metadata()`
  - 使用新的`generate_with_metadata()`
  - 提取并使用额外元数据
  - _Requirements: prepare模块适配_

- [ ] 3.2 更新 `_batch_process()` 函数
  - 适配新的批量接口
  - 处理额外元数据
  - 保持错误处理逻辑
  - _Requirements: 批量处理适配_

- [ ] 3.3 更新SingleTurnSample创建逻辑
  - 添加`retrieved_context_ids`字段
  - 添加`multi_responses`字段
  - 确保RAGAS兼容性
  - _Requirements: RAGAS集成_

## 4. 更新示例实现
- [ ] 4.1 更新 `DummyRAG` 实现
  - 实现新的`retrieve_with_metadata()`
  - 实现新的`generate_with_metadata()`
  - 生成模拟的元数据
  - _Requirements: 示例更新_

- [ ] 4.2 更新 `SimpleRAG` 实现
  - 实现新接口
  - 返回实际的相似度分数
  - 返回上下文ID
  - _Requirements: 示例更新_

## 5. 创建新的示例代码
- [ ] 5.1 创建 `examples/advanced_rag_integration.py`
  - 演示如何使用新接口
  - 展示返回额外元数据
  - 展示多候选答案
  - _Requirements: 高级示例_

- [ ] 5.2 创建迁移示例
  - 展示从旧接口迁移到新接口
  - 对比两种实现方式
  - 说明迁移的好处
  - _Requirements: 迁移指南_

## 6. 更新文档
- [ ] 6.1 更新 `src/rag_benchmark/prepare/README.md`
  - 添加新接口说明
  - 添加迁移指南
  - 更新API参考
  - _Requirements: 文档更新_

- [ ] 6.2 创建迁移指南文档
  - 说明为什么要迁移
  - 提供迁移步骤
  - 列出常见问题
  - _Requirements: 迁移文档_

- [ ] 6.3 更新接口文档字符串
  - 详细说明新旧接口的关系
  - 提供使用示例
  - 说明最佳实践
  - _Requirements: API文档_

## 7. 兼容性测试
- [ ]* 7.1 测试旧接口继续工作
  - 创建使用旧接口的测试RAG
  - 验证prepare流程正常
  - 确保无破坏性变更
  - _Requirements: 向后兼容测试_

- [ ]* 7.2 测试新接口功能
  - 创建使用新接口的测试RAG
  - 验证额外元数据正确传递
  - 验证multi_responses正确处理
  - _Requirements: 新接口测试_

- [ ]* 7.3 测试混合使用场景
  - 部分实现新接口，部分使用旧接口
  - 验证默认实现正确工作
  - 验证互相调用的逻辑
  - _Requirements: 混合场景测试_

## 8. 性能验证
- [ ]* 8.1 验证新接口无性能损失
  - 对比新旧接口的性能
  - 确保额外开销可忽略
  - _Requirements: 性能测试_

## 9. 最终检查
- [ ] 9.1 代码质量检查
  - 运行 `black` 格式化
  - 运行 `isort` 整理导入
  - 运行 `mypy` 类型检查
  - _Requirements: 代码质量_

- [ ] 9.2 确保所有测试通过
  - 运行所有单元测试
  - 运行所有集成测试
  - 运行兼容性测试
  - _Requirements: 测试通过_

- [ ] 9.3 验证示例代码可运行
  - 运行所有示例脚本
  - 验证新旧接口都能工作
  - 确保文档与代码一致
  - _Requirements: 示例验证_
