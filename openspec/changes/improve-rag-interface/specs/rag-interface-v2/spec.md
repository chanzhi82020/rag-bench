# RAG Interface V2 Specification

## MODIFIED Requirements

### Requirement: 扩展检索接口
RAG接口应当支持返回检索的额外元数据，而不仅仅是上下文文本。

#### Scenario: 返回上下文ID
- **GIVEN** 用户实现了一个RAG系统
- **WHEN** 系统检索相关上下文
- **THEN** 系统应当能够返回每个上下文的唯一ID
- **AND** 这些ID应当可以用于RAGAS评测

#### Scenario: 返回相似度分数
- **GIVEN** 用户实现了一个RAG系统
- **WHEN** 系统检索相关上下文
- **THEN** 系统应当能够返回每个上下文的相似度分数
- **AND** 分数应当可以用于分析和调试

#### Scenario: 返回检索元数据
- **GIVEN** 用户实现了一个RAG系统
- **WHEN** 系统检索相关上下文
- **THEN** 系统应当能够返回额外的元数据
- **AND** 元数据可以包含来源、时间戳等信息

### Requirement: 扩展生成接口
RAG接口应当支持返回生成的额外信息，包括多个候选答案。

#### Scenario: 返回多个候选答案
- **GIVEN** 用户实现了一个RAG系统
- **WHEN** 系统生成答案
- **THEN** 系统应当能够返回多个候选答案
- **AND** 这些候选答案应当可以用于评测和分析

#### Scenario: 返回生成置信度
- **GIVEN** 用户实现了一个RAG系统
- **WHEN** 系统生成答案
- **THEN** 系统应当能够返回答案的置信度
- **AND** 置信度应当在0-1之间

#### Scenario: 返回生成元数据
- **GIVEN** 用户实现了一个RAG系统
- **WHEN** 系统生成答案
- **THEN** 系统应当能够返回额外的元数据
- **AND** 元数据可以包含token数、耗时等信息

### Requirement: 向后兼容性
新接口必须保持与现有代码的完全兼容。

#### Scenario: 旧代码继续工作
- **GIVEN** 用户已经实现了旧版RAGInterface
- **WHEN** 系统升级到新版本
- **THEN** 旧代码应当无需修改继续工作
- **AND** 所有功能应当保持正常

#### Scenario: 旧接口调用新接口
- **GIVEN** 用户只实现了新的`retrieve_with_metadata()`
- **WHEN** 系统调用旧的`retrieve()`方法
- **THEN** 系统应当自动调用新接口并提取contexts
- **AND** 行为应当与直接实现旧接口一致

#### Scenario: 新接口调用旧接口
- **GIVEN** 用户只实现了旧的`retrieve()`方法
- **WHEN** 系统调用新的`retrieve_with_metadata()`
- **THEN** 系统应当自动调用旧接口并包装结果
- **AND** 返回的RetrievalResult应当包含contexts字段

### Requirement: 渐进式迁移
用户应当能够逐步迁移到新接口，而不是一次性全部迁移。

#### Scenario: 只迁移检索接口
- **GIVEN** 用户想要返回上下文ID
- **WHEN** 用户实现新的`retrieve_with_metadata()`
- **THEN** 用户应当可以继续使用旧的`generate()`方法
- **AND** 系统应当正常工作

#### Scenario: 只迁移生成接口
- **GIVEN** 用户想要返回多个候选答案
- **WHEN** 用户实现新的`generate_with_metadata()`
- **THEN** 用户应当可以继续使用旧的`retrieve()`方法
- **AND** 系统应当正常工作

#### Scenario: 完全迁移到新接口
- **GIVEN** 用户想要使用所有新功能
- **WHEN** 用户实现所有新接口方法
- **THEN** 系统应当使用新接口
- **AND** 所有额外元数据应当正确传递

### Requirement: 数据结构设计
新的数据结构应当清晰、易用、可扩展。

#### Scenario: RetrievalResult结构
- **GIVEN** 系统定义了RetrievalResult数据类
- **WHEN** 用户创建RetrievalResult实例
- **THEN** 必需字段应当是contexts
- **AND** 可选字段应当包括context_ids、scores、metadata
- **AND** 应当提供便捷的转换方法

#### Scenario: GenerationResult结构
- **GIVEN** 系统定义了GenerationResult数据类
- **WHEN** 用户创建GenerationResult实例
- **THEN** 必需字段应当是response
- **AND** 可选字段应当包括multi_responses、confidence、metadata
- **AND** 应当提供便捷的转换方法

#### Scenario: 元数据扩展性
- **GIVEN** 用户需要添加自定义元数据
- **WHEN** 用户使用metadata字段
- **THEN** 系统应当接受任意键值对
- **AND** 不应当限制元数据的内容

### Requirement: prepare模块集成
prepare模块应当能够利用新接口的额外信息。

#### Scenario: 使用retrieved_context_ids
- **GIVEN** RAG系统返回了context_ids
- **WHEN** prepare模块处理记录
- **THEN** 系统应当将context_ids传递给SingleTurnSample
- **AND** RAGAS应当能够使用这些ID进行评测

#### Scenario: 使用multi_responses
- **GIVEN** RAG系统返回了多个候选答案
- **WHEN** prepare模块处理记录
- **THEN** 系统应当将multi_responses传递给SingleTurnSample
- **AND** RAGAS应当能够使用这些答案进行评测

#### Scenario: 处理缺失的元数据
- **GIVEN** RAG系统没有返回可选元数据
- **WHEN** prepare模块处理记录
- **THEN** 系统应当正常工作
- **AND** 可选字段应当为None

### Requirement: 批量处理扩展
批量处理接口也应当支持返回额外元数据。

#### Scenario: 批量检索返回元数据
- **GIVEN** 用户实现了批量检索
- **WHEN** 系统调用`batch_retrieve_with_metadata()`
- **THEN** 系统应当返回RetrievalResult列表
- **AND** 每个结果应当包含对应的元数据

#### Scenario: 批量生成返回元数据
- **GIVEN** 用户实现了批量生成
- **WHEN** 系统调用`batch_generate_with_metadata()`
- **THEN** 系统应当返回GenerationResult列表
- **AND** 每个结果应当包含对应的元数据

### Requirement: 文档和示例
应当提供清晰的文档和示例帮助用户理解和使用新接口。

#### Scenario: 迁移指南
- **GIVEN** 用户想要迁移到新接口
- **WHEN** 用户查看文档
- **THEN** 应当有清晰的迁移步骤
- **AND** 应当有代码示例对比

#### Scenario: 高级用法示例
- **GIVEN** 用户想要使用所有新功能
- **WHEN** 用户查看示例代码
- **THEN** 应当有完整的实现示例
- **AND** 应当展示如何返回所有元数据

#### Scenario: API参考文档
- **GIVEN** 用户需要了解接口细节
- **WHEN** 用户查看API文档
- **THEN** 应当有详细的参数说明
- **AND** 应当有返回值说明
- **AND** 应当有使用注意事项

### Requirement: 性能考虑
新接口不应当引入显著的性能开销。

#### Scenario: 最小化对象创建开销
- **GIVEN** 系统使用数据类包装结果
- **WHEN** 处理大量数据
- **THEN** 对象创建开销应当可忽略
- **AND** 不应当影响整体性能

#### Scenario: 避免不必要的数据复制
- **GIVEN** 系统在新旧接口间转换
- **WHEN** 调用接口方法
- **THEN** 应当避免不必要的数据复制
- **AND** 应当尽可能使用引用

### Requirement: 类型安全
新接口应当提供良好的类型提示，帮助IDE和类型检查器。

#### Scenario: 完整的类型注解
- **GIVEN** 系统定义了新接口
- **WHEN** 用户使用IDE
- **THEN** IDE应当能够提供准确的自动完成
- **AND** 类型检查器应当能够发现类型错误

#### Scenario: 泛型支持
- **GIVEN** 用户需要自定义元数据类型
- **WHEN** 用户使用metadata字段
- **THEN** 类型系统应当支持
- **AND** 应当保持类型安全
