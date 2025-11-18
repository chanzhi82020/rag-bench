# Prepare Module Specification

## ADDED Requirements

### Requirement: Experiment Dataset Schema
系统应当提供标准化的实验数据集Schema，用于存储RAG评测所需的完整数据。

#### Scenario: 创建实验记录
- **WHEN** 用户需要创建一个实验记录
- **THEN** 系统应当提供包含所有必需字段的ExperimentRecord数据结构
- **AND** 该数据结构应当包含Golden Dataset的所有字段
- **AND** 该数据结构应当包含retrieved_contexts和response字段

#### Scenario: RAGAS格式转换
- **WHEN** 用户需要将实验数据集用于RAGAS评测
- **THEN** 系统应当提供to_ragas_dataset()方法
- **AND** 字段映射应当正确（user_input→question, reference→ground_truth等）

### Requirement: RAG系统接口
系统应当提供标准化的RAG系统接口，允许用户集成自己的RAG系统。

#### Scenario: 实现自定义RAG
- **WHEN** 用户需要集成自己的RAG系统
- **THEN** 系统应当提供RAGInterface抽象基类
- **AND** 接口应当定义retrieve()和generate()方法
- **AND** 用户只需实现这两个方法即可集成

#### Scenario: 配置RAG参数
- **WHEN** 用户需要配置RAG系统参数
- **THEN** 系统应当提供RAGConfig配置类
- **AND** 支持配置检索参数（top_k, similarity_threshold等）
- **AND** 支持配置生成参数（max_length, temperature等）

### Requirement: 实验数据集准备
系统应当提供prepare_experiment_dataset()函数，自动化生成实验数据集。

#### Scenario: 使用自定义RAG准备数据
- **WHEN** 用户提供Golden Dataset和RAG系统实例
- **THEN** 系统应当遍历所有golden records
- **AND** 对每条记录调用RAG系统的retrieve()和generate()方法
- **AND** 返回填充完整的ExperimentDataset

#### Scenario: 显示处理进度
- **WHEN** 系统正在处理大量数据
- **THEN** 系统应当使用tqdm显示进度条
- **AND** 显示当前处理的记录数和总记录数
- **AND** 显示预计剩余时间

#### Scenario: 处理RAG调用失败
- **WHEN** RAG系统调用过程中发生异常
- **THEN** 系统应当捕获异常并记录错误
- **AND** 继续处理其他记录而不中断整个流程
- **AND** 在处理完成后提供失败记录的汇总报告

### Requirement: 数据持久化
系统应当支持实验数据集的保存和加载。

#### Scenario: 保存实验数据集
- **WHEN** 用户需要保存实验数据集
- **THEN** 系统应当提供save_experiment_dataset()函数
- **AND** 数据应当保存为JSONL格式
- **AND** 每行一个JSON对象，使用UTF-8编码

#### Scenario: 加载实验数据集
- **WHEN** 用户需要加载已保存的实验数据集
- **THEN** 系统应当提供load_experiment_dataset()函数
- **AND** 验证数据格式的正确性
- **AND** 返回ExperimentDataset对象

#### Scenario: 验证数据格式
- **WHEN** 加载实验数据集时
- **THEN** 系统应当验证所有必需字段是否存在
- **AND** 验证字段类型是否正确
- **AND** 如果验证失败，提供详细的错误信息

### Requirement: 示例实现
系统应当提供示例RAG实现，帮助用户理解接口使用方式。

#### Scenario: 使用DummyRAG进行测试
- **WHEN** 用户需要快速测试prepare流程
- **THEN** 系统应当提供DummyRAG类
- **AND** DummyRAG应当返回模拟的检索结果和生成答案
- **AND** 不依赖外部服务，可立即使用

#### Scenario: 使用SimpleRAG作为参考
- **WHEN** 用户需要了解如何实现简单的RAG
- **THEN** 系统应当提供SimpleRAG类作为示例
- **AND** 使用BM25进行简单检索
- **AND** 使用模板进行简单生成
- **AND** 代码清晰易懂，便于学习

### Requirement: 批量处理支持
系统应当支持批量处理以提高大数据集的处理效率。

#### Scenario: 批量调用RAG系统
- **WHEN** RAG系统支持批量接口
- **THEN** prepare函数应当支持批量调用
- **AND** 减少网络往返次数
- **AND** 提高整体处理速度

### Requirement: 文档和示例
系统应当提供完整的文档和示例代码。

#### Scenario: 查看模块文档
- **WHEN** 用户需要了解prepare模块的使用方法
- **THEN** 系统应当提供README.md文档
- **AND** 包含快速开始指南
- **AND** 包含完整的API文档
- **AND** 包含RAG接口实现要求

#### Scenario: 运行示例代码
- **WHEN** 用户需要学习如何使用prepare模块
- **THEN** 系统应当提供examples/prepare_experiment_dataset.py
- **AND** 演示完整的数据准备流程
- **AND** 代码可以直接运行
- **AND** 输出清晰易懂
