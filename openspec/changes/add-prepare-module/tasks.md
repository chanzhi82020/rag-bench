# Implementation Tasks

## 1. 定义数据Schema
- [x] 1.1 创建 `ExperimentRecord` dataclass
  - 包含所有Golden Dataset字段
  - 新增 `retrieved_contexts: List[str]` 字段
  - 新增 `response: str` 字段
  - 使用 `@dataclass` 装饰器，支持序列化
  - _Requirements: 实验数据集Schema定义_

- [x] 1.2 创建 `ExperimentDataset` 类
  - 封装 `List[ExperimentRecord]`
  - 提供迭代器接口
  - 提供统计信息方法（记录数、完整性检查）
  - _Requirements: 实验数据集容器_

- [x] 1.3 实现与RAGAS格式的转换
  - 提供 `to_ragas_dataset()` 方法
  - 确保字段映射正确（user_input→question, reference→ground_truth等）
  - _Requirements: RAGAS兼容性_

## 2. 定义RAG系统接口
- [x] 2.1 创建 `RAGInterface` 抽象基类
  - 定义 `retrieve(query: str, top_k: int) -> List[str]` 方法
  - 定义 `generate(query: str, contexts: List[str]) -> str` 方法
  - 使用 `abc.ABC` 和 `@abstractmethod`
  - _Requirements: RAG系统抽象接口_

- [x] 2.2 创建 `RAGConfig` dataclass
  - 配置检索参数（top_k, similarity_threshold等）
  - 配置生成参数（max_length, temperature等）
  - _Requirements: RAG配置管理_

## 3. 实现核心prepare函数
- [x] 3.1 实现 `prepare_experiment_dataset()` 函数
  - 接收 `GoldenDataset` 和 `RAGInterface` 作为参数
  - 遍历golden records，调用RAG系统填充数据
  - 返回 `ExperimentDataset` 对象
  - _Requirements: 实验数据集准备_

- [x] 3.2 添加进度显示
  - 使用 `tqdm` 显示处理进度
  - 显示当前处理的记录数和总数
  - _Requirements: 用户体验_

- [x] 3.3 添加错误处理
  - 捕获RAG系统调用异常
  - 记录失败的记录，继续处理其他记录
  - 提供失败记录的汇总报告
  - _Requirements: 错误处理_

- [x] 3.4 支持批量处理
  - 支持批量调用RAG系统（如果RAG系统支持）
  - 优化大数据集的处理效率
  - _Requirements: 性能优化_

## 4. 实现数据持久化
- [x] 4.1 实现 `save_experiment_dataset()` 函数
  - 保存为JSONL格式
  - 每行一个JSON对象
  - 使用UTF-8编码
  - _Requirements: 数据保存_

- [x] 4.2 实现 `load_experiment_dataset()` 函数
  - 从JSONL文件加载
  - 验证数据格式
  - 返回 `ExperimentDataset` 对象
  - _Requirements: 数据加载_

- [x] 4.3 添加数据验证
  - 验证必需字段是否存在
  - 验证字段类型是否正确
  - 提供详细的验证错误信息
  - _Requirements: 数据验证_

## 5. 创建示例RAG实现
- [x] 5.1 创建 `DummyRAG` 类用于测试
  - 实现 `RAGInterface` 接口
  - retrieve方法返回模拟的上下文
  - generate方法返回模拟的答案
  - 用于单元测试和示例演示
  - _Requirements: 测试支持_

- [x] 5.2 创建 `SimpleRAG` 类作为简单示例
  - 基于BM25的简单检索
  - 基于模板的简单生成
  - 不依赖外部模型，便于快速测试
  - _Requirements: 示例实现_

## 6. 编写示例代码
- [x] 6.1 创建 `examples/prepare_experiment_dataset.py`
  - 演示如何加载Golden Dataset
  - 演示如何使用DummyRAG准备实验数据
  - 演示如何保存和加载实验数据集
  - _Requirements: 使用示例_

- [x] 6.2 创建 `examples/custom_rag_integration.py`
  - 演示如何实现自定义RAG接口
  - 演示如何集成用户自己的RAG系统
  - _Requirements: 集成示例_

## 7. 编写文档
- [x] 7.1 创建 `src/rag_benchmark/prepare/README.md`
  - 说明prepare模块的功能和用途
  - 提供快速开始指南
  - 说明RAG接口的实现要求
  - 提供完整的API文档
  - _Requirements: 模块文档_

- [x] 7.2 更新项目主README
  - 添加prepare模块的介绍
  - 更新使用流程说明
  - 添加prepare阶段的示例
  - _Requirements: 项目文档_

## 8. 单元测试
- [ ]* 8.1 测试ExperimentRecord和ExperimentDataset
  - 测试数据创建和序列化
  - 测试RAGAS格式转换
  - 测试统计信息计算
  - _Requirements: Schema测试_

- [ ]* 8.2 测试prepare_experiment_dataset函数
  - 使用DummyRAG测试正常流程
  - 测试错误处理
  - 测试进度显示
  - _Requirements: 核心功能测试_

- [ ]* 8.3 测试数据持久化
  - 测试保存和加载
  - 测试数据验证
  - 测试错误格式处理
  - _Requirements: 持久化测试_

## 9. 集成测试
- [ ]* 9.1 端到端测试
  - 从加载Golden Dataset到生成Experiment Dataset
  - 验证生成的数据格式正确
  - 验证数据可以被RAGAS使用
  - _Requirements: 集成测试_

## 10. 最终检查
- [x] 10.1 代码质量检查
  - 运行 `black` 格式化代码
  - 运行 `isort` 整理导入
  - 运行 `mypy` 类型检查
  - _Requirements: 代码质量_

- [x] 10.2 确保所有测试通过
  - 运行所有单元测试
  - 运行所有集成测试
  - 确保测试覆盖率达标
  - _Requirements: 测试通过_

- [x] 10.3 验证示例代码可运行
  - 运行所有示例脚本
  - 确保示例输出正确
  - 验证文档与代码一致
  - _Requirements: 示例验证_
