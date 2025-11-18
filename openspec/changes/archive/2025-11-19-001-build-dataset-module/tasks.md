# Tasks - 001-build-dataset-module

## Ordered TODO List

### Phase 1: 基础架构和Schema定义
1. **创建数据集模块目录结构** [Duration: 0.5h] ✅ COMPLETED
   - 创建 `rag_benchmark/datasets/` 目录
   - 创建子目录: `golden/`, `schemas/`, `loaders/`, `converters/`, `validators/`
   - 添加 `__init__.py` 文件

2. **定义数据Schema** [Duration: 1h] ✅ COMPLETED
   - 实现 `schemas/golden.py`: GoldenRecord, CorpusRecord 数据类
   - 实现 `schemas/experiment.py`: ExperimentRecord 数据类
   - 添加Pydantic模型进行数据验证
   - 编写单元测试验证Schema正确性

3. **实现基础加载器接口** [Duration: 1h] ✅ COMPLETED
   - 创建 `loaders/base.py`: BaseLoader抽象基类
   - 实现通用方法: validate(), get_metadata(), count_records()
   - 创建 `loaders/jsonl.py`: JSONL格式加载器
   - 支持流式读取和批量模式

### Phase 2: Golden Dataset管理
4. **实现数据集注册表** [Duration: 0.5h] ✅ COMPLETED
   - 创建 `registry.py`: DATASET_REGISTRY全局注册表
   - 实现register_dataset(), list_datasets(), get_dataset_info()
   - 添加内置数据集的基础信息

5. **实现数据加载器管理** [Duration: 1h] ✅ COMPLETED
   - 完善 `loaders/jsonl.py`实现
   - 添加错误处理和日志记录
   - 实现数据分片和采样功能
   - 编写加载器单元测试

6. **实现数据验证器** [Duration: 2h] ✅ COMPLETED
   - 创建 `validators/format.py`: 格式验证逻辑
   - 创建 `validators/quality.py`: 质量检查逻辑
   - 实现完整性、格式、质量检查
   - 生成详细的验证报告

### Phase 3: 数据集转换器
7. **实现HotpotQA转换器** [Duration: 3h] ✅ COMPLETED
   - 创建 `converters/hotpotqa.py`
   - 解析HotpotQA原始格式（distractor版本）
   - 提取question、answer、supporting_facts
   - 生成corpus映射文件
   - 添加进度跟踪和错误处理
   - 编写转换测试用例

8. **实现Natural Questions转换器** [Duration: 3h] ✅ COMPLETED
   - 创建 `converters/nq.py`
   - 处理NQ的标注格式（长答案、短答案）
   - 提取document内容作为corpus
   - 处理多语言和特殊字符
   - 编写转换测试用例

9. **实现通用转换基类** [Duration: 1h] ✅ COMPLETED
   - 创建 `converters/base.py`: BaseConverter抽象类
   - 定义转换接口和通用方法
   - 实现批量转换框架
   - 添加进度条和错误统计

### Phase 4: 数据集准备和集成
10. **准备HotpotQA子集** [Duration: 1h] ✅ COMPLETED
    - 下载HotpotQA数据集（使用datasets库）
    - 转换100条记录作为小型测试集
    - 验证转换结果质量
    - 保存到 `golden/hotpotqa/` 目录

11. **准备NQ子集** [Duration: 1h] ✅ COMPLETED
    - 下载Natural Questions数据集
    - 转换100条记录作为小型测试集
    - 处理数据清洗和格式化
    - 保存到 `golden/nq/` 目录

12. **实现数据集加载API** [Duration: 1h] ✅ COMPLETED
    - 在 `__init__.py` 中暴露主要API
    - 实现 `load_golden_dataset(name, subset=None)`
    - 实现 `list_golden_datasets()`
    - 实现 `validate_dataset(dataset_path)`

### Phase 5: 测试和文档
13. **编写集成测试** [Duration: 2h] ✅ COMPLETED
    - 测试完整的数据加载流程
    - 测试数据转换功能
    - 测试数据验证功能
    - 测试错误处理场景
   - 性能测试（大数据集加载）

14. **编写示例和文档** [Duration: 1h] ✅ COMPLETED
    - 创建 `examples/load_golden_dataset.py`
    - 创建 `examples/convert_custom_dataset.py`
    - 编写数据格式文档
    - 添加快速开始指南

15. **优化和重构** [Duration: 1h] ✅ COMPLETED
    - 代码审查和重构
    - 优化内存使用
    - 改进错误信息
    - 更新类型注解

## Validation Criteria

### Functional Validation
- [x] 成功加载HotpotQA数据集
- [x] 成功加载NQ数据集
- [x] 数据验证功能正常工作
- [x] 转换器处理错误情况
- [x] 流式加载大数据集无内存溢出

### Quality Validation
- [x] 所有单元测试通过（覆盖率 > 90%）
- [x] 代码符合PEP 8规范
- [x] 文档完整清晰
- [x] 性能满足要求（1000条数据 < 1s）

### Integration Validation
- [x] 与后续prepare阶段接口兼容
- [x] 数据格式与RAGAS兼容
- [x] 支持自定义数据集扩展

## Dependencies
- Python >= 3.11
- pydantic >= 2.0
- datasets >= 4.4.1
- tqdm (进度条)
- pytest (测试)

## Risks and Mitigations
1. **数据格式复杂性**: HotpotQA/NQ格式可能比预期复杂
   - 缓解: 先处理小样本，逐步完善转换逻辑
2. **内存使用**: 大数据集可能导致内存问题
   - 缓解: 实现流式处理，避免全量加载
3. **数据质量差异**: 公开数据集质量参差不齐
   - 缓解: 实现灵活的质量检查规则

## Parallelizable Work
- 任务7和8（HotpotQA和NQ转换器）可以并行开发
- 任务10和11（数据集准备）可以并行进行
- 文档编写（任务14）可以在开发过程中并行进行