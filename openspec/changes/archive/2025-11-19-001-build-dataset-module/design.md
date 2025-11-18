# Dataset Module Design

## Overview
数据集模块是RAG Benchmark的基础组件，负责管理和提供标准化的评测数据。该模块定义了Golden Dataset的标准格式，提供数据加载、转换和验证功能。

## Architecture Decisions

### 1. 数据格式标准化
采用JSONL作为标准数据格式，原因：
- 流式处理，内存效率高
- 易于版本控制
- 支持大规模数据集
- 便于追加和更新

### 2. 模块结构设计
```
rag_benchmark/datasets/
├── golden/              # 内置Golden数据集
│   ├── hotpotqa/       # HotpotQA转换后的数据
│   ├── nq/             # Natural Questions转换后的数据
│   └── private/        # 私有数据集（如智能客服）
├── schemas/            # 数据模式定义
│   ├── golden.py      # Golden Dataset Pydantic模型
│   └── experiment.py  # Experiment Dataset Pydantic模型
├── loaders/           # 数据加载器
│   ├── base.py        # 基础加载器接口
│   └── jsonl.py       # JSONL加载器实现
├── converters/        # 数据转换器
│   ├── hotpotqa.py   # HotpotQA转换器
│   └── nq.py         # NQ转换器
├── validators/        # 数据验证器
│   ├── quality.py    # 数据质量检查
│   └── format.py     # 格式验证
└── registry.py       # 数据集注册表
```

### 3. 核心组件设计

#### Golden Dataset Schema
```python
@dataclass
class GoldenRecord:
    user_input: str                     # 用户问题
    reference: str                      # 参考答案
    reference_contexts: List[str]       # 相关上下文段落
    reference_context_ids: List[str]    # 上下文ID（可选）
    
@dataclass
class CorpusRecord:
    reference_context: str              # 上下文内容
    reference_context_id: str           # 上下文唯一ID
    title: str                         # 所属文档标题
    metadata: Dict[str, Any]           # 额外元数据
```

#### 数据加载器接口
```python
class BaseLoader:
    def load_qac(self) -> Iterator[GoldenRecord]: ...
    def load_corpus(self) -> Iterator[CorpusRecord]: ...
    def validate(self) -> ValidationResult: ...
```

#### 转换器模式
每种公开数据集都有专门的转换器，负责：
1. 解析原始数据格式
2. 提取question、answer、context
3. 生成corpus映射
4. 输出标准JSONL格式

### 4. 数据集注册表
使用注册表模式管理所有数据集：
```python
DATASET_REGISTRY = {
    "hotpotqa": HotpotQAConverter,
    "hotpotqa-distractor": HotpotQADistractorConverter,
    "nq": NaturalQuestionsConverter,
    "customer-service": CustomerServiceConverter,
}
```

### 5. 数据质量保证

#### 验证维度
- **完整性检查**: 必填字段是否存在
- **格式检查**: 数据类型是否符合schema
- **质量检查**: 
  - 问题长度合理性（10-500字符）
  - 答案与问题的相关性（基于关键词重叠）
  - 上下文的覆盖率（是否包含答案所需信息）

#### 质量指标
- 数据规模（小型/中型/大型）
- 平均问题长度
- 平均上下文数量
- 领域分布统计

## Implementation Considerations

### 1. 性能优化
- 使用生成器避免一次性加载全部数据
- 支持数据分片处理
- 实现懒加载机制

### 2. 扩展性设计
- 插件式转换器，易于添加新数据集
- 可配置的验证规则
- 支持自定义数据字段

### 3. 错误处理
- 详细的错误信息和修复建议
- 部分数据损坏时的恢复机制
- 转换过程的日志记录

### 4. 版本管理
- 数据集版本控制
- Schema向后兼容性
- 转换器版本追踪

## Trade-offs

1. **选择JSONL而非Parquet**:
   - 优点: 人类可读、易调试、无需额外依赖
   - 缺点: 文件体积较大、读取速度稍慢
   - 理由: MVP阶段优先考虑简单性和可维护性

2. **内存使用 vs 响应速度**:
   - 选择流式处理减少内存占用
   - 通过缓存机制平衡性能

3. **数据转换的复杂度**:
   - MVP阶段只支持基础转换
   - 复杂的数据清洗留到后续版本

## Future Extensions

1. **支持更多数据格式**: Parquet、HDF5等
2. **增量更新机制**: 支持数据集的增量更新
3. **智能质量评估**: 使用LLM进行数据质量评分
4. **数据增强**: 自动生成类似问题和答案
5. **分布式处理**: 支持大规模数据集的并行转换