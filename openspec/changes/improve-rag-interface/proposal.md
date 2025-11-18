# Proposal: Improve RAGInterface Extensibility

## Why

当前的`RAGInterface`设计存在扩展性问题：

### 问题1：retrieve()返回类型过于简单
```python
def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
```

**限制：**
- 只能返回上下文文本，无法返回额外信息
- 无法返回`retrieved_context_ids`（RAGAS评测需要）
- 无法返回相似度分数
- 无法返回检索元数据（如来源、时间戳等）

### 问题2：generate()返回类型过于简单
```python
def generate(self, query: str, contexts: List[str]) -> str:
```

**限制：**
- 只能返回单个答案，无法支持多候选答案
- 无法返回生成置信度
- 无法返回生成元数据（如token数、耗时等）

### 问题3：向后兼容性
任何接口改动都会破坏现有用户代码。

## What Changes

### 核心设计原则

1. **使用结构化返回类型**：引入`RetrievalResult`和`GenerationResult`数据类
2. **保持向后兼容**：提供适配器方法，支持旧接口
3. **可选扩展字段**：额外信息通过可选字段提供
4. **渐进式迁移**：允许用户逐步迁移到新接口

### 新的数据结构

```python
@dataclass
class RetrievalResult:
    """检索结果
    
    Attributes:
        contexts: 检索到的上下文文本列表
        context_ids: 上下文ID列表（可选）
        scores: 相似度分数列表（可选）
        metadata: 额外元数据（可选）
    """
    contexts: List[str]
    context_ids: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """生成结果
    
    Attributes:
        response: 主要答案
        multi_responses: 多个候选答案（可选）
        confidence: 置信度（可选）
        metadata: 额外元数据（可选）
    """
    response: str
    multi_responses: Optional[List[str]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

### 新的接口设计

```python
class RAGInterface(ABC):
    """RAG系统抽象接口（改进版）"""
    
    # 新接口（推荐使用）
    def retrieve_with_metadata(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalResult:
        """检索相关上下文（带元数据）
        
        子类应该优先实现此方法。如果未实现，将调用旧的retrieve()方法。
        """
        # 默认实现：调用旧接口并包装结果
        contexts = self.retrieve(query, top_k)
        return RetrievalResult(contexts=contexts)
    
    def generate_with_metadata(
        self, query: str, contexts: List[str]
    ) -> GenerationResult:
        """生成答案（带元数据）
        
        子类应该优先实现此方法。如果未实现，将调用旧的generate()方法。
        """
        # 默认实现：调用旧接口并包装结果
        response = self.generate(query, contexts)
        return GenerationResult(response=response)
    
    # 旧接口（保持向后兼容）
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """检索相关上下文（旧接口，保持兼容）
        
        默认实现：调用新接口并提取contexts字段。
        子类可以继续实现此方法以保持兼容性。
        """
        result = self.retrieve_with_metadata(query, top_k)
        return result.contexts
    
    def generate(self, query: str, contexts: List[str]) -> str:
        """生成答案（旧接口，保持兼容）
        
        默认实现：调用新接口并提取response字段。
        子类可以继续实现此方法以保持兼容性。
        """
        result = self.generate_with_metadata(query, contexts)
        return result.response
```

### 迁移策略

#### 策略1：用户继续使用旧接口（无需改动）
```python
class MyRAG(RAGInterface):
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        # 旧代码继续工作
        return ["context1", "context2"]
    
    def generate(self, query: str, contexts: List[str]) -> str:
        # 旧代码继续工作
        return "answer"
```

#### 策略2：用户迁移到新接口（获得扩展能力）
```python
class MyRAG(RAGInterface):
    def retrieve_with_metadata(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalResult:
        # 新接口，可以返回额外信息
        contexts = ["context1", "context2"]
        context_ids = ["id1", "id2"]
        scores = [0.95, 0.87]
        return RetrievalResult(
            contexts=contexts,
            context_ids=context_ids,
            scores=scores
        )
    
    def generate_with_metadata(
        self, query: str, contexts: List[str]
    ) -> GenerationResult:
        # 新接口，可以返回多个候选答案
        return GenerationResult(
            response="main answer",
            multi_responses=["answer1", "answer2", "answer3"],
            confidence=0.92
        )
```

### prepare模块的适配

```python
def _process_single_record(
    golden_record: GoldenRecord,
    rag_system: RAGInterface,
    top_k: Optional[int] = None,
) -> SingleTurnSample:
    """处理单条记录（改进版）"""
    
    # 使用新接口（如果可用）
    retrieval_result = rag_system.retrieve_with_metadata(
        query=golden_record.user_input,
        top_k=top_k,
    )
    
    generation_result = rag_system.generate_with_metadata(
        query=golden_record.user_input,
        contexts=retrieval_result.contexts,
    )
    
    # 创建RAGAS的SingleTurnSample，使用额外信息
    exp_record = SingleTurnSample(
        user_input=golden_record.user_input,
        reference=golden_record.reference,
        reference_contexts=golden_record.reference_contexts,
        retrieved_contexts=retrieval_result.contexts,
        response=generation_result.response,
        # 使用新的元数据
        retrieved_context_ids=retrieval_result.context_ids,  # 新增
        multi_responses=generation_result.multi_responses,    # 新增
    )
    
    return exp_record
```

## Impact

### Affected Specs
- 更新 `prepare` 模块的spec，添加扩展接口说明

### Affected Code
- `src/rag_benchmark/prepare/rag_interface.py` - 添加新的数据类和接口方法
- `src/rag_benchmark/prepare/prepare.py` - 适配新接口
- `src/rag_benchmark/prepare/dummy_rag.py` - 更新示例实现
- `examples/custom_rag_integration.py` - 添加新接口使用示例
- `src/rag_benchmark/prepare/README.md` - 更新文档

### Dependencies
- 无新增依赖
- 完全向后兼容，不破坏现有代码

### Benefits
1. **扩展性**：支持返回额外元数据
2. **向后兼容**：现有代码无需修改
3. **渐进式迁移**：用户可以按需迁移
4. **更好的RAGAS集成**：支持`retrieved_context_ids`、`multi_responses`等字段
5. **未来扩展**：为未来需求预留空间

### Migration Path

**Phase 1: 添加新接口（不破坏现有代码）**
- 添加`RetrievalResult`和`GenerationResult`
- 添加`retrieve_with_metadata()`和`generate_with_metadata()`
- 保持旧接口完全兼容

**Phase 2: 更新示例和文档**
- 更新文档说明新接口
- 提供迁移指南
- 更新示例代码

**Phase 3: 鼓励迁移（可选）**
- 在旧接口添加deprecation警告（可选）
- 提供自动迁移工具（可选）

### Non-Goals (本次不包含)
- 不强制用户迁移到新接口
- 不移除旧接口
- 不改变现有API的行为
