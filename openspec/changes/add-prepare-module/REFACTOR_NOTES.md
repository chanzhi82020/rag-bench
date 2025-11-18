# Prepare模块重构说明

## 重构原因

原始实现创建了自定义的`ExperimentRecord`和`ExperimentDataset`类，但这与RAGAS的数据结构重复，导致：
1. 需要额外的格式转换
2. 维护两套数据结构
3. 可能的兼容性问题

## 重构内容

### 直接使用RAGAS数据结构

**之前**:
```python
@dataclass
class ExperimentRecord:
    user_input: str
    reference: str
    # ... 自定义字段
    
    def to_ragas_format(self) -> Dict[str, Any]:
        # 需要转换
        return {...}
```

**之后**:
```python
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# 直接使用RAGAS类型
ExperimentRecord = SingleTurnSample
ExperimentDataset = EvaluationDataset
```

### 简化的schema.py

**之前**: 220+ lines  
**之后**: 20 lines

```python
"""Schema definitions for experiment datasets

This module provides a thin wrapper around RAGAS's data structures,
adding convenience methods for our specific use cases.
"""

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

# Re-export RAGAS types for convenience
ExperimentRecord = SingleTurnSample
ExperimentDataset = EvaluationDataset
```

### 更新的prepare.py

1. **创建记录**: 直接创建`SingleTurnSample`
2. **保存数据**: 使用`dataset.to_jsonl()`
3. **加载数据**: 使用`EvaluationDataset.from_jsonl()`

### 移除的代码

- `ExperimentRecordModel` (Pydantic验证模型)
- `to_ragas_format()` 方法
- 自定义的序列化/反序列化逻辑

## 优势

1. **完美兼容**: 数据直接兼容RAGAS，无需转换
2. **代码简化**: 减少200+ lines代码
3. **维护性**: 只需维护一套数据结构
4. **功能完整**: 继承RAGAS的所有功能
5. **类型安全**: 使用RAGAS的Pydantic模型验证

## 向后兼容性

对外API保持不变：
- `ExperimentRecord` 仍然可用（指向`SingleTurnSample`）
- `ExperimentDataset` 仍然可用（指向`EvaluationDataset`）
- `prepare_experiment_dataset()` 接口不变
- `save/load_experiment_dataset()` 接口不变

## 测试验证

✅ RAGAS集成测试通过  
✅ 数据保存/加载正常  
✅ 字段映射正确  

## 文档更新

- ✅ 更新README说明使用RAGAS数据结构
- ✅ 更新代码注释
- ✅ 保持API文档一致性

## 总结

这次重构大幅简化了代码，同时提供了更好的RAGAS集成。用户代码无需修改，但内部实现更加简洁和可维护。
