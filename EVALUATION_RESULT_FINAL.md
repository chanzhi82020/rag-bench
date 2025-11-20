# 评测结果处理 - 最终方案

## 核心思路

**直接使用ragas内部计算好的结果，而不是自己重新计算。**

## ragas EvaluationResult 结构

```python
@dataclass
class EvaluationResult:
    scores: List[Dict[str, Any]]  # 每个样本的评测结果
    dataset: EvaluationDataset
    
    def __post_init__(self):
        # ragas内部会计算指标平均值
        self._scores_dict = {
            k: [d[k] for d in self.scores] 
            for k in self.scores[0].keys()
        }
        
        self._repr_dict = {}
        for metric_name in self._scores_dict.keys():
            value = safe_nanmean(self._scores_dict[metric_name])
            self._repr_dict[metric_name] = value
```

### 关键属性

1. **`scores`**: 每个样本的评测结果
   ```python
   [
       {"user_input": "...", "faithfulness": 0.95, "answer_relevancy": 0.88},
       {"user_input": "...", "faithfulness": 0.90, "answer_relevancy": 0.85},
   ]
   ```

2. **`_repr_dict`**: 指标平均值（ragas内部计算）
   ```python
   {
       "faithfulness": 0.9250,
       "answer_relevancy": 0.8650
   }
   ```

3. **`to_pandas()`**: 返回完整的DataFrame
   ```
   user_input  response  faithfulness  answer_relevancy
   "..."       "..."     0.95          0.88
   "..."       "..."     0.90          0.85
   ```

## 后端处理

### 代码实现

```python
# 阶段4: 处理结果
try:
    # 方法1: 使用ragas内部计算好的指标平均值
    if hasattr(result, '_repr_dict'):
        metrics = dict(result._repr_dict)
    else:
        # 备用方法: 解析str(result)
        import json
        metrics_str = str(result)
        metrics = json.loads(metrics_str.replace("'", '"'))
    
    # 获取详细结果（用于前端展开显示）
    df = result.to_pandas()
    detailed_results = df.to_dict('records')
    sample_count = len(df)
    
except Exception as e:
    logger.error(f"提取指标失败: {e}", exc_info=True)
    metrics = {}
    detailed_results = []
    sample_count = 0

# 返回结果
update_task_status(
    task_id,
    status="completed",
    result={
        "metrics": metrics,              # 指标平均值
        "detailed_results": detailed_results,  # 详细结果
        "sample_count": sample_count,
        "eval_type": request.eval_type
    }
)
```

### 返回数据结构

```json
{
  "task_id": "xxx",
  "status": "completed",
  "result": {
    "metrics": {
      "faithfulness": 0.9267,
      "answer_relevancy": 0.8767,
      "context_precision": 0.9033
    },
    "detailed_results": [
      {
        "user_input": "What is Python?",
        "response": "Python is...",
        "faithfulness": 0.95,
        "answer_relevancy": 0.88,
        "context_precision": 0.92
      },
      {
        "user_input": "What is AI?",
        "response": "AI is...",
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "context_precision": 0.88
      }
    ],
    "sample_count": 2,
    "eval_type": "e2e"
  }
}
```

## 前端显示

### 1. 指标平均值（默认显示）

```tsx
<div className="border border-gray-200 rounded-lg p-4">
  <h4 className="font-semibold text-gray-900 mb-3">
    评测指标（平均值）
  </h4>
  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
    {Object.entries(taskStatus.result.metrics).map(([key, value]) => (
      <div key={key} className="bg-gray-50 p-3 rounded">
        <div className="text-xs text-gray-500 uppercase">{key}</div>
        <div className="text-lg font-bold text-gray-900">
          {value.toFixed(4)}
        </div>
      </div>
    ))}
  </div>
  <div className="mt-3 text-sm text-gray-600">
    样本数: {taskStatus.result.sample_count} | 
    评测类型: {taskStatus.result.eval_type}
  </div>
</div>
```

### 2. 详细结果（可展开）

```tsx
<div className="border border-gray-200 rounded-lg p-4">
  <details className="group">
    <summary className="font-semibold text-gray-900 cursor-pointer">
      <span>详细评测结果 ({detailed_results.length} 个样本)</span>
      <span className="group-open:rotate-180">▼</span>
    </summary>
    <div className="mt-4 overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            <th>样本</th>
            {Object.keys(metrics).map(metric => (
              <th key={metric}>{metric}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {detailed_results.map((row, idx) => (
            <tr key={idx}>
              <td>#{idx + 1}</td>
              {Object.keys(metrics).map(metric => (
                <td key={metric}>
                  {row[metric]?.toFixed(4) || '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </details>
</div>
```

### 显示效果

**默认状态（折叠）**:
```
评测指标（平均值）
┌─────────────────────┬─────────┐
│ faithfulness        │ 0.9267  │
│ answer_relevancy    │ 0.8767  │
│ context_precision   │ 0.9033  │
└─────────────────────┴─────────┘
样本数: 3 | 评测类型: e2e

▶ 详细评测结果 (3 个样本)
```

**展开状态**:
```
评测指标（平均值）
┌─────────────────────┬─────────┐
│ faithfulness        │ 0.9267  │
│ answer_relevancy    │ 0.8767  │
│ context_precision   │ 0.9033  │
└─────────────────────┴─────────┘
样本数: 3 | 评测类型: e2e

▼ 详细评测结果 (3 个样本)
┌──────┬──────────────┬──────────────────┬───────────────────┐
│ 样本 │ faithfulness │ answer_relevancy │ context_precision │
├──────┼──────────────┼──────────────────┼───────────────────┤
│ #1   │ 0.9500       │ 0.8800           │ 0.9200            │
│ #2   │ 0.9000       │ 0.8500           │ 0.8800            │
│ #3   │ 0.9300       │ 0.9000           │ 0.9100            │
└──────┴──────────────┴──────────────────┴───────────────────┘
```

## 优势

### 1. 正确性
- ✅ 使用ragas内部计算的结果
- ✅ 考虑了NaN值和特殊情况
- ✅ 只包含指标列，不包含输入/输出列

### 2. 完整性
- ✅ 提供指标平均值（快速查看）
- ✅ 提供详细结果（深入分析）
- ✅ 支持展开/折叠（灵活显示）

### 3. 可靠性
- ✅ 不需要自己判断哪些是指标列
- ✅ 不需要担心response列包含数字
- ✅ 使用ragas官方计算逻辑

### 4. 用户体验
- ✅ 默认显示摘要（指标平均值）
- ✅ 可选查看详情（每个样本）
- ✅ 表格形式清晰易读

## 对比之前的方案

### ❌ 错误方案1: 对所有列计算平均值
```python
df = result.to_pandas()
metrics = {k: float(v) for k, v in df.mean().to_dict().items()}
# 问题: response列如果包含数字也会被计算
```

### ❌ 错误方案2: 使用select_dtypes
```python
numeric_cols = df.select_dtypes(include=['number']).columns
metrics = {col: df[col].mean() for col in numeric_cols}
# 问题: 可能包含非指标的数值列
```

### ✅ 正确方案: 使用_repr_dict
```python
metrics = dict(result._repr_dict)
# 优势: ragas内部计算，只包含指标
```

## 总结

**核心原则**: 信任ragas的内部实现，使用它计算好的结果。

**实现方式**:
1. 指标平均值: `result._repr_dict`
2. 详细结果: `result.to_pandas().to_dict('records')`

**前端显示**:
1. 默认: 指标平均值卡片
2. 可选: 详细结果表格（展开）

**用户价值**:
- 快速了解整体表现（平均值）
- 深入分析个别样本（详细结果）
- 灵活的展示方式（展开/折叠）
