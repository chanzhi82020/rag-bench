# Analysis Module

结果分析和对比模块，用于分析和可视化RAG系统的评测结果。

## 功能特性

- ✅ 多模型结果对比
- ✅ 指标统计分析（均值、标准差、最大/最小值）
- ✅ 可视化图表生成
- ✅ 最差样本分析
- ✅ 结果导出

## 快速开始

### 1. 对比多个评测结果

```python
from rag_benchmark.analysis import compare_results

# 对比两个模型的评测结果
comparison = compare_results(
    results=[result1, result2],
    names=["Baseline RAG", "Improved RAG"],
    metrics=["faithfulness", "answer_relevancy", "context_recall"]
)

# 查看对比摘要
print(comparison.summary())
```

输出示例：
```
  Model/System  faithfulness  answer_relevancy  context_recall
0  Baseline RAG      0.756234          0.823451        0.678912
1  Improved RAG      0.834567          0.891234        0.756789
```

### 2. 查找最佳模型

```python
# 找出faithfulness最高的模型
best = comparison.get_best("faithfulness", higher_is_better=True)
print(f"Best model: {best['name']} with score {best['score']:.4f}")
```

### 3. 分析最差样本

```python
# 获取第一个模型在faithfulness上表现最差的5个样本
worst_cases = comparison.get_worst_cases(
    metric="faithfulness",
    n=5,
    model_idx=0
)
print(worst_cases)
```

### 4. 可视化对比

```python
from rag_benchmark.analysis import plot_metrics, plot_comparison
import matplotlib.pyplot as plt

# 绘制多指标对比图
fig = plot_metrics(
    comparison,
    metrics=["faithfulness", "answer_relevancy", "context_recall"],
    save_path="output/comparison.png"
)
plt.show()

# 绘制单个指标的详细对比（带误差条）
fig = plot_comparison(
    comparison,
    metric="faithfulness",
    save_path="output/faithfulness_comparison.png"
)
plt.show()
```

### 5. 查看指标分布

```python
from rag_benchmark.analysis import plot_distribution

# 绘制第一个模型的faithfulness分布
fig = plot_distribution(
    comparison,
    metric="faithfulness",
    model_idx=0,
    save_path="output/faithfulness_dist.png"
)
plt.show()
```

## API 参考

### compare_results()

对比多个评测结果。

**参数：**
- `results` (List[EvaluationResult]): 评测结果列表
- `names` (Optional[List[str]]): 模型名称列表，默认为"Model 1", "Model 2"等
- `metrics` (Optional[List[str]]): 要对比的指标列表，默认对比所有指标

**返回：**
- `ResultComparison`: 对比结果对象

### ResultComparison

评测结果对比类。

**属性：**
- `names`: 模型名称列表
- `results`: 评测结果列表
- `metrics`: 对比的指标列表
- `comparison_df`: 对比结果DataFrame

**方法：**

#### summary(metrics=None)
生成对比摘要表格。

**参数：**
- `metrics` (Optional[List[str]]): 要显示的指标，默认显示所有

**返回：**
- `pd.DataFrame`: 对比摘要

#### get_best(metric, higher_is_better=True)
获取指定指标的最佳模型。

**参数：**
- `metric` (str): 指标名称
- `higher_is_better` (bool): True表示越高越好

**返回：**
- `Dict`: 包含name和score的字典

#### get_worst_cases(metric, n=5, model_idx=0)
获取表现最差的样本。

**参数：**
- `metric` (str): 指标名称
- `n` (int): 返回样本数量
- `model_idx` (int): 模型索引

**返回：**
- `pd.DataFrame`: 最差样本

#### save(path)
保存对比结果到CSV文件。

### 可视化函数

#### plot_metrics()
绘制多指标对比柱状图。

**参数：**
- `comparison` (ResultComparison): 对比结果对象
- `metrics` (Optional[List[str]]): 要绘制的指标
- `figsize` (Tuple[int, int]): 图表大小
- `save_path` (Optional[str]): 保存路径

#### plot_comparison()
绘制单个指标的详细对比图（带误差条）。

**参数：**
- `comparison` (ResultComparison): 对比结果对象
- `metric` (str): 要绘制的指标
- `figsize` (Tuple[int, int]): 图表大小
- `save_path` (Optional[str]): 保存路径

#### plot_distribution()
绘制指标分布直方图。

**参数：**
- `comparison` (ResultComparison): 对比结果对象
- `metric` (str): 要绘制的指标
- `model_idx` (int): 模型索引
- `bins` (int): 直方图bins数量
- `figsize` (Tuple[int, int]): 图表大小
- `save_path` (Optional[str]): 保存路径

## 完整示例

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, DummyRAG
from rag_benchmark.evaluate import evaluate_e2e
from rag_benchmark.analysis import compare_results, plot_metrics
import matplotlib.pyplot as plt

# 1. 准备两个不同的RAG系统
golden_ds = GoldenDataset("xquad", subset="zh")
rag1 = DummyRAG()  # Baseline
rag2 = ImprovedRAG()  # Your improved RAG

# 2. 生成实验数据集
exp_ds1 = prepare_experiment_dataset(golden_ds, rag1)
exp_ds2 = prepare_experiment_dataset(golden_ds, rag2)

# 3. 评测
result1 = evaluate_e2e(exp_ds1, experiment_name="baseline")
result2 = evaluate_e2e(exp_ds2, experiment_name="improved")

# 4. 对比分析
comparison = compare_results(
    results=[result1, result2],
    names=["Baseline", "Improved"]
)

# 5. 查看结果
print(comparison.summary())
print(f"\nBest faithfulness: {comparison.get_best('faithfulness')}")

# 6. 可视化
plot_metrics(comparison, save_path="output/comparison.png")
plt.show()

# 7. 分析问题样本
worst = comparison.get_worst_cases("faithfulness", n=3, model_idx=0)
print("\nWorst cases:")
print(worst)

# 8. 保存结果
comparison.save("output/comparison_results.csv")
```

## 注意事项

1. **指标选择**：确保对比的模型都计算了相同的指标
2. **样本数量**：对比的数据集应该有足够的样本以获得统计显著性
3. **可视化**：需要安装matplotlib：`pip install matplotlib`
4. **内存使用**：对比大量结果时注意内存占用

## 高级用法

### 自定义对比逻辑

```python
# 访问原始DataFrame进行自定义分析
df = comparison.comparison_df

# 计算相对提升
baseline_faith = df.loc[0, "faithfulness_mean"]
improved_faith = df.loc[1, "faithfulness_mean"]
improvement = (improved_faith - baseline_faith) / baseline_faith * 100
print(f"Faithfulness improved by {improvement:.2f}%")
```

### 批量对比

```python
# 对比多个版本
results = [result_v1, result_v2, result_v3, result_v4]
names = ["v1.0", "v1.1", "v1.2", "v2.0"]

comparison = compare_results(results, names)
plot_metrics(comparison, save_path="output/version_comparison.png")
```
