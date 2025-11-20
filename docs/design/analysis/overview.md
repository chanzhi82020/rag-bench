# Analysis Module Design

## Overview

The analysis module provides result comparison and visualization capabilities for analyzing RAG evaluation results, enabling multi-model comparison, statistical analysis, and visual insights.

## Architecture

```
analysis/
├── compare.py     # Result comparison logic
└── visualize.py   # Visualization functions
```

## Core Components

### 1. ResultComparison

Central class for comparing multiple evaluation results:

```python
@dataclass
class ResultComparison:
    names: List[str]                      # Model/system names
    results: List[EvaluationResult]       # RAGAS evaluation results
    metrics: List[str]                    # Metrics to compare
    comparison_df: Optional[pd.DataFrame] # Comparison DataFrame
```

**Key Methods**:

**summary()**: Generate comparison summary table
```python
def summary(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - Model/System
    - metric1_mean
    - metric2_mean
    - ...
    """
```

**get_best()**: Find best performing model
```python
def get_best(self, metric: str, higher_is_better: bool = True) -> Dict:
    """
    Returns: {"name": "Model A", "score": 0.85}
    """
```

**get_worst_cases()**: Identify problematic samples
```python
def get_worst_cases(
    self, 
    metric: str, 
    n: int = 5,
    model_idx: int = 0
) -> pd.DataFrame:
    """
    Returns DataFrame with worst performing samples
    """
```

**save()**: Export comparison results
```python
def save(self, path: str):
    """Save comparison to CSV"""
```

**Design Rationale**:
- Encapsulates comparison logic
- Provides rich statistical analysis
- Enables easy result export

### 2. Comparison DataFrame Structure

The internal comparison DataFrame stores comprehensive statistics:

```
Columns:
- name: Model/system name
- {metric}_mean: Mean score for metric
- {metric}_std: Standard deviation
- {metric}_min: Minimum score
- {metric}_max: Maximum score

Example:
| name      | faithfulness_mean | faithfulness_std | faithfulness_min | faithfulness_max |
|-----------|-------------------|------------------|------------------|------------------|
| Model A   | 0.85              | 0.12             | 0.45             | 0.98             |
| Model B   | 0.78              | 0.15             | 0.38             | 0.95             |
```

**Design Rationale**:
- Comprehensive statistics enable deep analysis
- Structured format supports various visualizations
- Easy to extend with additional statistics

### 3. Visualization Functions

**plot_metrics()**: Multi-metric comparison bar chart
```python
def plot_metrics(
    comparison: ResultComparison,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates grouped bar chart comparing multiple metrics across models
    
    Features:
    - Side-by-side bars for each model
    - Grid for easy reading
    - Legend for model identification
    - Automatic layout adjustment
    """
```

**plot_comparison()**: Single metric detailed comparison
```python
def plot_comparison(
    comparison: ResultComparison,
    metric: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates bar chart with error bars for single metric
    
    Features:
    - Error bars showing standard deviation
    - Value labels (mean ± std)
    - Grid for easy reading
    """
```

**plot_distribution()**: Metric distribution histogram
```python
def plot_distribution(
    comparison: ResultComparison,
    metric: str,
    model_idx: int = 0,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates histogram showing metric distribution
    
    Features:
    - Histogram of metric values
    - Mean and median lines
    - Legend with statistics
    - Grid for easy reading
    """
```

**Design Rationale**:
- Matplotlib-based for flexibility
- Consistent styling across plots
- Optional save for reports
- Informative defaults

### 4. compare_results() Function

Main entry point for creating comparisons:

```python
def compare_results(
    results: List[EvaluationResult],
    names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> ResultComparison:
    """
    Compare multiple evaluation results
    
    Args:
        results: List of RAGAS EvaluationResult objects
        names: Model names (default: "Model 1", "Model 2", ...)
        metrics: Metrics to compare (default: all available)
    
    Returns:
        ResultComparison object
    """
```

**Design Rationale**:
- Simple API for common use case
- Automatic metric extraction
- Sensible defaults

## Data Flow

### Comparison Creation Flow
```
[EvaluationResult] + [names]
    ↓
compare_results()
    ↓
Extract metrics from each result
    ↓
For each result:
    - Convert to DataFrame
    - Compute statistics (mean, std, min, max)
    - Store in comparison_df
    ↓
ResultComparison object
```

### Visualization Flow
```
ResultComparison
    ↓
plot_metrics() / plot_comparison() / plot_distribution()
    ↓
Extract data from comparison_df
    ↓
Create matplotlib figure
    ↓
Apply styling and labels
    ↓
Optional: Save to file
    ↓
Return figure
```

### Analysis Flow
```
ResultComparison
    ↓
summary() → Summary DataFrame
get_best() → Best model info
get_worst_cases() → Problematic samples
    ↓
Insights for improvement
```

## Design Principles

1. **Simplicity**:
   - Single function to create comparison
   - Intuitive method names
   - Sensible defaults

2. **Flexibility**:
   - Support any number of models
   - Select specific metrics
   - Customize visualizations

3. **Informativeness**:
   - Comprehensive statistics
   - Multiple visualization types
   - Error analysis support

4. **Integration**:
   - Works with RAGAS results
   - Compatible with pandas workflow
   - Matplotlib for customization

5. **Reproducibility**:
   - Save results to CSV
   - Save plots to files
   - Structured data format

## Usage Patterns

### Pattern 1: Quick Comparison
```python
from rag_benchmark.analysis import compare_results

comparison = compare_results(
    [result1, result2],
    names=["Baseline", "Improved"]
)
print(comparison.summary())
```

### Pattern 2: Visual Comparison
```python
from rag_benchmark.analysis import compare_results, plot_metrics
import matplotlib.pyplot as plt

comparison = compare_results([result1, result2], names=["A", "B"])
plot_metrics(comparison, metrics=["faithfulness", "answer_relevancy"])
plt.show()
```

### Pattern 3: Detailed Analysis
```python
comparison = compare_results([result1, result2, result3], 
                            names=["v1", "v2", "v3"])

# Find best model
best = comparison.get_best("faithfulness")
print(f"Best: {best['name']} with {best['score']:.3f}")

# Analyze worst cases
worst = comparison.get_worst_cases("faithfulness", n=5, model_idx=0)
print(worst[["user_input", "response", "faithfulness"]])

# Visualize distribution
plot_distribution(comparison, "faithfulness", model_idx=0)
plt.show()
```

### Pattern 4: Report Generation
```python
comparison = compare_results(results, names=model_names)

# Save summary
comparison.save("reports/comparison.csv")

# Save visualizations
plot_metrics(comparison, save_path="reports/metrics.png")
plot_comparison(comparison, "faithfulness", save_path="reports/faithfulness.png")
plot_distribution(comparison, "faithfulness", save_path="reports/dist.png")
```

### Pattern 5: Custom Analysis
```python
comparison = compare_results(results, names=names)

# Access raw DataFrame for custom analysis
df = comparison.comparison_df

# Calculate relative improvement
baseline_score = df.loc[0, "faithfulness_mean"]
improved_score = df.loc[1, "faithfulness_mean"]
improvement = (improved_score - baseline_score) / baseline_score * 100
print(f"Improvement: {improvement:.1f}%")

# Statistical significance testing
from scipy import stats
result1_scores = results[0].to_pandas()["faithfulness"]
result2_scores = results[1].to_pandas()["faithfulness"]
t_stat, p_value = stats.ttest_ind(result1_scores, result2_scores)
print(f"p-value: {p_value:.4f}")
```

## Statistical Analysis

### Metrics Computed

For each metric and model:
- **Mean**: Average performance
- **Std**: Variability/consistency
- **Min**: Worst case performance
- **Max**: Best case performance

### Interpretation Guidelines

**High Mean, Low Std**: Consistent good performance
**High Mean, High Std**: Good average but inconsistent
**Low Mean, Low Std**: Consistently poor
**Low Mean, High Std**: Unreliable, needs investigation

### Comparison Strategies

**Absolute Comparison**: Compare mean scores directly
```python
summary = comparison.summary()
print(summary)
```

**Relative Comparison**: Calculate improvement percentages
```python
baseline = df.loc[0, "faithfulness_mean"]
improved = df.loc[1, "faithfulness_mean"]
improvement = (improved - baseline) / baseline * 100
```

**Statistical Testing**: Test for significance
```python
from scipy import stats
scores1 = results[0].to_pandas()["faithfulness"]
scores2 = results[1].to_pandas()["faithfulness"]
t_stat, p_value = stats.ttest_ind(scores1, scores2)
```

## Visualization Guidelines

### When to Use Each Plot

**plot_metrics()**: 
- Compare multiple metrics at once
- Get overview of system performance
- Identify strengths and weaknesses

**plot_comparison()**:
- Focus on single metric
- Show variability (error bars)
- Compare consistency across models

**plot_distribution()**:
- Understand metric distribution
- Identify outliers
- Check for bimodal distributions

### Customization Examples

```python
# Custom figure size
fig = plot_metrics(comparison, figsize=(16, 8))

# Custom metrics selection
fig = plot_metrics(comparison, metrics=["faithfulness", "answer_relevancy"])

# Custom bins for distribution
fig = plot_distribution(comparison, "faithfulness", bins=30)

# Further customization
fig = plot_metrics(comparison)
ax = fig.gca()
ax.set_ylim(0, 1)
ax.set_title("Custom Title", fontsize=16)
plt.tight_layout()
plt.show()
```

## Error Analysis

### Identifying Problem Cases

```python
# Get worst performing samples
worst = comparison.get_worst_cases("faithfulness", n=10, model_idx=0)

# Analyze patterns
for idx, row in worst.iterrows():
    print(f"Question: {row['user_input']}")
    print(f"Response: {row['response']}")
    print(f"Reference: {row['reference']}")
    print(f"Score: {row['faithfulness']}")
    print("-" * 80)
```

### Common Issues to Look For

1. **Low Faithfulness**: Hallucinations, unsupported claims
2. **Low Answer Relevancy**: Off-topic responses
3. **Low Context Recall**: Missing relevant information
4. **High Variance**: Inconsistent performance

## Performance Considerations

### Memory Usage
- Comparison stores full DataFrames in memory
- For large datasets, consider sampling
- Visualizations create matplotlib figures (memory overhead)

### Computation Time
- Comparison creation: O(n * m) where n=samples, m=metrics
- Visualization: O(n) for data extraction, O(1) for plotting
- Statistical tests: O(n) per comparison

### Optimization Tips
- Reuse ResultComparison object for multiple analyses
- Close matplotlib figures after saving to free memory
- Use specific metrics instead of all metrics

## Future Enhancements

1. **Interactive Visualizations**: Plotly/Bokeh support
2. **Statistical Tests**: Built-in significance testing
3. **Correlation Analysis**: Metric correlation heatmaps
4. **Automated Reports**: Generate PDF/HTML reports
5. **Metric Weighting**: Weighted aggregate scores
6. **Confidence Intervals**: Bootstrap confidence intervals
7. **A/B Testing**: Built-in A/B test analysis
8. **Time Series**: Track metrics over time
9. **Cost Analysis**: Compare evaluation costs
10. **Export Formats**: Support JSON, Excel exports
