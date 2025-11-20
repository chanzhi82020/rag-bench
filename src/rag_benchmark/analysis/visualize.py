"""可视化模块"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compare import ResultComparison

logger = logging.getLogger(__name__)


def plot_metrics(
    comparison: ResultComparison,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制指标对比图
    
    Args:
        comparison: ResultComparison实例
        metrics: 要绘制的指标列表，默认绘制所有指标
        figsize: 图表大小
        save_path: 保存路径，如果提供则保存图表
        
    Returns:
        matplotlib Figure对象
        
    Example:
        >>> fig = plot_metrics(comparison, metrics=["faithfulness", "answer_relevancy"])
        >>> plt.show()
    """
    if metrics is None:
        metrics = comparison.metrics
    
    # 准备数据
    summary = comparison.summary(metrics)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置x轴位置
    x = np.arange(len(metrics))
    width = 0.8 / len(comparison.names)
    
    # 为每个模型绘制柱状图
    for i, name in enumerate(comparison.names):
        values = [summary.loc[summary["Model/System"] == name, metric].values[0] 
                 for metric in metrics]
        offset = (i - len(comparison.names)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name)
    
    # 设置图表属性
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("RAG System Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")
    
    return fig


def plot_comparison(
    comparison: ResultComparison,
    metric: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制单个指标的详细对比图（包含误差条）
    
    Args:
        comparison: ResultComparison实例
        metric: 要绘制的指标
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        matplotlib Figure对象
    """
    # 提取数据
    names = comparison.names
    means = [comparison.comparison_df.loc[i, f"{metric}_mean"] for i in range(len(names))]
    stds = [comparison.comparison_df.loc[i, f"{metric}_std"] for i in range(len(names))]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    
    # 设置图表属性
    ax.set_xlabel("Model/System")
    ax.set_ylabel(f"{metric} Score")
    ax.set_title(f"{metric} Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std, f'{mean:.3f}±{std:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")
    
    return fig


def plot_distribution(
    comparison: ResultComparison,
    metric: str,
    model_idx: int = 0,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制指标分布直方图
    
    Args:
        comparison: ResultComparison实例
        metric: 要绘制的指标
        model_idx: 模型索引
        bins: 直方图bins数量
        figsize: 图表大小
        save_path: 保存路径
        
    Returns:
        matplotlib Figure对象
    """
    result = comparison.results[model_idx]
    df = result.to_pandas()
    
    if metric not in df.columns:
        raise ValueError(f"指标 {metric} 不存在")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    values = df[metric].dropna()
    ax.hist(values, bins=bins, alpha=0.7, edgecolor='black')
    
    # 添加统计线
    mean = values.mean()
    median = values.median()
    ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.3f}')
    ax.axvline(median, color='g', linestyle='--', label=f'Median: {median:.3f}')
    
    # 设置图表属性
    ax.set_xlabel(f"{metric} Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{metric} Distribution - {comparison.names[model_idx]}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")
    
    return fig
