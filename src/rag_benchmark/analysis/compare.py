"""结果对比分析模块"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pandas as pd
from ragas.dataset_schema import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ResultComparison:
    """评测结果对比类
    
    Attributes:
        names: 模型/系统名称列表
        results: 评测结果列表
        metrics: 对比的指标列表
        comparison_df: 对比结果DataFrame
    """
    
    names: List[str]
    results: List[EvaluationResult]
    metrics: List[str] = field(default_factory=list)
    comparison_df: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if len(self.names) != len(self.results):
            raise ValueError("names和results长度必须相同")
        
        if not self.metrics:
            # 自动提取所有指标
            self.metrics = self._extract_metrics()
        
        # 生成对比DataFrame
        self.comparison_df = self._build_comparison_df()
    
    def _extract_metrics(self) -> List[str]:
        """提取所有可用的指标名称"""
        all_metrics = set()
        for result in self.results:
            df = result.to_pandas()
            # 排除非指标列
            metric_cols = [col for col in df.columns 
                          if col not in ['user_input', 'retrieved_contexts', 
                                       'reference', 'response', 'reference_contexts']]
            all_metrics.update(metric_cols)
        return sorted(list(all_metrics))
    
    def _build_comparison_df(self) -> pd.DataFrame:
        """构建对比DataFrame"""
        comparison_data = []
        
        for name, result in zip(self.names, self.results):
            df = result.to_pandas()
            row = {"name": name}
            
            for metric in self.metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        row[f"{metric}_mean"] = values.mean()
                        row[f"{metric}_std"] = values.std()
                        row[f"{metric}_min"] = values.min()
                        row[f"{metric}_max"] = values.max()
                    else:
                        row[f"{metric}_mean"] = None
                        row[f"{metric}_std"] = None
                        row[f"{metric}_min"] = None
                        row[f"{metric}_max"] = None
                else:
                    row[f"{metric}_mean"] = None
                    row[f"{metric}_std"] = None
                    row[f"{metric}_min"] = None
                    row[f"{metric}_max"] = None
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def summary(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """生成对比摘要
        
        Args:
            metrics: 要显示的指标列表，默认显示所有指标
            
        Returns:
            对比摘要DataFrame
        """
        if metrics is None:
            metrics = self.metrics
        
        # 只显示均值列
        cols = ["name"] + [f"{m}_mean" for m in metrics if f"{m}_mean" in self.comparison_df.columns]
        summary_df = self.comparison_df[cols].copy()
        
        # 重命名列
        summary_df.columns = ["Model/System"] + metrics
        
        return summary_df
    
    def get_best(self, metric: str, higher_is_better: bool = True) -> Dict[str, Union[str, float]]:
        """获取指定指标的最佳结果
        
        Args:
            metric: 指标名称
            higher_is_better: True表示越高越好，False表示越低越好
            
        Returns:
            包含最佳模型名称和分数的字典
        """
        col = f"{metric}_mean"
        if col not in self.comparison_df.columns:
            raise ValueError(f"指标 {metric} 不存在")
        
        if higher_is_better:
            idx = self.comparison_df[col].idxmax()
        else:
            idx = self.comparison_df[col].idxmin()
        
        return {
            "name": self.comparison_df.loc[idx, "name"],
            "score": self.comparison_df.loc[idx, col]
        }
    
    def get_worst_cases(
        self, 
        metric: str, 
        n: int = 5,
        model_idx: int = 0
    ) -> pd.DataFrame:
        """获取指定模型在某指标上表现最差的样本
        
        Args:
            metric: 指标名称
            n: 返回样本数量
            model_idx: 模型索引
            
        Returns:
            最差样本的DataFrame
        """
        result = self.results[model_idx]
        df = result.to_pandas()
        
        if metric not in df.columns:
            raise ValueError(f"指标 {metric} 不存在")
        
        # 按指标排序，取最差的n个
        worst = df.nsmallest(n, metric)
        
        # 只返回关键列
        cols = ["user_input", "response", "reference", metric]
        cols = [c for c in cols if c in worst.columns]
        
        return worst[cols]
    
    def save(self, path: str):
        """保存对比结果
        
        Args:
            path: 保存路径
        """
        self.comparison_df.to_csv(path, index=False)
        logger.info(f"对比结果已保存到: {path}")


def compare_results(
    results: List[EvaluationResult],
    names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> ResultComparison:
    """对比多个评测结果
    
    Args:
        results: 评测结果列表
        names: 模型/系统名称列表，默认使用"Model 1", "Model 2"等
        metrics: 要对比的指标列表，默认对比所有指标
        
    Returns:
        ResultComparison实例
        
    Example:
        >>> comparison = compare_results(
        ...     [result1, result2],
        ...     names=["Baseline", "Improved"],
        ...     metrics=["faithfulness", "answer_relevancy"]
        ... )
        >>> print(comparison.summary())
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(results))]
    
    return ResultComparison(
        names=names,
        results=results,
        metrics=metrics or []
    )
