"""评测结果数据结构"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """单个指标的评测结果

    Attributes:
        name: 指标名称
        score: 平均分数
        scores: 每个样本的分数列表
        metadata: 额外的指标信息
    """

    name: str
    score: float
    scores: List[float]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """验证数据"""
        if not self.scores:
            raise ValueError("Scores list cannot be empty")
        if len(self.scores) == 0:
            raise ValueError("Must have at least one score")

    @property
    def count(self) -> int:
        """样本数量"""
        return len(self.scores)

    @property
    def min_score(self) -> float:
        """最低分数"""
        return min(self.scores)

    @property
    def max_score(self) -> float:
        """最高分数"""
        return max(self.scores)

    @property
    def std_score(self) -> float:
        """分数标准差"""
        if len(self.scores) <= 1:
            return 0.0
        mean = self.score
        variance = sum((x - mean) ** 2 for x in self.scores) / len(self.scores)
        return variance**0.5

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "score": self.score,
            "count": self.count,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "std_score": self.std_score,
            "scores": self.scores,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """评测结果容器

    Attributes:
        name: 评测名称
        metrics: 指标结果字典
        dataset_size: 数据集大小
        timestamp: 评测时间戳
        metadata: 额外信息
    """

    name: str
    metrics: Dict[str, MetricResult]
    dataset_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """验证数据"""
        if not self.metrics:
            raise ValueError("Must have at least one metric result")
        if self.dataset_size <= 0:
            raise ValueError("Dataset size must be positive")

    def get_metric(self, name: str) -> Optional[MetricResult]:
        """获取指定指标的结果

        Args:
            name: 指标名称

        Returns:
            指标结果，如果不存在则返回None
        """
        return self.metrics.get(name)

    def get_score(self, name: str) -> Optional[float]:
        """获取指定指标的平均分数

        Args:
            name: 指标名称

        Returns:
            平均分数，如果不存在则返回None
        """
        metric = self.get_metric(name)
        return metric.score if metric else None

    def list_metrics(self) -> List[str]:
        """获取所有指标名称列表

        Returns:
            指标名称列表
        """
        return list(self.metrics.keys())

    def summary(self) -> Dict[str, Any]:
        """获取评测结果摘要

        Returns:
            包含摘要信息的字典
        """
        if not self.metrics:
            return {
                "name": self.name,
                "dataset_size": self.dataset_size,
                "metrics_count": 0,
                "timestamp": self.timestamp.isoformat(),
            }

        scores = [metric.score for metric in self.metrics.values()]
        return {
            "name": self.name,
            "dataset_size": self.dataset_size,
            "metrics_count": len(self.metrics),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "metrics": {name: metric.score for name, metric in self.metrics.items()},
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            字典表示
        """
        return {
            "name": self.name,
            "dataset_size": self.dataset_size,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
        }

    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """保存评测结果

        Args:
            path: 保存路径
            format: 保存格式，支持 'json' 或 'csv'

        Raises:
            ValueError: 不支持的格式
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            self._save_json(path)
        elif format.lower() == "csv":
            self._save_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

        logger.info(f"Saved evaluation result to {path}")

    def _save_json(self, path: Path) -> None:
        """保存为JSON格式"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_csv(self, path: Path) -> None:
        """保存为CSV格式"""
        # 创建DataFrame
        rows = []
        for metric_name, metric in self.metrics.items():
            for i, score in enumerate(metric.scores):
                rows.append(
                    {
                        "evaluation_name": self.name,
                        "metric_name": metric_name,
                        "sample_index": i,
                        "score": score,
                        "timestamp": self.timestamp.isoformat(),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResult":
        """从文件加载评测结果

        Args:
            path: 文件路径

        Returns:
            EvaluationResult实例

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Loading evaluation result from {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 重建MetricResult对象
        metrics = {}
        for name, metric_data in data["metrics"].items():
            metrics[name] = MetricResult(
                name=metric_data["name"],
                score=metric_data["score"],
                scores=metric_data["scores"],
                metadata=metric_data.get("metadata"),
            )

        # 重建EvaluationResult对象
        return cls(
            name=data["name"],
            metrics=metrics,
            dataset_size=data["dataset_size"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata"),
        )

    def compare_with(self, other: "EvaluationResult") -> Dict[str, Any]:
        """与另一个评测结果对比

        Args:
            other: 另一个评测结果

        Returns:
            对比结果字典
        """
        common_metrics = set(self.metrics.keys()) & set(other.metrics.keys())
        if not common_metrics:
            return {
                "common_metrics": [],
                "comparison": {},
                "message": "No common metrics found",
            }

        comparison = {}
        for metric_name in common_metrics:
            self_score = self.get_score(metric_name)
            other_score = other.get_score(metric_name)
            if self_score is not None and other_score is not None:
                comparison[metric_name] = {
                    "self_score": self_score,
                    "other_score": other_score,
                    "difference": self_score - other_score,
                    "improvement": ((self_score - other_score) / other_score * 100)
                    if other_score != 0
                    else float("inf"),
                }

        return {
            "self_name": self.name,
            "other_name": other.name,
            "common_metrics": list(common_metrics),
            "comparison": comparison,
        }

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"EvaluationResult("
            f"name='{self.name}', "
            f"metrics={len(self.metrics)}, "
            f"dataset_size={self.dataset_size})"
        )
