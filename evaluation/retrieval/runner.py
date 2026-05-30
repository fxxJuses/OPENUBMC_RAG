"""多模式检索对比 Runner。

遍历四种搜索模式（bm25_only, dense_only, hybrid, hybrid_reranked），
对比各模式的检索指标，用于 A/B 实验验证。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from evaluation.datasets.schema import RegressionDataset
from evaluation.retrieval.evaluator import RetrievalEvaluator
from evaluation.retrieval.metrics import RetrievalMetrics
from ubmc_rag.config.settings import AppConfig

logger = logging.getLogger(__name__)

ALL_MODES = ["bm25_only", "dense_only", "hybrid", "hybrid_reranked"]


@dataclass
class ComparisonResult:
    """多模式对比结果。

    Attributes:
        configurations: 模式名到指标的映射
        dataset_name: 数据集名称
        total_cases: 用例总数
    """

    configurations: dict[str, RetrievalMetrics] = field(default_factory=dict)
    dataset_name: str = ""
    total_cases: int = 0

    def to_dict(self) -> dict:
        """转换为 JSON 可序列化字典。"""
        return {
            "dataset_name": self.dataset_name,
            "total_cases": self.total_cases,
            "configurations": {
                name: metrics.to_dict() for name, metrics in self.configurations.items()
            },
        }


class RetrievalRunner:
    """多模式检索对比 Runner。

    对同一个数据集运行多种搜索模式，输出对比结果。
    """

    def run_comparison(
        self,
        config: AppConfig,
        dataset: RegressionDataset,
        modes: list[str] | None = None,
        top_k: int = 10,
    ) -> ComparisonResult:
        """运行多模式对比评估。

        Args:
            config: 应用配置
            dataset: 回归测试数据集
            modes: 要评估的模式列表，默认为全部四种
            top_k: 搜索返回的最大结果数

        Returns:
            包含各模式指标的对比结果
        """
        if modes is None:
            modes = ALL_MODES

        evaluator = RetrievalEvaluator(config)
        result = ComparisonResult(
            dataset_name=dataset.name,
            total_cases=len(dataset.test_cases),
        )

        for mode in modes:
            logger.info("Running evaluation in mode: %s", mode)
            metrics = evaluator.evaluate(dataset, top_k=top_k, search_mode=mode)
            result.configurations[mode] = metrics

        return result
