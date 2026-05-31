"""检索质量回归测试和指标函数单元测试。

回归测试使用 evaluation 框架的 regression_v1 数据集，
验证检索指标不低于基线值，防止检索质量退化。

单元测试验证指标计算的正确性，包括边界情况和新增指标。
"""

from __future__ import annotations

import pytest

from evaluation.datasets.loader import load_dataset
from evaluation.retrieval.evaluator import RetrievalEvaluator
from ubmc_rag.config.settings import AppConfig

# 数据集和配置路径
_DATASET_PATH = "evaluation/datasets/regression_v1.yaml"
_CONFIG_PATH = "config/default_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """加载应用配置。"""
    return AppConfig.from_yaml(_CONFIG_PATH)


@pytest.fixture(scope="module")
def dataset():
    """加载回归测试数据集。"""
    return load_dataset(_DATASET_PATH)


@pytest.fixture(scope="module")
def evaluator(config):
    """创建检索评估器。如果索引不存在则跳过所有测试。"""
    try:
        return RetrievalEvaluator(config)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def metrics(evaluator, dataset):
    """运行完整检索评估（hybrid_reranked 模式），缓存结果。"""
    return evaluator.evaluate(dataset, search_mode="hybrid_reranked")


# ---------------------------------------------------------------------------
# 基线回归测试
# ---------------------------------------------------------------------------


class TestRetrievalBaseline:
    """检索质量基线测试。确保核心指标不低于阈值。"""

    def test_file_at_5_above_baseline(self, metrics):
        """File@5 应 >= 0.40（JSON 启用 + BM25 修复 + file_path 归一化后）。"""
        assert metrics.file_at_5 >= 0.40, f"File@5 = {metrics.file_at_5:.4f}"

    def test_file_at_10_above_baseline(self, metrics):
        """File@10 应 >= 0.45。"""
        assert metrics.file_at_10 >= 0.45, f"File@10 = {metrics.file_at_10:.4f}"

    def test_mrr_above_baseline(self, metrics):
        """MRR 应 >= 0.30。"""
        assert metrics.mrr >= 0.30, f"MRR = {metrics.mrr:.4f}"

    def test_category_hit_above_baseline(self, metrics):
        """CategoryHit@5 应 >= 0.70。"""
        assert metrics.category_hit_at_5 >= 0.70, f"CategoryHit@5 = {metrics.category_hit_at_5:.4f}"

    def test_ndcg_at_5_above_baseline(self, metrics):
        """NDCG@5 应 >= 0.40。"""
        assert metrics.ndcg_at_5 >= 0.40, f"NDCG@5 = {metrics.ndcg_at_5:.4f}"

    def test_symbol_hit_at_5_above_baseline(self, metrics):
        """SymbolHit@5 应 >= 0.70。"""
        assert metrics.symbol_hit_at_5 >= 0.70, f"SymbolHit@5 = {metrics.symbol_hit_at_5:.4f}"

    def test_map_above_baseline(self, metrics):
        """MAP 应 >= 0.20。"""
        assert metrics.map_score >= 0.20, f"MAP = {metrics.map_score:.4f}"

    def test_precision_at_5_above_baseline(self, metrics):
        """Precision@5 应 >= 0.08。"""
        assert metrics.precision_at_5 >= 0.08, f"Precision@5 = {metrics.precision_at_5:.4f}"

    def test_dataset_has_expected_size(self, dataset):
        """数据集应包含 50 条用例。"""
        assert len(dataset.test_cases) == 50

    def test_breakdown_populated(self, metrics):
        """分组统计应包含所有类别。"""
        assert "single_function" in metrics.by_category
        assert "single_component" in metrics.by_category
        assert "cross_component" in metrics.by_category
        assert len(metrics.by_category) == 3

    def test_confidence_intervals_populated(self, metrics):
        """50 条用例应产生置信区间。"""
        assert len(metrics.confidence_intervals) > 0
        assert "file_at_5" in metrics.confidence_intervals
        lo, hi = metrics.confidence_intervals["file_at_5"]
        assert lo <= hi


# ---------------------------------------------------------------------------
# 单元测试：指标函数
# ---------------------------------------------------------------------------


class TestMetricFunctions:
    """测试指标计算函数的正确性（使用构造数据）。"""

    def _make_result(self, repo_name: str, file_path: str, symbols=None):
        """构造 Mock 搜索结果。"""
        from unittest.mock import MagicMock

        r = MagicMock()
        r.chunk.repo_name = repo_name
        r.chunk.file_path = file_path
        r.chunk.symbols = symbols or []
        return r

    def test_file_at_k_hit(self):
        """top-K 包含期望文件时应返回 True。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import file_at_k

        results = [
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("libipmi", "src/ipmi_sensor.c"),
        ]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert file_at_k(results, expected, k=1) is True
        assert file_at_k(results, expected, k=5) is True

    def test_file_at_k_miss(self):
        """top-K 不包含期望文件时应返回 False。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import file_at_k

        results = [self._make_result("sensor", "src/lualib/sensor_service.lua")]
        expected = [ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c")]

        assert file_at_k(results, expected, k=1) is False

    def test_recall_at_k_deduplication(self):
        """同一文件的多个 chunk 不应导致 recall > 1.0。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import recall_at_k

        results = [
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
        ]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        recall = recall_at_k(results, expected, k=5)
        assert recall == 1.0, f"recall should be 1.0 but got {recall}"

    def test_recall_at_k_partial(self):
        """部分命中时应返回正确比例。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import recall_at_k

        results = [self._make_result("sensor", "src/lualib/sensor_service.lua")]
        expected = [
            ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua"),
            ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c"),
        ]

        assert recall_at_k(results, expected, k=5) == 0.5

    def test_recall_at_k_two_expected_two_results(self):
        """两个期望文件，两个结果分别命中时应返回 1.0。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import recall_at_k

        results = [
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("libipmi", "src/ipmi_sensor.c"),
        ]
        expected = [
            ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua"),
            ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c"),
        ]

        assert recall_at_k(results, expected, k=5) == 1.0

    def test_precision_at_k_basic(self):
        """Precision@K 应正确计算相关结果占比。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import precision_at_k

        results = [
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("libipmi", "src/ipmi_sensor.c"),
            self._make_result("sensor", "src/lualib/other.lua"),
        ]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert precision_at_k(results, expected, k=1) == 1.0
        assert precision_at_k(results, expected, k=3) == pytest.approx(1.0 / 3)

    def test_precision_at_k_deduplication(self):
        """同一文件的多个 chunk 只计一次。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import precision_at_k

        results = [
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
        ]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert precision_at_k(results, expected, k=2) == 0.5

    def test_average_precision_perfect(self):
        """所有相关结果排在最前面时 AP = 1.0。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import average_precision

        results = [
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("libipmi", "src/ipmi_sensor.c"),
        ]
        expected = [
            ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua"),
            ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c"),
        ]

        assert average_precision(results, expected) == 1.0

    def test_average_precision_partial(self):
        """部分命中时 AP 应介于 0 和 1 之间。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import average_precision

        results = [
            self._make_result("other", "src/other.lua"),
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
            self._make_result("libipmi", "src/ipmi_sensor.c"),
        ]
        expected = [
            ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua"),
            ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c"),
        ]

        ap = average_precision(results, expected)
        assert 0.0 < ap < 1.0

    def test_average_precision_no_match(self):
        """无匹配时 AP = 0.0。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import average_precision

        results = [self._make_result("other", "src/other.lua")]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert average_precision(results, expected) == 0.0

    def test_mrr_first_rank(self):
        """第一个结果命中时 MRR = 1.0。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import mrr_score

        results = [self._make_result("sensor", "src/lualib/sensor_service.lua")]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert mrr_score(results, expected) == 1.0

    def test_mrr_second_rank(self):
        """第二个结果命中时 MRR = 0.5。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import mrr_score

        results = [
            self._make_result("libipmi", "src/ipmi_sensor.c"),
            self._make_result("sensor", "src/lualib/sensor_service.lua"),
        ]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert mrr_score(results, expected) == 0.5

    def test_ndcg_perfect_ranking(self):
        """完美排序时 NDCG = 1.0。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import ndcg_at_k

        results = [self._make_result("sensor", "src/lualib/sensor_service.lua")]
        expected = [
            ExpectedFile(
                repo_name="sensor",
                file_path="src/lualib/sensor_service.lua",
                relevance=3,
            )
        ]

        assert ndcg_at_k(results, expected, k=5) == 1.0

    def test_empty_expected_returns_zero_or_none(self):
        """无期望文件时各指标应返回安全默认值。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import (
            average_precision,
            ndcg_at_k,
            precision_at_k,
            recall_at_k,
        )

        results = [self._make_result("sensor", "src/lualib/sensor_service.lua")]
        expected: list[ExpectedFile] = []

        assert recall_at_k(results, expected, k=5) == 0.0
        assert precision_at_k(results, expected, k=5) == 0.0
        assert average_precision(results, expected) == 0.0
        assert ndcg_at_k(results, expected, k=5) == 0.0

    def test_empty_results(self):
        """无搜索结果时各指标应返回安全默认值。"""
        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import (
            average_precision,
            file_at_k,
            mrr_score,
            ndcg_at_k,
            precision_at_k,
            recall_at_k,
        )

        results = []
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert file_at_k(results, expected, k=5) is False
        assert recall_at_k(results, expected, k=5) == 0.0
        assert precision_at_k(results, expected, k=5) == 0.0
        assert mrr_score(results, expected) == 0.0
        assert average_precision(results, expected) == 0.0
        assert ndcg_at_k(results, expected, k=5) == 0.0


class TestComputeMetrics:
    """测试 compute_metrics 汇总函数。"""

    def test_empty_case_results(self):
        """空列表应返回零值指标。"""
        from evaluation.retrieval.metrics import compute_metrics

        metrics = compute_metrics([], enable_bootstrap=False)
        assert metrics.total_cases == 0
        assert metrics.file_at_5 == 0.0
        assert metrics.map_score == 0.0

    def test_breakdown_keys(self):
        """分组统计应正确按字段分组。"""
        from evaluation.retrieval.metrics import CaseResult, compute_metrics

        cases = [
            CaseResult(
                test_case_id="TC-1", query="q1", category="single_function",
                difficulty="easy", query_type="exact_match",
                file_at_k={5: True},
            ),
            CaseResult(
                test_case_id="TC-2", query="q2", category="cross_component",
                difficulty="hard", query_type="semantic_match",
                file_at_k={5: False},
            ),
        ]

        metrics = compute_metrics(cases, enable_bootstrap=False)
        assert "single_function" in metrics.by_category
        assert "cross_component" in metrics.by_category
        assert "easy" in metrics.by_difficulty
        assert "hard" in metrics.by_difficulty

    def test_confidence_intervals_disabled_for_few_cases(self):
        """少于 5 条用例不应计算置信区间。"""
        from evaluation.retrieval.metrics import CaseResult, compute_metrics

        cases = [
            CaseResult(test_case_id="TC-1", query="q", category="cat"),
        ]
        metrics = compute_metrics(cases)
        assert len(metrics.confidence_intervals) == 0

    def test_confidence_intervals_for_enough_cases(self):
        """5+ 条用例应计算置信区间。"""
        from evaluation.retrieval.metrics import CaseResult, compute_metrics

        cases = [
            CaseResult(
                test_case_id=f"TC-{i}", query="q", category="cat",
                file_at_k={5: i % 2 == 0},
            )
            for i in range(10)
        ]
        metrics = compute_metrics(cases)
        assert "file_at_5" in metrics.confidence_intervals
        lo, hi = metrics.confidence_intervals["file_at_5"]
        assert 0.0 <= lo <= hi <= 1.0
