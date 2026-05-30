"""检索质量回归测试。

使用 evaluation 框架的 regression_v1 数据集，
验证检索指标不低于基线值，防止检索质量退化。

这些测试需要预先构建的索引（运行 `ubmc-rag index`），
如果索引不存在则自动跳过。
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
        """File@5 应 ≥ 0.50（至少 50% 的查询在 top-5 中命中期望文件）。

        注意：当前 8/30 用例因 ChromaDB where 过滤 bug 被跳过，
        实际值会随修复提升。
        """
        assert metrics.file_at_5 >= 0.50, f"File@5 = {metrics.file_at_5:.4f}, below baseline 0.50"

    def test_file_at_10_above_baseline(self, metrics):
        """File@10 应 ≥ 0.50。"""
        assert metrics.file_at_10 >= 0.50, (
            f"File@10 = {metrics.file_at_10:.4f}, below baseline 0.50"
        )

    def test_mrr_above_baseline(self, metrics):
        """MRR 应 ≥ 0.40（首个相关结果的平均排名倒数）。"""
        assert metrics.mrr >= 0.40, f"MRR = {metrics.mrr:.4f}, below baseline 0.40"

    def test_category_hit_above_baseline(self, metrics):
        """CategoryHit@5 应 ≥ 0.70（组件级命中率）。"""
        assert metrics.category_hit_at_5 >= 0.70, (
            f"CategoryHit@5 = {metrics.category_hit_at_5:.4f}, below baseline 0.70"
        )

    def test_ndcg_at_5_above_baseline(self, metrics):
        """NDCG@5 应 ≥ 0.30。"""
        assert metrics.ndcg_at_5 >= 0.30, f"NDCG@5 = {metrics.ndcg_at_5:.4f}, below baseline 0.30"


# ---------------------------------------------------------------------------
# 单元测试：指标函数
# ---------------------------------------------------------------------------


class TestMetricFunctions:
    """测试指标计算函数的正确性（使用构造数据）。"""

    def test_file_at_k_hit(self):
        """top-K 包含期望文件时应返回 True。"""
        from unittest.mock import MagicMock

        from evaluation.datasets.schema import ExpectedFile

        # 构造搜索结果
        r1 = MagicMock()
        r1.chunk.repo_name = "sensor"
        r1.chunk.file_path = "src/lualib/sensor_service.lua"

        r2 = MagicMock()
        r2.chunk.repo_name = "libipmi"
        r2.chunk.file_path = "src/ipmi_sensor.c"

        results = [r1, r2]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        from evaluation.retrieval.metrics import file_at_k

        assert file_at_k(results, expected, k=1) is True
        assert file_at_k(results, expected, k=5) is True

    def test_file_at_k_miss(self):
        """top-K 不包含期望文件时应返回 False。"""
        from unittest.mock import MagicMock

        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import file_at_k

        r1 = MagicMock()
        r1.chunk.repo_name = "sensor"
        r1.chunk.file_path = "src/lualib/sensor_service.lua"

        results = [r1]
        expected = [ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c")]

        assert file_at_k(results, expected, k=1) is False

    def test_recall_at_k_partial(self):
        """部分命中时应返回正确比例。"""
        from unittest.mock import MagicMock

        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import recall_at_k

        r1 = MagicMock()
        r1.chunk.repo_name = "sensor"
        r1.chunk.file_path = "src/lualib/sensor_service.lua"

        results = [r1]
        expected = [
            ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua"),
            ExpectedFile(repo_name="libipmi", file_path="src/ipmi_sensor.c"),
        ]

        assert recall_at_k(results, expected, k=5) == 0.5

    def test_mrr_first_rank(self):
        """第一个结果命中时 MRR = 1.0。"""
        from unittest.mock import MagicMock

        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import mrr_score

        r1 = MagicMock()
        r1.chunk.repo_name = "sensor"
        r1.chunk.file_path = "src/lualib/sensor_service.lua"

        results = [r1]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert mrr_score(results, expected) == 1.0

    def test_mrr_second_rank(self):
        """第二个结果命中时 MRR = 0.5。"""
        from unittest.mock import MagicMock

        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import mrr_score

        r1 = MagicMock()
        r1.chunk.repo_name = "libipmi"
        r1.chunk.file_path = "src/ipmi_sensor.c"

        r2 = MagicMock()
        r2.chunk.repo_name = "sensor"
        r2.chunk.file_path = "src/lualib/sensor_service.lua"

        results = [r1, r2]
        expected = [ExpectedFile(repo_name="sensor", file_path="src/lualib/sensor_service.lua")]

        assert mrr_score(results, expected) == 0.5

    def test_ndcg_perfect_ranking(self):
        """完美排序时 NDCG = 1.0。"""
        from unittest.mock import MagicMock

        from evaluation.datasets.schema import ExpectedFile
        from evaluation.retrieval.metrics import ndcg_at_k

        r1 = MagicMock()
        r1.chunk.repo_name = "sensor"
        r1.chunk.file_path = "src/lualib/sensor_service.lua"

        results = [r1]
        expected = [
            ExpectedFile(
                repo_name="sensor",
                file_path="src/lualib/sensor_service.lua",
                relevance=3,
            )
        ]

        assert ndcg_at_k(results, expected, k=5) == 1.0
