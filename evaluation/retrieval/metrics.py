"""检索评估指标计算。

实现信息检索标准指标：File@K, Recall@K, MRR, NDCG, CategoryHit@K, SymbolHit@K。
所有指标函数均为纯函数，输入搜索结果列表和 Ground Truth，输出指标值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evaluation.datasets.schema import ExpectedFile, TestCase
from ubmc_rag.models.search_result import SearchResult

# ---------------------------------------------------------------------------
# 单条用例结果
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """单条测试用例的评估结果。

    Attributes:
        test_case_id: 测试用例 ID
        query: 查询文本
        category: 查询类别
        file_at_k: 各 K 值下的文件命中结果 {1: True, 3: False, ...}
        recall_at_k: 各 K 值下的召回率 {5: 0.6, 10: 1.0, ...}
        first_relevant_rank: 第一个相关结果的排名（1-based），无命中时为 None
        ndcg_at_k: 各 K 值下的 NDCG 分数 {5: 0.85, 10: 0.92, ...}
        category_hit: top-K 中是否包含期望仓库
        symbol_hit: top-K 中是否包含期望符号
    """

    test_case_id: str
    query: str
    category: str
    file_at_k: dict[int, bool] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    first_relevant_rank: int | None = None
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    category_hit: bool = False
    symbol_hit: bool = False


# ---------------------------------------------------------------------------
# 汇总指标
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """汇总检索评估指标。

    Attributes:
        file_at_1: File@1 文件命中率
        file_at_3: File@3 文件命中率
        file_at_5: File@5 文件命中率
        file_at_10: File@10 文件命中率
        recall_at_5: Recall@5 召回率
        recall_at_10: Recall@10 召回率
        mrr: Mean Reciprocal Rank
        ndcg_at_5: NDCG@5
        ndcg_at_10: NDCG@10
        category_hit_at_5: 仓库级命中率 (top-5)
        symbol_hit_at_5: 符号级命中率 (top-5)
        total_cases: 测试用例总数
        per_case_results: 每条用例的详细结果
    """

    file_at_1: float = 0.0
    file_at_3: float = 0.0
    file_at_5: float = 0.0
    file_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    category_hit_at_5: float = 0.0
    symbol_hit_at_5: float = 0.0
    total_cases: int = 0
    per_case_results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为 JSON 可序列化字典。"""
        return {
            "file_at_1": round(self.file_at_1, 4),
            "file_at_3": round(self.file_at_3, 4),
            "file_at_5": round(self.file_at_5, 4),
            "file_at_10": round(self.file_at_10, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "category_hit_at_5": round(self.category_hit_at_5, 4),
            "symbol_hit_at_5": round(self.symbol_hit_at_5, 4),
            "total_cases": self.total_cases,
        }


# ---------------------------------------------------------------------------
# 匹配辅助
# ---------------------------------------------------------------------------


def _match_key(expected: ExpectedFile) -> str:
    """生成期望文件的唯一匹配键（repo_name + file_path）。

    file_path 使用 '/' 分隔符统一格式。
    """
    return f"{expected.repo_name}:{expected.file_path}"


def _get_relevant_set(expected: list[ExpectedFile]) -> set[str]:
    """获取所有期望文件的匹配键集合。"""
    return {_match_key(e) for e in expected}


def _get_relevance_map(expected: list[ExpectedFile]) -> dict[str, int]:
    """获取期望文件的相关度映射。"""
    return {_match_key(e): e.relevance for e in expected}


def _result_key(result: SearchResult) -> str:
    """生成搜索结果的匹配键。

    支持后缀匹配：索引中的 file_path 可能包含 `data/repos/` 前缀，
    而数据集中的 file_path 是相对路径。此处做归一化处理。
    """
    repo = result.chunk.repo_name
    path = result.chunk.file_path
    # 剥离可能的 data/repos/ 前缀
    prefix = f"data/repos/{repo}/"
    if path.startswith(prefix):
        path = path[len(prefix) :]
    return f"{repo}:{path}"


def _is_match(result_key: str, expected_keys: set[str]) -> bool:
    """检查结果键是否匹配任一期望键。

    先精确匹配，再尝试后缀匹配（兼容不同路径前缀）。
    """
    if result_key in expected_keys:
        return True
    # 后缀匹配：期望路径可能是实际路径的后缀
    result_repo, result_path = result_key.split(":", 1)
    for ek in expected_keys:
        exp_repo, exp_path = ek.split(":", 1)
        if result_repo == exp_repo and result_path.endswith(exp_path):
            return True
    return False


def _find_relevance(result_key: str, rel_map: dict[str, int]) -> int:
    """从相关度映射中查找匹配的相关度等级。

    支持精确匹配和后缀匹配。
    """
    if result_key in rel_map:
        return rel_map[result_key]
    # 后缀匹配
    result_repo, result_path = result_key.split(":", 1)
    for ek, rel in rel_map.items():
        exp_repo, exp_path = ek.split(":", 1)
        if result_repo == exp_repo and result_path.endswith(exp_path):
            return rel
    return 0


# ---------------------------------------------------------------------------
# 单项指标函数
# ---------------------------------------------------------------------------


def file_at_k(
    results: list[SearchResult],
    expected: list[ExpectedFile],
    k: int,
) -> bool:
    """计算 File@K：top-K 结果中是否包含任一期望文件。

    Args:
        results: 搜索结果列表（已按分数降序排列）
        expected: 期望文件列表
        k: 截断位置

    Returns:
        True 如果 top-K 中包含至少一个期望文件
    """
    relevant = _get_relevant_set(expected)
    top_k_results = results[:k]
    return any(_is_match(_result_key(r), relevant) for r in top_k_results)


def recall_at_k(
    results: list[SearchResult],
    expected: list[ExpectedFile],
    k: int,
) -> float:
    """计算 Recall@K：top-K 中命中的期望文件占比。

    Args:
        results: 搜索结果列表
        expected: 期望文件列表
        k: 截断位置

    Returns:
        命中数 / 总期望数，无期望文件时返回 0.0
    """
    if not expected:
        return 0.0
    relevant = _get_relevant_set(expected)
    top_k_results = results[:k]
    found = sum(1 for r in top_k_results if _is_match(_result_key(r), relevant))
    return found / len(relevant)


def mrr_score(
    results: list[SearchResult],
    expected: list[ExpectedFile],
) -> float | None:
    """计算 MRR (Mean Reciprocal Rank)：第一个相关结果的排名倒数。

    Args:
        results: 搜索结果列表
        expected: 期望文件列表

    Returns:
        1/rank（1-based），无命中时返回 0.0，无期望文件时返回 None
    """
    if not expected:
        return None
    relevant = _get_relevant_set(expected)
    for i, result in enumerate(results):
        if _is_match(_result_key(result), relevant):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    results: list[SearchResult],
    expected: list[ExpectedFile],
    k: int,
) -> float:
    """计算 NDCG@K (Normalized Discounted Cumulative Gain)。

    使用期望文件中的 relevance 等级 (1/2/3) 作为增益值，
    对排序质量进行评估。

    Args:
        results: 搜索结果列表
        expected: 期望文件列表
        k: 截断位置

    Returns:
        NDCG 分数 [0, 1]，无期望文件时返回 0.0
    """
    if not expected:
        return 0.0

    rel_map = _get_relevance_map(expected)
    top_k_results = results[:k]

    # DCG@K
    dcg = 0.0
    for i, result in enumerate(top_k_results):
        key = _result_key(result)
        gain = _find_relevance(key, rel_map)
        if gain > 0:
            dcg += gain / math.log2(i + 2)  # i+2 因为 rank 从 1 开始，log2(rank+1)

    # IDCG@K：将所有期望文件按 relevance 降序排列，取前 K 个
    ideal_gains = sorted(rel_map.values(), reverse=True)[:k]
    idcg = 0.0
    for i, gain in enumerate(ideal_gains):
        idcg += gain / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def category_hit_at_k(
    results: list[SearchResult],
    expected_repos: list[str],
    k: int,
) -> bool:
    """计算 CategoryHit@K：top-K 中是否包含任一期望仓库。

    Args:
        results: 搜索结果列表
        expected_repos: 期望仓库名列表
        k: 截断位置

    Returns:
        True 如果 top-K 中包含至少一个期望仓库的结果
    """
    if not expected_repos:
        return True
    top_k_results = results[:k]
    repos_in_top_k = {r.chunk.repo_name for r in top_k_results}
    return any(repo in repos_in_top_k for repo in expected_repos)


def symbol_hit_at_k(
    results: list[SearchResult],
    expected_symbols: list[str],
    k: int,
) -> bool:
    """计算 SymbolHit@K：top-K 中是否包含任一期望符号。

    Args:
        results: 搜索结果列表
        expected_symbols: 期望符号名列表
        k: 截断位置

    Returns:
        True 如果 top-K 结果的符号中包含至少一个期望符号
    """
    if not expected_symbols:
        return True
    top_k_results = results[:k]
    for r in top_k_results:
        for symbol in r.chunk.symbols:
            if symbol.name in expected_symbols:
                return True
    return False


# ---------------------------------------------------------------------------
# 单条用例评估 + 批量汇总
# ---------------------------------------------------------------------------


def evaluate_case(
    results: list[SearchResult],
    test_case: TestCase,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> CaseResult:
    """对单条测试用例计算所有指标。

    Args:
        results: 搜索引擎返回的结果列表
        test_case: 回归测试用例
        ks: 需要计算的 K 值元组

    Returns:
        包含所有指标的 CaseResult
    """
    cr = CaseResult(
        test_case_id=test_case.id,
        query=test_case.query,
        category=test_case.category,
    )

    recall_ks = (5, 10)
    ndcg_ks = (5, 10)

    # File@K
    for k in ks:
        cr.file_at_k[k] = file_at_k(results, test_case.expected_files, k)

    # Recall@K
    for k in recall_ks:
        cr.recall_at_k[k] = recall_at_k(results, test_case.expected_files, k)

    # MRR
    cr.first_relevant_rank = mrr_score(results, test_case.expected_files)

    # NDCG@K
    for k in ndcg_ks:
        cr.ndcg_at_k[k] = ndcg_at_k(results, test_case.expected_files, k)

    # CategoryHit@5
    cr.category_hit = category_hit_at_k(
        results,
        test_case.expected_repos,
        k=5,
    )

    # SymbolHit@5
    cr.symbol_hit = symbol_hit_at_k(
        results,
        test_case.expected_symbols,
        k=5,
    )

    return cr


def compute_metrics(case_results: list[CaseResult]) -> RetrievalMetrics:
    """汇总所有用例结果为 RetrievalMetrics。

    Args:
        case_results: 单条用例评估结果列表

    Returns:
        汇总后的 RetrievalMetrics
    """
    if not case_results:
        return RetrievalMetrics()

    n = len(case_results)

    def avg_field(field_name: str, k: int) -> float:
        values = [getattr(cr, field_name).get(k, 0) for cr in case_results]
        numeric = [v for v in values if isinstance(v, (int, float))]
        return sum(numeric) / len(numeric) if numeric else 0.0

    # File@K
    fa1 = avg_field("file_at_k", 1)
    fa3 = avg_field("file_at_k", 3)
    fa5 = avg_field("file_at_k", 5)
    fa10 = avg_field("file_at_k", 10)

    # Recall@K
    r5 = avg_field("recall_at_k", 5)
    r10 = avg_field("recall_at_k", 10)

    # MRR
    mrr_values = [
        cr.first_relevant_rank for cr in case_results if cr.first_relevant_rank is not None
    ]
    avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0

    # NDCG@K
    n5 = avg_field("ndcg_at_k", 5)
    n10 = avg_field("ndcg_at_k", 10)

    # CategoryHit@5
    cat_hits = sum(1 for cr in case_results if cr.category_hit)
    avg_cat = cat_hits / n

    # SymbolHit@5
    sym_hits = sum(1 for cr in case_results if cr.symbol_hit)
    avg_sym = sym_hits / n

    return RetrievalMetrics(
        file_at_1=fa1,
        file_at_3=fa3,
        file_at_5=fa5,
        file_at_10=fa10,
        recall_at_5=r5,
        recall_at_10=r10,
        mrr=avg_mrr,
        ndcg_at_5=n5,
        ndcg_at_10=n10,
        category_hit_at_5=avg_cat,
        symbol_hit_at_5=avg_sym,
        total_cases=n,
        per_case_results=case_results,
    )
