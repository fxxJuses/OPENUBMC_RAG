"""检索评估指标计算。

实现信息检索标准指标：File@K, Precision@K, Recall@K, MRR, MAP, NDCG,
CategoryHit@K, SymbolHit@K。
支持按类别/难度/查询类型分组统计和 Bootstrap 置信区间。
所有指标函数均为纯函数，输入搜索结果列表和 Ground Truth，输出指标值。
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
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
        difficulty: 难度等级
        query_type: 查询类型
        file_at_k: 各 K 值下的文件命中结果 {1: True, 3: False, ...}
        precision_at_k: 各 K 值下的精确率 {5: 0.6, 10: 0.3, ...}
        recall_at_k: 各 K 值下的召回率 {5: 0.6, 10: 1.0, ...}
        first_relevant_rank: 第一个相关结果的排名（1-based），无命中时为 None
        average_precision: 平均精确率 (AP)
        ndcg_at_k: 各 K 值下的 NDCG 分数 {5: 0.85, 10: 0.92, ...}
        category_hit: top-K 中是否包含期望仓库
        symbol_hit: top-K 中是否包含期望符号
    """

    test_case_id: str
    query: str
    category: str
    difficulty: str = "normal"
    query_type: str = "semantic_match"
    file_at_k: dict[int, bool] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    first_relevant_rank: int | None = None
    average_precision: float = 0.0
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
        precision_at_5: Precision@5 精确率
        precision_at_10: Precision@10 精确率
        recall_at_5: Recall@5 召回率
        recall_at_10: Recall@10 召回率
        mrr: Mean Reciprocal Rank
        map_score: Mean Average Precision
        ndcg_at_5: NDCG@5
        ndcg_at_10: NDCG@10
        category_hit_at_5: 仓库级命中率 (top-5)
        symbol_hit_at_5: 符号级命中率 (top-5)
        total_cases: 测试用例总数
        per_case_results: 每条用例的详细结果
        confidence_intervals: Bootstrap 95% 置信区间
        by_category: 按查询类别分组的指标
        by_difficulty: 按难度等级分组的指标
        by_query_type: 按查询类型分组的指标
    """

    file_at_1: float = 0.0
    file_at_3: float = 0.0
    file_at_5: float = 0.0
    file_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    category_hit_at_5: float = 0.0
    symbol_hit_at_5: float = 0.0
    total_cases: int = 0
    per_case_results: list[CaseResult] = field(default_factory=list)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    by_category: dict[str, RetrievalMetrics] = field(default_factory=dict)
    by_difficulty: dict[str, RetrievalMetrics] = field(default_factory=dict)
    by_query_type: dict[str, RetrievalMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为 JSON 可序列化字典。"""
        result = {
            "file_at_1": round(self.file_at_1, 4),
            "file_at_3": round(self.file_at_3, 4),
            "file_at_5": round(self.file_at_5, 4),
            "file_at_10": round(self.file_at_10, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "category_hit_at_5": round(self.category_hit_at_5, 4),
            "symbol_hit_at_5": round(self.symbol_hit_at_5, 4),
            "total_cases": self.total_cases,
        }
        if self.confidence_intervals:
            result["confidence_intervals_95"] = {
                k: (round(lo, 4), round(hi, 4))
                for k, (lo, hi) in self.confidence_intervals.items()
            }
        if self.by_category:
            result["by_category"] = {
                k: v.to_dict() for k, v in self.by_category.items()
            }
        if self.by_difficulty:
            result["by_difficulty"] = {
                k: v.to_dict() for k, v in self.by_difficulty.items()
            }
        if self.by_query_type:
            result["by_query_type"] = {
                k: v.to_dict() for k, v in self.by_query_type.items()
            }
        return result


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

    归一化 file_path：剥离可能的 data/repos/{repo}/ 前缀，
    确保与数据集中的相对路径格式一致。
    """
    repo = result.chunk.repo_name
    path = result.chunk.file_path
    prefix = f"data/repos/{repo}/"
    if path.startswith(prefix):
        path = path[len(prefix):]
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


def _find_matching_expected(result_key: str, expected_keys: set[str]) -> str | None:
    """查找结果键匹配的期望键，返回匹配到的期望键或 None。"""
    if result_key in expected_keys:
        return result_key
    result_repo, result_path = result_key.split(":", 1)
    for ek in expected_keys:
        exp_repo, exp_path = ek.split(":", 1)
        if result_repo == exp_repo and result_path.endswith(exp_path):
            return ek
    return None


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
    """计算 File@K：top-K 结果中是否包含任一期望文件。"""
    relevant = _get_relevant_set(expected)
    top_k_results = results[:k]
    return any(_is_match(_result_key(r), relevant) for r in top_k_results)


def precision_at_k(
    results: list[SearchResult],
    expected: list[ExpectedFile],
    k: int,
) -> float:
    """计算 Precision@K：top-K 中相关结果的比例（按唯一文件去重）。

    Returns:
        命中的唯一期望文件数 / K，K=0 时返回 0.0
    """
    if k == 0:
        return 0.0
    relevant = _get_relevant_set(expected)
    top_k_results = results[:k]
    matched = set()
    for r in top_k_results:
        rkey = _result_key(r)
        ek = _find_matching_expected(rkey, relevant)
        if ek is not None:
            matched.add(ek)
    return len(matched) / k


def recall_at_k(
    results: list[SearchResult],
    expected: list[ExpectedFile],
    k: int,
) -> float:
    """计算 Recall@K：top-K 中命中的唯一期望文件占比。

    修复：按唯一期望文件去重，确保 recall 不超过 1.0。

    Returns:
        命中的唯一期望文件数 / 总期望数，无期望文件时返回 0.0
    """
    if not expected:
        return 0.0
    relevant = _get_relevant_set(expected)
    top_k_results = results[:k]
    matched = set()
    for r in top_k_results:
        rkey = _result_key(r)
        ek = _find_matching_expected(rkey, relevant)
        if ek is not None:
            matched.add(ek)
    return len(matched) / len(relevant)


def mrr_score(
    results: list[SearchResult],
    expected: list[ExpectedFile],
) -> float | None:
    """计算 MRR (Mean Reciprocal Rank)：第一个相关结果的排名倒数。

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


def average_precision(
    results: list[SearchResult],
    expected: list[ExpectedFile],
) -> float:
    """计算 Average Precision (AP)：所有相关位置的精确率均值。

    对每个排名位置，如果该位置的结果相关，则计算 Precision@rank，
    最终取所有相关位置的 Precision 均值。

    Returns:
        AP 分数 [0, 1]，无期望文件时返回 0.0
    """
    if not expected:
        return 0.0
    relevant = _get_relevant_set(expected)
    matched = set()
    precisions_at_hits = []
    for i, r in enumerate(results):
        rkey = _result_key(r)
        ek = _find_matching_expected(rkey, relevant)
        if ek is not None and ek not in matched:
            matched.add(ek)
            precisions_at_hits.append(len(matched) / (i + 1))
    if not precisions_at_hits:
        return 0.0
    return sum(precisions_at_hits) / len(relevant)


def ndcg_at_k(
    results: list[SearchResult],
    expected: list[ExpectedFile],
    k: int,
) -> float:
    """计算 NDCG@K (Normalized Discounted Cumulative Gain)。

    使用期望文件中的 relevance 等级 (1/2/3) 作为增益值，
    对排序质量进行评估。

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
    """计算 CategoryHit@K：top-K 中是否包含任一期望仓库。"""
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
    """计算 SymbolHit@K：top-K 中是否包含任一期望符号。"""
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
    """对单条测试用例计算所有指标。"""
    cr = CaseResult(
        test_case_id=test_case.id,
        query=test_case.query,
        category=test_case.category,
        difficulty=test_case.difficulty,
        query_type=test_case.query_type,
    )

    precision_ks = (5, 10)
    recall_ks = (5, 10)
    ndcg_ks = (5, 10)

    # File@K
    for k in ks:
        cr.file_at_k[k] = file_at_k(results, test_case.expected_files, k)

    # Precision@K
    for k in precision_ks:
        cr.precision_at_k[k] = precision_at_k(results, test_case.expected_files, k)

    # Recall@K
    for k in recall_ks:
        cr.recall_at_k[k] = recall_at_k(results, test_case.expected_files, k)

    # MRR
    cr.first_relevant_rank = mrr_score(results, test_case.expected_files)

    # AP
    cr.average_precision = average_precision(results, test_case.expected_files)

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


def compute_metrics(
    case_results: list[CaseResult],
    enable_bootstrap: bool = True,
    bootstrap_samples: int = 1000,
) -> RetrievalMetrics:
    """汇总所有用例结果为 RetrievalMetrics。

    包括按类别/难度/查询类型分组统计和 Bootstrap 置信区间。
    """
    if not case_results:
        return RetrievalMetrics()

    n = len(case_results)
    metrics = _compute_flat_metrics(case_results)

    # 分组统计
    metrics.by_category = _compute_breakdown(case_results, "category")
    metrics.by_difficulty = _compute_breakdown(case_results, "difficulty")
    metrics.by_query_type = _compute_breakdown(case_results, "query_type")

    # Bootstrap 置信区间
    if enable_bootstrap and n >= 5:
        metrics.confidence_intervals = compute_confidence_intervals(
            case_results, num_samples=bootstrap_samples,
        )

    return metrics


def _compute_flat_metrics(case_results: list[CaseResult]) -> RetrievalMetrics:
    """计算全局指标（不包含分组和置信区间，避免递归）。"""
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

    # Precision@K
    p5 = avg_field("precision_at_k", 5)
    p10 = avg_field("precision_at_k", 10)

    # Recall@K
    r5 = avg_field("recall_at_k", 5)
    r10 = avg_field("recall_at_k", 10)

    # MRR
    mrr_values = [
        cr.first_relevant_rank for cr in case_results if cr.first_relevant_rank is not None
    ]
    avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0

    # MAP
    avg_ap = sum(cr.average_precision for cr in case_results) / n if n > 0 else 0.0

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
        precision_at_5=p5,
        precision_at_10=p10,
        recall_at_5=r5,
        recall_at_10=r10,
        mrr=avg_mrr,
        map_score=avg_ap,
        ndcg_at_5=n5,
        ndcg_at_10=n10,
        category_hit_at_5=avg_cat,
        symbol_hit_at_5=avg_sym,
        total_cases=n,
        per_case_results=case_results,
    )


def _compute_breakdown(
    case_results: list[CaseResult],
    group_field: str,
) -> dict[str, RetrievalMetrics]:
    """按指定字段分组计算指标。"""
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for cr in case_results:
        key = getattr(cr, group_field, "unknown")
        groups[key].append(cr)
    return {
        key: _compute_flat_metrics(cases)
        for key, cases in sorted(groups.items())
    }


# ---------------------------------------------------------------------------
# Bootstrap 置信区间
# ---------------------------------------------------------------------------


def compute_confidence_intervals(
    case_results: list[CaseResult],
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """使用 Bootstrap 方法计算指标的置信区间。

    对 case_results 进行有放回重采样，对每次采样计算指标，
    取 alpha/2 和 1-alpha/2 分位数作为置信区间。

    Args:
        case_results: 单条用例评估结果列表
        num_samples: Bootstrap 重采样次数
        confidence: 置信水平（默认 0.95）
        seed: 随机种子（确保可复现）

    Returns:
        指标名到 (下界, 上界) 的映射
    """
    rng = random.Random(seed)
    n = len(case_results)

    # 收集每次采样的指标
    file_at_5_samples = []
    mrr_samples = []
    map_samples = []
    ndcg_at_5_samples = []
    recall_at_5_samples = []
    precision_at_5_samples = []

    for _ in range(num_samples):
        sample = rng.choices(case_results, k=n)
        m = _compute_flat_metrics(sample)
        file_at_5_samples.append(m.file_at_5)
        mrr_samples.append(m.mrr)
        map_samples.append(m.map_score)
        ndcg_at_5_samples.append(m.ndcg_at_5)
        recall_at_5_samples.append(m.recall_at_5)
        precision_at_5_samples.append(m.precision_at_5)

    alpha = 1.0 - confidence
    lo_idx = int(num_samples * alpha / 2)
    hi_idx = int(num_samples * (1 - alpha / 2))

    def ci(values: list[float]) -> tuple[float, float]:
        sorted_vals = sorted(values)
        return (sorted_vals[lo_idx], sorted_vals[hi_idx])

    return {
        "file_at_5": ci(file_at_5_samples),
        "mrr": ci(mrr_samples),
        "map": ci(map_samples),
        "ndcg_at_5": ci(ndcg_at_5_samples),
        "recall_at_5": ci(recall_at_5_samples),
        "precision_at_5": ci(precision_at_5_samples),
    }
