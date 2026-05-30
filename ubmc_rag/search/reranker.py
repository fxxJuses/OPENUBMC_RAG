"""搜索结果重排序器，应用提升规则和多样性过滤。

对混合搜索的初步结果进行二次排序，通过以下策略提升结果质量：
1. 符号名精确匹配提升
2. 文件路径匹配提升
3. MDS 模型类名匹配提升
4. 同文件结果多样性降权（防止同一文件占据过多结果）
"""

from __future__ import annotations

from ubmc_rag.config.settings import SearchConfig
from ubmc_rag.models.search_result import SearchResult


class Reranker:
    """搜索结果重排序器，应用多维度提升规则和多样性过滤。

    Attributes:
        config: 搜索配置，包含各提升规则的倍数参数
    """

    def __init__(self, config: SearchConfig):
        self.config = config

    def rerank(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """对搜索结果应用重排序规则。

        处理步骤：
        1. 应用符号名、文件路径、MDS 模型匹配提升
        2. 按提升后分数重新排序
        3. 同一文件的重复结果降权（超过 diversity_max_per_file 的结果分数 ×0.7）

        Args:
            results: 待重排序的搜索结果列表
            query: 原始查询文本

        Returns:
            重排序后的搜索结果列表
        """
        if not results:
            return results

        boosted = []
        for r in results:
            score = r.score
            query_lower = query.lower()

            # 符号名精确匹配提升
            for sym in r.chunk.symbols:
                if sym.name.lower() in query_lower:
                    score *= self.config.symbol_match_boost
                    break

            # 文件路径匹配提升
            query_parts = [p for p in query_lower.split() if len(p) > 2]
            if any(part in r.chunk.file_path.lower() for part in query_parts):
                score *= self.config.filepath_match_boost

            # MDS 模型类名匹配提升
            mds_class = r.chunk.metadata.get("mds_class", "")
            if mds_class and mds_class.lower() in query_lower:
                score *= self.config.mds_model_match_boost

            boosted.append(SearchResult(
                chunk=r.chunk,
                score=score,
                source=r.source,
            ))

        # 按提升后分数降序排列
        boosted.sort(key=lambda x: x.score, reverse=True)

        # 多样性过滤：同一文件的超出部分降权
        filtered = []
        file_counts: dict[str, int] = {}
        for r in boosted:
            key = r.chunk.file_path
            count = file_counts.get(key, 0)
            if count >= self.config.diversity_max_per_file:
                r = SearchResult(
                    chunk=r.chunk,
                    score=r.score * 0.7,
                    source=r.source,
                )
            filtered.append(r)
            file_counts[key] = count + 1

        return filtered
