"""
搜索结果重排序器，应用提升规则和多样性过滤。

对混合搜索的初步结果进行二次排序，通过以下策略提升结果质量：
1. 符号名精确匹配提升
2. 仓库名匹配提升（新增）
3. 文件路径匹配提升（改进：支持中英文混合）
4. MDS 模型类名匹配提升
5. 同文件结果多样性降权（防止同一文件占据过多结果）
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
        1. 应用符号名、仓库名、文件路径、MDS 模型匹配提升
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

        query_lower = query.lower()

        # 提取查询中的标识符 token（用于精确匹配）
        query_tokens = set()
        import re
        for token in re.findall(r'[a-zA-Z_]\w*', query_lower):
            query_tokens.add(token)
        # 也添加短 token（如 "c", "h", "ip" 等可能出现在文件路径中）
        for token in re.findall(r'\b[a-zA-Z_]{1,2}\b', query_lower):
            query_tokens.add(token)

        # 提取内容关键词 token（用于 content_keyword_boost，长度 ≥ 2 的词）
        content_keywords = [w for w in re.findall(r'\w+', query_lower) if len(w) >= 2]

        boosted = []
        for r in results:
            score = r.score

            # 1. 符号名精确匹配提升
            for sym in r.chunk.symbols:
                sym_lower = sym.name.lower()
                if sym_lower in query_lower or any(
                    t in sym_lower for t in query_tokens if len(t) >= 3
                ):
                    score *= self.config.symbol_match_boost
                    break

            # 2. 仓库名匹配提升
            repo_lower = r.chunk.repo_name.lower()
            if repo_lower in query_lower:
                score *= self.config.filepath_match_boost
            elif any(token in repo_lower for token in query_tokens if len(token) >= 3):
                score *= self.config.filepath_match_boost * 0.8  # 部分匹配打 8 折

            # 3. 文件路径匹配提升
            file_path_lower = r.chunk.file_path.lower()
            if file_path_lower in query_lower:
                score *= self.config.filepath_match_boost
            else:
                # 检查路径中的每个部分是否匹配查询 token
                path_parts = set()
                for part in re.split(r'[/_.-]', file_path_lower):
                    if part:
                        path_parts.add(part)
                matched_parts = sum(1 for p in path_parts if p in query_tokens or p in query_lower)
                if matched_parts >= 2:
                    score *= self.config.filepath_match_boost
                elif matched_parts == 1:
                    score *= self.config.filepath_match_boost * 0.8

            # 4. MDS 模型类名匹配提升
            mds_class = r.chunk.metadata.get("mds_class", "")
            if mds_class and mds_class.lower() in query_lower:
                score *= self.config.mds_model_match_boost

            # 5. 内容关键词匹配提升：content 中包含至少 2 个 query 关键词则 boost
            if content_keywords:
                content_lower = r.chunk.content.lower()
                matched_kw = sum(1 for kw in content_keywords if kw in content_lower)
                if matched_kw >= 2:
                    score *= self.config.content_keyword_boost

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

        # 终排：按最终分数降序
        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered
