"""
搜索结果重排序器，集成 RRF 融合、提升规则和多样性过滤。

迭代5：将 RRF 融合从 hybrid_search 移入 Reranker，使 Reranker 成为
统一的融合+排序模块。Reranker 接收原始的 Dense 和 BM25 SearchResult 列表，
内部完成 RRF 融合、符号/路径/仓库匹配提升、多样性过滤。

迭代6-P0：增加可选的交叉编码器重排序步骤，在 boosting 之后、
diversity 之前对候选结果进行深度语义评分。

工作流程：
1. RRF 融合 Dense + BM25 双路结果
2. 符号名精确匹配提升（加法 bonus）
3. 仓库名匹配提升（加法 bonus）
4. 文件路径匹配提升（加法 bonus）
5. MDS 模型类名匹配提升（加法 bonus）
6. [可选] 交叉编码器深度语义重排序
7. 按最终分数重新排序
8. 同文件结果多样性降权
"""

from __future__ import annotations

import logging
from typing import Optional

from ubmc_rag.config.settings import SearchConfig
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.cross_encoder import CrossEncoderReranker
from ubmc_rag.search.dashscope_reranker import DashScopeReranker

logger = logging.getLogger(__name__)

# H3: 加法 bonus 常量（对标乘法 boost 效果，适配 RRF 分值范围）
SYMBOL_BONUS = 0.008        # 原 symbol_match_boost=1.5, 等效 +0.005-0.008
FILEPATH_BONUS = 0.006      # 原 filepath_match_boost=1.3, 等效 +0.003-0.005
REPO_BONUS = 0.006          # 仓库名匹配奖励
MDS_MODEL_BONUS = 0.012     # 原 mds_model_match_boost=2.0, 等效 +0.01-0.015
PARTIAL_MULTIPLIER = 0.8    # 部分匹配时 bonus 打折


class Reranker:
    """搜索结果重排序器，集成 RRF 融合、多维度提升和多样性过滤。

    迭代5：RRF 融合逻辑从 HybridSearchEngine 移入此处。
    Reranker 接收原始 Dense 和 BM25 结果，内部完成 RRF 融合，
    然后应用 boosting 和 diversity，返回最终排序结果。

    迭代6-P0：支持可选的交叉编码器深度重排序步骤。
    迭代6-B：支持可选的 DashScope qwen3-rerank API 重排序。

    Attributes:
        config: 搜索配置，包含 RRF 参数和提升规则参数
        cross_encoder: 交叉编码器重排序器实例（延迟初始化）
        dashscope_reranker: DashScope 重排序器实例（延迟初始化）
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self._cross_encoder: Optional[CrossEncoderReranker] = None
        self._cross_encoder_init_attempted = False
        self._dashscope_reranker: Optional[DashScopeReranker] = None
        self._dashscope_init_attempted = False

    def _get_cross_encoder(self) -> Optional[CrossEncoderReranker]:
        """延迟初始化交叉编码器（仅在启用且首次使用时加载）。"""
        if not self.config.cross_encoder_enabled:
            return None
        if self._cross_encoder_init_attempted:
            return self._cross_encoder
        self._cross_encoder_init_attempted = True
        try:
            self._cross_encoder = CrossEncoderReranker(
                model_name=self.config.cross_encoder_model,
                device=self.config.cross_encoder_device,
            )
            if self._cross_encoder.is_fallback:
                logger.info(
                    "Cross-encoder initialized in fallback mode (heuristic rerank)"
                )
            else:
                logger.info(
                    "Cross-encoder initialized: %s", self.config.cross_encoder_model,
                )
        except Exception as e:
            logger.warning("Failed to initialize cross-encoder: %s", e)
            self._cross_encoder = None
        return self._cross_encoder

    def _get_dashscope_reranker(self) -> Optional[DashScopeReranker]:
        """延迟初始化 DashScope 重排序器（仅在启用且首次使用时加载）。"""
        if not self.config.dashscope_reranker_enabled:
            return None
        if self._dashscope_init_attempted:
            return self._dashscope_reranker
        self._dashscope_init_attempted = True
        try:
            import os
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            self._dashscope_reranker = DashScopeReranker(
                api_key=api_key,
                model=self.config.dashscope_reranker_model,
                top_n=self.config.dashscope_reranker_top_n,
            )
            if self._dashscope_reranker.available:
                logger.info(
                    "DashScope reranker initialized: %s", self.config.dashscope_reranker_model,
                )
            else:
                logger.warning(
                    "DashScope reranker has no API key; will return un-reranked results"
                )
        except Exception as e:
            logger.warning("Failed to initialize DashScope reranker: %s", e)
            self._dashscope_reranker = None
        return self._dashscope_reranker

    def rrf_fuse(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
        k: int | None = None,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion 融合 Dense 和 BM25 检索结果。

        RRF 公式：score(d) = Σ w / (k + rank(d))
        每路结果按其权重加权后求和，k 控制低排名结果的平滑程度。

        同一 chunk 可能同时出现在两路结果中，此时 RRF 分数为两路之和。

        Args:
            dense_results: Dense 检索结果列表（已带原始分数）
            bm25_results: BM25 检索结果列表（已带原始分数）
            bm25_weight: BM25 路径权重，默认使用配置值
            dense_weight: Dense 路径权重，默认使用配置值
            k: RRF 平滑参数，默认使用配置值

        Returns:
            融合后的 SearchResult 列表，按 RRF 分数降序排列
        """
        bm25_w = bm25_weight if bm25_weight is not None else self.config.bm25_weight
        dense_w = dense_weight if dense_weight is not None else self.config.dense_weight
        rrf_k = k if k is not None else self.config.rrf_k

        # 收集所有 chunk：构建 chunk → (dense_rank, bm25_rank, chunk) 映射
        chunk_map: dict[str, dict] = {}

        for rank, sr in enumerate(dense_results):
            cid = sr.chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = {"chunk": sr.chunk, "dense_rank": rank, "bm25_rank": None}
            else:
                chunk_map[cid]["dense_rank"] = rank

        for rank, sr in enumerate(bm25_results):
            cid = sr.chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = {"chunk": sr.chunk, "dense_rank": None, "bm25_rank": rank}
            else:
                chunk_map[cid]["bm25_rank"] = rank

        # 计算 RRF 分数
        rrf_results: list[SearchResult] = []
        for cid, info in chunk_map.items():
            score = 0.0
            if info["dense_rank"] is not None:
                score += dense_w / (rrf_k + info["dense_rank"] + 1)
            if info["bm25_rank"] is not None:
                score += bm25_w / (rrf_k + info["bm25_rank"] + 1)
            rrf_results.append(SearchResult(
                chunk=info["chunk"],
                score=score,
                source="hybrid",
            ))

        rrf_results.sort(key=lambda x: x.score, reverse=True)
        return rrf_results

    def rerank(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        query: str,
        top_k: int = 10,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
        skip_boost: bool = False,
        skip_cross_encoder: bool = False,
        skip_dashscope_reranker: bool = False,
    ) -> list[SearchResult]:
        """对双路检索结果执行 RRF 融合 + 重排序。

        处理步骤（迭代6-B 增强）：
        1. RRF 融合 Dense + BM25 结果
        2. 如果未 skip_boost：应用符号名、仓库名、文件路径、MDS 模型匹配提升
        3. [可选] 交叉编码器深度语义重排序
        4. [可选] DashScope qwen3-rerank API 重排序
        5. 按最终分数重新排序
        6. 同一文件的重复结果降权（超过 diversity_max_per_file 的结果分数 ×0.7）

        Args:
            dense_results: Dense 向量检索结果列表
            bm25_results: BM25 关键词检索结果列表
            query: 原始查询文本
            top_k: 返回结果数量上限
            bm25_weight: BM25 路径权重，默认使用配置值
            dense_weight: Dense 路径权重，默认使用配置值
            skip_boost: 是否跳过 boosting（仅 RRF + diversity）
            skip_cross_encoder: 是否跳过交叉编码器重排序
            skip_dashscope_reranker: 是否跳过 DashScope 重排序

        Returns:
            重排序后的搜索结果列表
        """
        if not dense_results and not bm25_results:
            return []

        # 步骤 1: RRF 融合（取 top_k * 3 候选给后续 boosting）
        fused = self.rrf_fuse(
            dense_results, bm25_results,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

        rerank_candidates = min(top_k * 3, len(fused))
        candidates = fused[:rerank_candidates]

        if skip_boost:
            # 仅 RRF + diversity，不做 boosting
            return self._apply_diversity(candidates)[:top_k]

        # 步骤 2: 应用 boosting + 重排
        boosted = self._apply_boosts(candidates, query)

        # 步骤 3 (P0): 交叉编码器深度语义重排序
        if not skip_cross_encoder:
            cross_enc = self._get_cross_encoder()
            if cross_enc is not None:
                boosted = cross_enc.rerank(query, boosted, top_k=len(boosted))

        # 步骤 4 (6-B): DashScope qwen3-rerank API 重排序
        if not skip_dashscope_reranker:
            ds_reranker = self._get_dashscope_reranker()
            if ds_reranker is not None:
                boosted = ds_reranker.rerank(query, boosted, top_k=len(boosted))

        # 步骤 5: 多样性过滤
        diversified = self._apply_diversity(boosted)

        return diversified[:top_k]

    def _apply_boosts(
        self, results: list[SearchResult], query: str,
    ) -> list[SearchResult]:
        """对搜索结果应用多维提升规则（加法 bonus）。

        提升策略：
        1. 符号名匹配：查询中存在 chunk 的符号名 → +SYMBOL_BONUS
        2. 仓库名匹配：查询中包含仓库名 → +REPO_BONUS
        3. 文件路径匹配：查询 token 匹配路径片段 → +FILEPATH_BONUS
        4. MDS 模型类名匹配：查询中包含 MDS 类名 → +MDS_MODEL_BONUS

        Args:
            results: RRF 融合后的搜索结果
            query: 原始查询文本

        Returns:
            提升并重排后的结果列表
        """
        import re

        query_lower = query.lower()

        # 提取查询中的标识符 token
        query_tokens: set[str] = set()
        for token in re.findall(r'[a-zA-Z_]\w*', query_lower):
            query_tokens.add(token)
        for token in re.findall(r'\b[a-zA-Z_]{1,2}\b', query_lower):
            query_tokens.add(token)

        boosted = []
        for r in results:
            bonus = 0.0

            # 1. 符号名精确匹配提升
            for sym in r.chunk.symbols:
                sym_lower = sym.name.lower()
                if sym_lower in query_lower or any(
                    t in sym_lower for t in query_tokens if len(t) >= 3
                ):
                    bonus += SYMBOL_BONUS
                    break

            # 2. 仓库名匹配提升
            repo_lower = r.chunk.repo_name.lower()
            if repo_lower in query_lower:
                bonus += REPO_BONUS
            elif any(token in repo_lower for token in query_tokens if len(token) >= 3):
                bonus += REPO_BONUS * PARTIAL_MULTIPLIER

            # 3. 文件路径匹配提升
            file_path_lower = r.chunk.file_path.lower()
            if file_path_lower in query_lower:
                bonus += FILEPATH_BONUS
            else:
                path_parts: set[str] = set()
                for part in re.split(r'[/_.-]', file_path_lower):
                    if part:
                        path_parts.add(part)
                matched_parts = sum(
                    1 for p in path_parts if p in query_tokens or p in query_lower
                )
                if matched_parts >= 2:
                    bonus += FILEPATH_BONUS
                elif matched_parts == 1:
                    bonus += FILEPATH_BONUS * PARTIAL_MULTIPLIER

            # 4. MDS 模型类名匹配提升
            mds_class = r.chunk.metadata.get("mds_class", "")
            if mds_class and mds_class.lower() in query_lower:
                bonus += MDS_MODEL_BONUS

            boosted.append(SearchResult(
                chunk=r.chunk,
                score=r.score + bonus,
                source=r.source,
            ))

        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def _apply_diversity(
        self, results: list[SearchResult],
    ) -> list[SearchResult]:
        """应用多样性过滤：同一文件的超出部分降权。

        每个文件最多保留 diversity_max_per_file 个全分结果，
        超出部分分数 ×0.7。返回按最终分数降序排列的结果。

        Args:
            results: 待过滤的结果列表

        Returns:
            多样性过滤后的结果列表
        """
        filtered = []
        file_counts: dict[str, int] = {}
        for r in results:
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

        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered
