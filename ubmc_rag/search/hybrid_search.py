"""混合搜索引擎 —— BM25 关键词检索 + Dense 向量检索，通过 RRF 融合。

实现双路检索架构：
1. Dense 路径：通过 DashScope 嵌入模型 + ChromaDB 向量搜索
2. BM25 路径：通过代码感知分词器 + Okapi BM25 关键词匹配
3. 融合：使用 Reciprocal Rank Fusion (RRF) 合并两路结果
4. 重排序：应用符号匹配、路径匹配等提升规则和多样性过滤
"""

from __future__ import annotations

import logging
from typing import Optional

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.bm25_index import BM25Index
from ubmc_rag.indexing.embedder import Embedder
from ubmc_rag.indexing.vector_store import VectorStore
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.query_processor import QueryProcessor
from ubmc_rag.search.reranker import Reranker

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """混合搜索引擎，融合 BM25 和 Dense 双路检索结果。

    工作流程：
    1. QueryProcessor 分析查询意图和提取过滤条件
    2. 分别执行 BM25 和 Dense 检索
    3. RRF 融合两路结果
    4. Reranker 应用提升规则和多样性过滤

    Attributes:
        embedder: 向量嵌入服务
        vector_store: ChromaDB 向量存储
        bm25: BM25 关键词索引
        config: 应用配置
        query_processor: 查询处理器
        reranker: 结果重排序器
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        config: AppConfig,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.config = config
        self.query_processor = QueryProcessor()
        self.reranker = Reranker(config.search)
        self._chunk_cache: dict[str, CodeChunk] = {}

    def set_chunk_index(self, chunks: list[CodeChunk]) -> None:
        """设置分块查找索引，用于从搜索结果重建 CodeChunk 对象。"""
        self._chunk_cache = {c.chunk_id: c for c in chunks}

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language: str | None = None,
        repo: str | None = None,
        chunk_type: str | None = None,
        is_code_query: bool | None = None,
    ) -> list[SearchResult]:
        """执行混合搜索，返回融合并重排序后的结果。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量，默认使用配置值
            language: 按编程语言过滤（如 "lua", "c"）
            repo: 按仓库名过滤（如 "sensor"）
            chunk_type: 按分块类型过滤（如 "function", "mds_model"）
            is_code_query: 是否为代码类查询（影响 BM25/Dense 权重）

        Returns:
            重排序后的搜索结果列表
        """
        search_config = self.config.search
        top_k = top_k or search_config.default_top_k
        top_k = min(top_k, search_config.max_top_k)

        # 查询分析：提取意图和过滤条件
        processed = self.query_processor.process(query)
        if is_code_query is not None:
            processed.is_code_query = is_code_query

        # 构建 ChromaDB where 过滤条件
        where = {}
        if language:
            where["language"] = language
        elif "language" in processed.filters:
            where["language"] = processed.filters["language"]
        if repo:
            where["repo_name"] = repo
        if chunk_type:
            where["chunk_type"] = chunk_type
        elif "chunk_type" in processed.filters:
            where["chunk_type"] = processed.filters["chunk_type"]

        # Dense 向量检索
        query_embedding = self.embedder.embed_query(processed.original)
        dense_results = self.vector_store.search(
            query_embedding, top_k=top_k * 3, where=where or None,
        )

        # BM25 关键词检索
        bm25_results = self.bm25.search(query, top_k=top_k * 3)

        # RRF 融合
        fused = self._rrf_fuse(
            dense_results, bm25_results,
            bm25_weight=self._get_bm25_weight(processed.is_code_query),
            dense_weight=self._get_dense_weight(processed.is_code_query),
            k=search_config.rrf_k,
        )

        # 构建搜索结果
        results = []
        for chunk_id, score in fused[:top_k]:
            chunk = self._chunk_cache.get(chunk_id)
            if chunk is None:
                chunk = self._reconstruct_chunk(chunk_id, dense_results)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score, source="hybrid"))

        # 重排序
        results = self.reranker.rerank(results, query)

        return results[:top_k]

    def _rrf_fuse(
        self,
        dense_results: list[dict],
        bm25_results: list[tuple[str, float]],
        bm25_weight: float,
        dense_weight: float,
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion 融合 Dense 和 BM25 检索结果。

        RRF 公式：score(d) = Σ 1/(k + rank(d))
        每路结果按其权重加权后求和，k 控制低排名结果的平滑程度。

        Args:
            dense_results: Dense 检索结果列表
            bm25_results: BM25 检索结果列表
            bm25_weight: BM25 路径权重
            dense_weight: Dense 路径权重
            k: RRF 平滑参数

        Returns:
            (chunk_id, fused_score) 元组列表，按分数降序排列
        """
        scores: dict[str, float] = {}

        for rank, item in enumerate(dense_results):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + dense_weight / (k + rank + 1)

        for rank, (chunk_id, _score) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _get_bm25_weight(self, is_code_query: bool) -> float:
        """获取 BM25 权重，代码类查询时额外提升关键词匹配的重要性。"""
        base = self.config.search.bm25_weight
        if is_code_query:
            base += self.config.search.code_query_bm25_boost
        return base

    def _get_dense_weight(self, is_code_query: bool) -> float:
        """获取 Dense 权重，代码类查询时适当降低语义匹配的占比。"""
        base = self.config.search.dense_weight
        if is_code_query:
            base -= self.config.search.code_query_bm25_boost
        return max(base, 0.1)

    def _reconstruct_chunk(self, chunk_id: str, dense_results: list[dict]) -> Optional[CodeChunk]:
        """从 Dense 检索结果中重建 CodeChunk 对象（分块不在缓存中时的降级方案）。"""
        for item in dense_results:
            if item["chunk_id"] == chunk_id:
                meta = item["metadata"]
                return CodeChunk(
                    chunk_id=chunk_id,
                    content=item["content"],
                    file_path=meta.get("file_path", ""),
                    repo_name=meta.get("repo_name", ""),
                    language=meta.get("language", ""),
                    component_name=meta.get("component_name", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    chunk_type=meta.get("chunk_type", ""),
                )
        return None
