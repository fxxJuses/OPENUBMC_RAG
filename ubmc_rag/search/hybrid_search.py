"""
混合搜索引擎 —— BM25 关键词检索 + Dense 向量检索，通过 Reranker 融合。

实现双路检索架构：
1. Dense 路径：通过 DashScope 嵌入模型 + ChromaDB 向量搜索
2. BM25 路径：通过代码感知分词器 + Okapi BM25 关键词匹配
3. 融合+重排序：Reranker 内部完成 RRF 融合 → boosting → diversity

迭代5：RRF 融合逻辑已移入 Reranker，HybridSearchEngine 仅负责
双路检索和结果组装，不再直接执行 RRF 融合。
"""

from __future__ import annotations

import logging
import re
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

# 依赖/接口相关查询关键词 —— 触发 service.json 定向检索
_DEPENDENCY_QUERY_RE = re.compile(
    r"依赖|dependency|dependencies|接口定义|interface|"
    r"组件.*关系|component.*dep|依赖关系|dep graph|"
    r"service\.json|component info",
    re.IGNORECASE,
)


class HybridSearchEngine:
    """混合搜索引擎，执行双路检索并委托 Reranker 融合+排序。

    工作流程：
    1. QueryProcessor 分析查询意图、提取过滤条件、扩展术语
    2. 分别执行 BM25 和 Dense 检索
    3. 构建 SearchResult 列表（Dense + BM25 各自）
    4. 委托 Reranker 执行 RRF 融合 → boosting → diversity
    5. 返回最终 top_k 结果

    Attributes:
        embedder: 向量嵌入服务
        vector_store: ChromaDB 向量存储
        bm25: BM25 关键词索引
        config: 应用配置
        query_processor: 查询处理器
        reranker: 结果重排序器（内部集成 RRF 融合）
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

        # ChromaDB 多条件过滤需要 $and 操作符
        if len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}

        # --- 双路检索 ---

        # Dense 向量检索
        query_embedding = self.embedder.embed_query(processed.original)
        dense_raw = self.vector_store.search(
            query_embedding, top_k=top_k * 3, where=where or None,
        )

        # BM25 关键词检索（使用扩展后的查询以增强关键词覆盖）
        bm25_raw = self.bm25.search(processed.expanded, top_k=top_k * 3)

        # --- 构建 SearchResult 列表 ---

        dense_results = []
        for item in dense_raw:
            chunk_id = item["chunk_id"]
            chunk = self._chunk_cache.get(chunk_id)
            if chunk is None:
                chunk = CodeChunk.from_chroma_metadata(
                    chunk_id=chunk_id,
                    content=item["content"],
                    meta=item["metadata"],
                )
            if chunk:
                dense_results.append(SearchResult(
                    chunk=chunk,
                    score=item.get("distance", 0.0),
                    source="dense",
                ))

        bm25_results = []
        for chunk_id, score in bm25_raw:
            chunk = self._chunk_cache.get(chunk_id)
            if chunk is None:
                chunk = self._reconstruct_chunk(chunk_id, dense_raw)
            if chunk:
                bm25_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    source="bm25",
                ))

        # --- 定向补充：依赖/接口类查询注入 service.json 候选 ---
        if self._is_dependency_query(query, processed.expanded):
            existing_ids = {r.chunk.chunk_id for r in dense_results}
            svc_results = self._retrieve_mds_service(query_embedding, existing_ids)
            if svc_results and dense_results:
                # 注入 Dense 结果前部（rank 5），确保 RRF 融合获得较高权重
                insert_pos = min(5, len(dense_results))
                for i, r in enumerate(svc_results[:3]):
                    dense_results.insert(insert_pos + i, r)
                # 同步注入 BM25 结果前部，使这些 chunk 在双路都有 RRF 贡献
                bm25_insert = min(10, len(bm25_results))
                for i, r in enumerate(svc_results[:3]):
                    bm25_results.insert(bm25_insert + i, SearchResult(
                        chunk=r.chunk, score=10.0, source="bm25",
                    ))
                logger.debug("Injected %d mds_service candidates", len(svc_results[:3]))

        # --- 计算 RRF 权重 ---
        bm25_w = search_config.bm25_weight
        dense_w = search_config.dense_weight
        if processed.is_code_query:
            bm25_w += search_config.code_query_bm25_boost
            dense_w -= search_config.code_query_bm25_boost
            dense_w = max(dense_w, 0.1)

        # --- 委托 Reranker 执行 RRF 融合 + boosting + diversity ---
        return self.reranker.rerank(
            dense_results=dense_results,
            bm25_results=bm25_results,
            query=query,
            top_k=top_k,
            bm25_weight=bm25_w,
            dense_weight=dense_w,
        )

    def _is_dependency_query(self, query: str, expanded: str) -> bool:
        """检测查询是否涉及依赖/接口关系（触发 service.json 定向检索）。"""
        return bool(_DEPENDENCY_QUERY_RE.search(query) or _DEPENDENCY_QUERY_RE.search(expanded))

    def _retrieve_mds_service(
        self, query_embedding: list[float], existing_chunk_ids: set[str],
    ) -> list[SearchResult]:
        """定向检索 mds_service 分块，补充依赖/接口类查询的候选池。"""
        raw = self.vector_store.search(
            query_embedding, top_k=10, where={"chunk_type": "mds_service"},
        )
        results = []
        for item in raw:
            cid = item["chunk_id"]
            if cid in existing_chunk_ids:
                continue
            chunk = self._chunk_cache.get(cid)
            if chunk is None:
                chunk = CodeChunk.from_chroma_metadata(
                    chunk_id=cid, content=item["content"], meta=item["metadata"],
                )
            if chunk:
                results.append(SearchResult(
                    chunk=chunk, score=item.get("distance", 0.0), source="dense",
                ))
        return results

    def search_raw(
        self,
        query: str,
        top_k: int | None = None,
        is_code_query: bool | None = None,
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """执行双路检索，返回原始的 Dense 和 BM25 结果（不做融合）。

        供需要手动控制融合逻辑的场景使用（如评估框架的 hybrid 模式）。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            is_code_query: 是否为代码类查询

        Returns:
            (dense_results, bm25_results) 元组
        """
        search_config = self.config.search
        top_k = top_k or search_config.default_top_k
        top_k = min(top_k, search_config.max_top_k)

        processed = self.query_processor.process(query)
        if is_code_query is not None:
            processed.is_code_query = is_code_query

        # Dense
        query_embedding = self.embedder.embed_query(processed.original)
        dense_raw = self.vector_store.search(
            query_embedding, top_k=top_k * 3,
        )

        # BM25
        bm25_raw = self.bm25.search(processed.expanded, top_k=top_k * 3)

        # Build results
        dense_results = []
        for item in dense_raw:
            chunk_id = item["chunk_id"]
            chunk = self._chunk_cache.get(chunk_id)
            if chunk is None:
                chunk = CodeChunk.from_chroma_metadata(
                    chunk_id=chunk_id,
                    content=item["content"],
                    meta=item["metadata"],
                )
            if chunk:
                dense_results.append(SearchResult(
                    chunk=chunk,
                    score=item.get("distance", 0.0),
                    source="dense",
                ))

        bm25_results = []
        for chunk_id, score in bm25_raw:
            chunk = self._chunk_cache.get(chunk_id)
            if chunk is None:
                chunk = self._reconstruct_chunk(chunk_id, dense_raw)
            if chunk:
                bm25_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    source="bm25",
                ))

        return dense_results, bm25_results

    def _reconstruct_chunk(self, chunk_id: str, dense_results: list[dict]) -> Optional[CodeChunk]:
        """从 Dense 检索结果中重建 CodeChunk 对象（分块不在缓存中时的降级方案）。"""
        for item in dense_results:
            if item["chunk_id"] == chunk_id:
                return CodeChunk.from_chroma_metadata(
                    chunk_id=chunk_id,
                    content=item["content"],
                    meta=item["metadata"],
                )
        return None
