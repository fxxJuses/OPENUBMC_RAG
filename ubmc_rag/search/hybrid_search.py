"""Hybrid search engine — BM25 + Dense retrieval with RRF fusion."""

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
        """Set chunk lookup index for reconstructing results."""
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
        """Perform hybrid search with RRF fusion."""
        search_config = self.config.search
        top_k = top_k or search_config.default_top_k
        top_k = min(top_k, search_config.max_top_k)

        # Process query
        processed = self.query_processor.process(query)
        if is_code_query is not None:
            processed.is_code_query = is_code_query

        # Build ChromaDB where filter
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

        # Dense search
        query_embedding = self.embedder.embed_query(processed.original)
        dense_results = self.vector_store.search(
            query_embedding, top_k=top_k * 3, where=where or None,
        )

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k * 3)

        # RRF fusion
        fused = self._rrf_fuse(
            dense_results, bm25_results,
            bm25_weight=self._get_bm25_weight(processed.is_code_query),
            dense_weight=self._get_dense_weight(processed.is_code_query),
            k=search_config.rrf_k,
        )

        # Build SearchResult objects
        results = []
        for chunk_id, score in fused[:top_k]:
            chunk = self._chunk_cache.get(chunk_id)
            if chunk is None:
                # Reconstruct from dense results
                chunk = self._reconstruct_chunk(chunk_id, dense_results)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score, source="hybrid"))

        # Rerank
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
        """Reciprocal Rank Fusion of dense and BM25 results."""
        scores: dict[str, float] = {}

        # Dense results
        for rank, item in enumerate(dense_results):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + dense_weight / (k + rank + 1)

        # BM25 results
        for rank, (chunk_id, _score) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _get_bm25_weight(self, is_code_query: bool) -> float:
        base = self.config.search.bm25_weight
        if is_code_query:
            base += self.config.search.code_query_bm25_boost
        return base

    def _get_dense_weight(self, is_code_query: bool) -> float:
        base = self.config.search.dense_weight
        if is_code_query:
            base -= self.config.search.code_query_bm25_boost
        return max(base, 0.1)

    def _reconstruct_chunk(self, chunk_id: str, dense_results: list[dict]) -> Optional[CodeChunk]:
        """Reconstruct a CodeChunk from dense search results."""
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
