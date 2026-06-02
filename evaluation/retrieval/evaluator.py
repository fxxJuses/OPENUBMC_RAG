"""检索评估器：对 HybridSearchEngine 多种搜索模式运行回归评测。

支持七种搜索模式，用于 A/B 对比各检索路径的效果：
- "bm25_only": 仅 BM25 关键词检索
- "dense_only": 仅向量语义检索
- "hybrid": RRF 融合，不经过 Reranker boosting
- "hybrid_reranked": 完整管线（默认，与生产环境一致）
- "hybrid_cross_encoder": RRF + BGE-reranker-v2-m3 交叉编码器重排序
- "hybrid_dashscope": RRF + DashScope qwen3-rerank 云端重排序
- "hybrid_full": 完整管线（LLM 重写 + RRF + DashScope 重排序）

迭代5：_search_hybrid_no_rerank 改用 Reranker.rrf_fuse() 获取仅融合结果。
迭代6-P0：增加 hybrid_cross_encoder 模式，评估交叉编码器真实效果。
迭代9：增加 hybrid_dashscope 和 hybrid_full 模式。
"""

from __future__ import annotations

import logging
from typing import Optional

from evaluation.datasets.schema import RegressionDataset
from evaluation.retrieval.metrics import (
    CaseResult,
    RetrievalMetrics,
    compute_metrics,
    evaluate_case,
)
from ubmc_rag.config.settings import AppConfig, SearchConfig
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.cross_encoder import CrossEncoderReranker
from ubmc_rag.search.dashscope_reranker import DashScopeReranker
from ubmc_rag.search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """检索质量评估器。

    加载索引、构建搜索引擎、遍历回归数据集，计算检索指标。

    Attributes:
        config: 应用配置
        index_mgr: 索引管理器
        engine: 混合搜索引擎
        chunks: 所有已索引的代码分块
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.index_mgr = IndexManager(config)

        loaded = self.index_mgr.load_index()
        if not loaded:
            raise RuntimeError("No index found. Run 'ubmc-rag index' first to build the index.")

        self.chunks = self.index_mgr.get_all_chunks()
        logger.info("Loaded %d chunks from index", len(self.chunks))

        # 构建搜索引擎（与 search_cmd.py 一致）
        self.engine = HybridSearchEngine(
            embedder=self.index_mgr.embedder,
            vector_store=self.index_mgr.vector_store,
            bm25=self.index_mgr.bm25,
            config=config,
        )
        self.engine.set_chunk_index(self.chunks)

        # 交叉编码器（延迟初始化）
        self._cross_encoder: Optional[CrossEncoderReranker] = None

        # DashScope 重排序器（延迟初始化）
        self._dashscope_reranker: Optional[DashScopeReranker] = None

    def evaluate(
        self,
        dataset: RegressionDataset,
        top_k: int = 10,
        search_mode: str = "hybrid_reranked",
    ) -> RetrievalMetrics:
        """对整个回归数据集运行评估。

        Args:
            dataset: 回归测试数据集
            top_k: 搜索返回的最大结果数
            search_mode: 搜索模式
                "bm25_only" / "dense_only" / "hybrid" / "hybrid_reranked"

        Returns:
            汇总的检索指标
        """
        case_results: list[CaseResult] = []
        skipped = 0

        for tc in dataset.test_cases:
            try:
                results = self._search(tc.query, top_k=top_k, search_mode=search_mode)
                cr = evaluate_case(results, tc)
                case_results.append(cr)
                logger.debug(
                    "Case %s: File@5=%s, MRR=%s",
                    tc.id,
                    cr.file_at_k.get(5),
                    cr.first_relevant_rank,
                )
            except Exception as e:
                skipped += 1
                logger.warning("Case %s failed (skipped): %s", tc.id, e)

        metrics = compute_metrics(case_results)
        if skipped > 0:
            logger.warning("Skipped %d/%d cases due to errors", skipped, len(dataset.test_cases))
        logger.info(
            "Evaluation complete: %d/%d cases, mode=%s, File@5=%.4f, MRR=%.4f",
            metrics.total_cases,
            len(dataset.test_cases),
            search_mode,
            metrics.file_at_5,
            metrics.mrr,
        )
        return metrics

    def _search(
        self,
        query: str,
        top_k: int = 10,
        search_mode: str = "hybrid_reranked",
    ) -> list[SearchResult]:
        """根据搜索模式执行检索。

        Args:
            query: 查询文本
            top_k: 返回结果数
            search_mode: 搜索模式

        Returns:
            搜索结果列表
        """
        if search_mode == "hybrid_reranked":
            return self.engine.search(query, top_k=top_k)

        if search_mode == "hybrid_cross_encoder":
            return self._search_hybrid_cross_encoder(query, top_k)

        if search_mode == "hybrid_dashscope":
            return self._search_hybrid_dashscope(query, top_k)

        if search_mode == "hybrid_full":
            return self._search_hybrid_full(query, top_k)

        if search_mode == "hybrid":
            return self._search_hybrid_no_rerank(query, top_k)

        if search_mode == "bm25_only":
            return self._search_bm25_only(query, top_k)

        if search_mode == "dense_only":
            return self._search_dense_only(query, top_k)

        raise ValueError(f"Unknown search_mode: {search_mode}")

    def _search_bm25_only(self, query: str, top_k: int) -> list[SearchResult]:
        """仅 BM25 检索。"""
        bm25_results = self.engine.bm25.search(query, top_k=top_k)
        results = []
        for chunk_id, score in bm25_results[:top_k]:
            chunk = self.engine._chunk_cache.get(chunk_id)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score, source="bm25"))
        return results

    def _search_dense_only(self, query: str, top_k: int) -> list[SearchResult]:
        """仅 Dense 向量检索。"""
        query_embedding = self.engine.embedder.embed_query(query)
        dense_raw = self.engine.vector_store.search(
            query_embedding,
            top_k=top_k,
        )
        results = []
        for item in dense_raw[:top_k]:
            chunk = self._reconstruct_from_dense(item)
            if chunk:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=item.get("distance", 0.0),
                        source="dense",
                    )
                )
        return results

    def _search_hybrid_no_rerank(self, query: str, top_k: int) -> list[SearchResult]:
        """RRF 融合但不经过 boosting（仅 RRF + diversity）。

        迭代5：使用 engine.search_raw() 获取双路原始结果，
        然后通过 Reranker.rrf_fuse() 做纯 RRF 融合。
        """
        search_config = self.config.search

        # 获取双路原始 SearchResult
        dense_results, bm25_results = self.engine.search_raw(query, top_k=top_k)

        # 使用 Reranker 的纯 RRF 融合（不 boosting）
        rrf_results = self.engine.reranker.rrf_fuse(
            dense_results,
            bm25_results,
            bm25_weight=search_config.bm25_weight,
            dense_weight=search_config.dense_weight,
        )

        # 应用 diversity 后返回
        return self.engine.reranker._apply_diversity(rrf_results)[:top_k]

    def _search_hybrid_cross_encoder(self, query: str, top_k: int) -> list[SearchResult]:
        """完整管线 + 交叉编码器重排序。

        流程：RRF 融合 → 交叉编码器 top-(top_k*3) → boosting → diversity → top_k。
        交叉编码器对 RRF 融合后的候选进行深度语义评分，提升排序精度。
        """
        search_config = self.config.search

        # 获取双路原始 SearchResult
        dense_results, bm25_results = self.engine.search_raw(query, top_k=top_k)

        # RRF 融合
        rrf_results = self.engine.reranker.rrf_fuse(
            dense_results,
            bm25_results,
            bm25_weight=search_config.bm25_weight,
            dense_weight=search_config.dense_weight,
        )

        # 取 top-(top_k*3) 候选送入交叉编码器
        candidate_count = min(top_k * 3, len(rrf_results))
        ce_candidates = rrf_results[:candidate_count]

        # 交叉编码器重排序
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoderReranker(
                model_name=search_config.cross_encoder_model,
                device=search_config.cross_encoder_device,
            )
            logger.info(
                "Cross-encoder initialized: model=%s device=%s fallback=%s",
                search_config.cross_encoder_model,
                search_config.cross_encoder_device,
                self._cross_encoder.is_fallback,
            )

        ce_reranked = self._cross_encoder.rerank(query, ce_candidates)

        # 应用 boosting（在交叉编码器分数之上）
        boosted = self.engine.reranker._apply_boosts(ce_reranked, query)

        # 应用 diversity
        diversified = self.engine.reranker._apply_diversity(boosted)

        return diversified[:top_k]

    def _reconstruct_from_dense(self, item: dict) -> Optional[CodeChunk]:
        """从 Dense 检索结果重建 CodeChunk。"""
        return CodeChunk.from_chroma_metadata(
            chunk_id=item.get("chunk_id", ""),
            content=item.get("content", ""),
            meta=item.get("metadata", {}),
        )

    def _search_hybrid_dashscope(self, query: str, top_k: int) -> list[SearchResult]:
        """RRF + DashScope qwen3-rerank 云端重排序。

        流程：search_raw → RRF → boosting → DashScope rank 作为第三路 RRF 信号叠加。
        DashScope 的排名不替换 RRF 分数，而是作为额外加分，避免分数尺度不匹配。
        """
        search_config = self.config.search

        dense_results, bm25_results = self.engine.search_raw(query, top_k=top_k)

        rrf_results = self.engine.reranker.rrf_fuse(
            dense_results,
            bm25_results,
            bm25_weight=search_config.bm25_weight,
            dense_weight=search_config.dense_weight,
        )

        candidate_count = min(top_k * 3, len(rrf_results))
        candidates = rrf_results[:candidate_count]

        # boosting 在 RRF 分数基础上
        boosted = self.engine.reranker._apply_boosts(candidates, query)

        # DashScope reranker 作为第三路信号叠加
        boosted = self._apply_dashscope_signal(query, boosted, search_config)

        # diversity
        diversified = self.engine.reranker._apply_diversity(boosted)

        return diversified[:top_k]

    def _search_hybrid_full(self, query: str, top_k: int) -> list[SearchResult]:
        """完整管线：LLM 查询重写 + RRF + DashScope 重排序。

        流程：search()（含 LLM 重写注入）→ DashScope rank 叠加 → diversity。
        DashScope 排名作为额外 RRF 信号，不覆盖原始分数。
        """
        search_config = self.config.search

        # search() 包含 LLM 重写 + RRF + boosting
        candidates = self.engine.search(query, top_k=top_k * 3)

        # DashScope reranker 作为第三路信号叠加
        results = self._apply_dashscope_signal(query, candidates, search_config)

        # diversity
        diversified = self.engine.reranker._apply_diversity(results)

        return diversified[:top_k]

    def _apply_dashscope_signal(
        self,
        query: str,
        candidates: list[SearchResult],
        search_config: SearchConfig,
        rrf_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """将 DashScope 排名作为第三路 RRF 信号叠加到现有分数。

        不替换原始分数，而是额外加上 ds_weight / (rrf_k + ds_rank + 1)。
        这样保持了 RRF + boosting 的信号，同时融入了 DashScope 的语义判断。
        """
        if self._dashscope_reranker is None:
            self._dashscope_reranker = DashScopeReranker(
                model=search_config.dashscope_reranker_model,
            )

        if not self._dashscope_reranker.is_available:
            return candidates

        ds_ranked = self._dashscope_reranker.rerank(
            query, candidates, top_n=search_config.dashscope_reranker_top_n,
        )

        # 构建 chunk_id → DashScope rank 映射
        ds_rank_map: dict[str, int] = {}
        for rank, sr in enumerate(ds_ranked):
            ds_rank_map[sr.chunk.chunk_id] = rank

        # 叠加 DashScope RRF 信号
        results = []
        for sr in candidates:
            ds_rank = ds_rank_map.get(sr.chunk.chunk_id)
            bonus = 0.0
            if ds_rank is not None:
                bonus = rrf_weight / (rrf_k + ds_rank + 1)
            results.append(SearchResult(
                chunk=sr.chunk,
                score=sr.score + bonus,
                source=sr.source,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results
