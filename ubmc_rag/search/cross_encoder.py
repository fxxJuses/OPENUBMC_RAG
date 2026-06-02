"""
交叉编码器重排序模块 (P0)。

迭代6-P0：在 Reranker 管道中增加交叉编码器重排序步骤，
使用神经模型对查询-文档对进行深度语义相关性评分。

支持两种模式：
1. sentence-transformers 本地模型（推荐 BGE-reranker-v2-m3）
2. 启发式降级方案（当模型不可用时自动切换）

工作流程：
Query + Candidates -> Cross-Encoder Model -> Reranked Scores -> Top-K
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ubmc_rag.models.search_result import SearchResult

logger = logging.getLogger(__name__)

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available; cross-encoder will use heuristic fallback"
    )


class CrossEncoderReranker:
    """交叉编码器重排序器。

    使用神经模型对 (query, document) 对进行深度语义评分，
    重新排序候选结果以提升检索精度。

    Attributes:
        model_name: 交叉编码器模型名称
        device: 推理设备（cpu/cuda）
        model: 底层 sentence-transformers 模型实例
        fallback: 是否处于降级模式
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace 模型名或本地路径
            device: 推理设备
            max_length: 输入最大 token 长度
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model: Optional[STCrossEncoder] = None
        self.fallback = not _ST_AVAILABLE

        if _ST_AVAILABLE:
            try:
                # 默认使用 HuggingFace 镜像加速模型下载
                if not os.environ.get("HF_ENDPOINT"):
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                self.model = STCrossEncoder(
                    model_name,
                    device=device,
                    max_length=max_length,
                    trust_remote_code=True,
                )
                logger.info(
                    "Cross-encoder loaded: %s (device=%s)", model_name, device
                )
                self.fallback = False
            except Exception as e:
                logger.warning(
                    "Failed to load cross-encoder model '%s': %s. "
                    "Falling back to heuristic reranker.",
                    model_name, e,
                )
                self.fallback = True
        else:
            logger.info(
                "Cross-encoder using heuristic fallback (sentence-transformers not installed)"
            )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """对候选结果进行交叉编码器重排序。

        Args:
            query: 原始查询文本
            candidates: 待重排序的候选结果列表
            top_k: 返回的结果数量上限（None 则返回全部）

        Returns:
            按交叉编码器分数降序排列的结果列表
        """
        if not candidates:
            return []

        if self.fallback or self.model is None:
            return self._heuristic_rerank(query, candidates, top_k)

        try:
            return self._model_rerank(query, candidates, top_k)
        except Exception as e:
            logger.warning(
                "Cross-encoder model rerank failed: %s. Falling back to heuristic.",
                e,
            )
            return self._heuristic_rerank(query, candidates, top_k)

    def _model_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None,
    ) -> list[SearchResult]:
        """使用 sentence-transformers 模型进行重排序。"""
        # 构建 (query, document) 对
        pairs = [(query, c.chunk.content) for c in candidates]

        # 获取交叉编码器分数
        scores: list[float] = self.model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True,
        ).cpu().tolist()

        # 用新分数更新结果
        reranked = []
        for candidate, score in zip(candidates, scores):
            reranked.append(SearchResult(
                chunk=candidate.chunk,
                score=float(score),
                source="cross_encoder",
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k] if top_k else reranked

    def _heuristic_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None,
    ) -> list[SearchResult]:
        """启发式降级重排序。

        当交叉编码器模型不可用时，基于查询-文档关键词重叠
        和已有的融合分数进行加权重排序。
        """
        import re

        query_tokens = set(
            t.lower() for t in re.findall(r'[a-zA-Z_]\w*', query) if len(t) > 1
        )

        reranked = []
        for candidate in candidates:
            content_lower = candidate.chunk.content.lower()

            # 计算查询 token 在文档中的命中率
            if query_tokens:
                hit_count = sum(
                    1 for t in query_tokens if t in content_lower
                )
                overlap_score = hit_count / len(query_tokens)
            else:
                overlap_score = 0.0

            # 融合：70% 原始分数 + 30% 关键词重叠分数
            blended = candidate.score * 0.7 + overlap_score * 0.3
            reranked.append(SearchResult(
                chunk=candidate.chunk,
                score=blended,
                source="cross_encoder_fallback",
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k] if top_k else reranked

    @property
    def is_fallback(self) -> bool:
        """是否处于降级模式。"""
        return self.fallback


def create_cross_encoder(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
) -> CrossEncoderReranker:
    """工厂函数：创建交叉编码器重排序器。

    Args:
        model_name: 模型名称
        device: 推理设备

    Returns:
        CrossEncoderReranker 实例
    """
    return CrossEncoderReranker(model_name=model_name, device=device)
