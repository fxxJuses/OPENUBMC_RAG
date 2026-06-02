"""
DashScope qwen3-rerank 云端重排序模块。

使用阿里云 DashScope qwen3-rerank 模型对候选结果进行语义相关性重排序，
作为本地交叉编码器的云端替代方案。API 调用失败时自动降级返回原始候选列表。
"""

from __future__ import annotations

import logging
import os

import httpx

from ubmc_rag.models.search_result import SearchResult

logger = logging.getLogger(__name__)

_DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
_MAX_DOCUMENTS_PER_REQUEST = 500  # qwen3-rerank 单次请求最大文档数
_DEFAULT_INSTRUCT = (
    "Given a programming query about BMC software components, "
    "retrieve relevant code snippets and documentation."
)


class DashScopeReranker:
    """DashScope qwen3-rerank 云端重排序器。

    通过 DashScope 兼容 API 调用 qwen3-rerank 模型，
    对 (query, document) 对进行语义相关性评分并重排序。

    Attributes:
        model: DashScope rerank 模型名称
        api_key: DashScope API 密钥
    """

    def __init__(self, model: str = "qwen3-rerank", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "DASHSCOPE_API_KEY not set; DashScope reranker will always fall back"
            )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int = 20,
    ) -> list[SearchResult]:
        """对候选结果调用 DashScope rerank API 进行重排序。

        API 失败时返回原始候选列表（零退化）。

        Args:
            query: 原始查询文本
            candidates: 待重排序的候选结果列表
            top_n: 返回的最大结果数

        Returns:
            按 DashScope 相关性分数降序排列的结果列表
        """
        if not candidates or not self.api_key:
            return candidates[:top_n] if top_n else candidates

        try:
            return self._call_rerank(query, candidates, top_n)
        except Exception as e:
            logger.warning(
                "DashScope rerank API failed (model=%s): %s. "
                "Returning original candidates.",
                self.model, e,
            )
            return candidates[:top_n] if top_n else candidates

    def _call_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        """调用 DashScope rerank API 并返回重排序结果。

        qwen3-rerank 使用 /compatible-api/v1/reranks 端点，
        请求体为扁平结构（query/documents/top_n 与 model 同级），
        响应中 results 直接位于顶层（不嵌套在 output 下）。
        """
        documents = [c.chunk.content for c in candidates]

        # 限制文档数量
        if len(documents) > _MAX_DOCUMENTS_PER_REQUEST:
            documents = documents[:_MAX_DOCUMENTS_PER_REQUEST]
            candidates = candidates[:_MAX_DOCUMENTS_PER_REQUEST]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "instruct": _DEFAULT_INSTRUCT,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = httpx.post(
            _DASHSCOPE_RERANK_URL,
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()

        data = response.json()

        # qwen3-rerank 响应: {"results": [{"index": 0, "relevance_score": 0.93}, ...]}
        results = data.get("results", [])
        if not results:
            logger.warning("DashScope rerank returned empty results")
            return candidates[:top_n]

        # 按 relevance_score 降序排列（API 通常已排序，但确保一致性）
        results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)

        reranked = []
        for item in results:
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if idx < len(candidates):
                reranked.append(SearchResult(
                    chunk=candidates[idx].chunk,
                    score=float(score),
                    source="dashscope_reranker",
                ))

        tokens = data.get("usage", {}).get("total_tokens", 0)
        logger.debug(
            "DashScope rerank: %d candidates -> %d results, %d tokens consumed",
            len(candidates), len(reranked), tokens,
        )

        return reranked

    @property
    def is_available(self) -> bool:
        """DashScope reranker 是否可用（已配置 API key）。"""
        return bool(self.api_key)
