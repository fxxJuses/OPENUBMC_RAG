"""
DashScope qwen3-rerank 重排序器 (迭代6-B)。

通过 DashScope API 调用 qwen3-rerank 模型对查询-文档对进行
深度语义相关性评分，作为交叉编码器的替代后端。

工作流程：
Query + Candidates -> DashScope qwen3-rerank API -> Reranked Scores -> Top-K
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from ubmc_rag.models.search_result import SearchResult

logger = logging.getLogger(__name__)

# DashScope rerank API 配置常量
_DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
_DASHSCOPE_RERANK_MODEL = "qwen3-rerank"
_MAX_DOCUMENTS = 100            # 单次 API 调用最大文档数
_MAX_CHARS_PER_DOC = 4000       # 单个文档最大字符数
_MIN_INTERVAL = 0.1             # API 调用最小间隔（秒）
_MAX_RETRIES = 2                # 最大重试次数


class DashScopeReranker:
    """DashScope qwen3-rerank API 重排序器。

    使用阿里云 DashScope 的 text-rerank API 对搜索结果进行
    深度语义相关性评分，支持批量重排序和降级处理。

    Attributes:
        api_key: DashScope API 密钥
        model: 模型名称（默认 qwen3-rerank）
        top_n: 返回的最大结果数
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = _DASHSCOPE_RERANK_MODEL,
        top_n: int = 20,
    ):
        """
        Args:
            api_key: DashScope API 密钥，为空则从环境变量读取
            model: 模型名称
            top_n: 单次 API 调用返回的最大结果数
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.top_n = top_n
        self._last_call_time = 0.0

    @property
    def available(self) -> bool:
        """是否有可用的 API 密钥。"""
        return bool(self.api_key)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """对候选结果进行 DashScope qwen3-rerank 重排序。

        Args:
            query: 原始查询文本
            candidates: 待重排序的候选结果列表
            top_k: 返回的结果数量上限（None 则返回全部）

        Returns:
            按重排序分数降序排列的结果列表
        """
        if not candidates:
            return []

        if not self.available:
            logger.warning(
                "DashScope API key not configured; returning un-reranked candidates"
            )
            return candidates[:top_k] if top_k else candidates

        top_n = min(top_k or self.top_n, len(candidates))

        try:
            return self._api_rerank(query, candidates, top_n, top_k)
        except Exception as e:
            logger.warning(
                "DashScope rerank failed: %s. Returning un-reranked candidates.", e
            )
            return candidates[:top_k] if top_k else candidates

    def _api_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int,
        top_k: int | None,
    ) -> list[SearchResult]:
        """调用 DashScope text-rerank API 进行重排序。"""
        # 提取文档内容
        documents = [
            c.chunk.content[:_MAX_CHARS_PER_DOC] for c in candidates
        ]

        # 限流：确保最小调用间隔
        elapsed = time.time() - self._last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 带重试的 API 调用
        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    _DASHSCOPE_RERANK_URL,
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                self._last_call_time = time.time()

                if resp.status_code == 200:
                    data = resp.json()
                    return self._parse_response(data, candidates, top_k)

                # 非 200 响应
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("DashScope rerank attempt %d failed: %s", attempt + 1, error_msg)
                last_error = Exception(error_msg)

                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** attempt)  # 指数退避

            except requests.RequestException as e:
                logger.warning("DashScope rerank attempt %d failed: %s", attempt + 1, e)
                last_error = e
                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** attempt)

        raise last_error or Exception("DashScope rerank failed with unknown error")

    def _parse_response(
        self,
        data: dict,
        candidates: list[SearchResult],
        top_k: int | None,
    ) -> list[SearchResult]:
        """解析 DashScope API 响应，构建重排序结果。"""
        output = data.get("output", {})
        results_data = output.get("results", [])

        if not results_data:
            logger.warning("DashScope rerank returned empty results")
            return candidates[:top_k] if top_k else candidates

        # 按 API 返回的 relevance_score 构建结果
        reranked = []
        for result_item in results_data:
            index = result_item.get("index", 0)
            score = result_item.get("relevance_score", 0.0)

            if index < len(candidates):
                candidate = candidates[index]
                reranked.append(SearchResult(
                    chunk=candidate.chunk,
                    score=float(score),
                    source="dashscope_rerank",
                ))

        # 按分数降序排列
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k and len(reranked) > top_k:
            reranked = reranked[:top_k]

        logger.info(
            "DashScope rerank complete: %d candidates -> %d results",
            len(candidates), len(reranked),
        )
        return reranked


def create_dashscope_reranker(
    api_key: str = "",
    model: str = _DASHSCOPE_RERANK_MODEL,
    top_n: int = 20,
) -> DashScopeReranker:
    """工厂函数：创建 DashScope 重排序器。

    Args:
        api_key: DashScope API 密钥
        model: 模型名称
        top_n: 单次 API 调用返回的最大结果数

    Returns:
        DashScopeReranker 实例
    """
    return DashScopeReranker(api_key=api_key, model=model, top_n=top_n)
