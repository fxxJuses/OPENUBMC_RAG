"""向量嵌入服务，通过 DashScope text-embedding-v4 API 生成代码嵌入向量。

使用 OpenAI 兼容接口调用阿里云 DashScope 的文本嵌入模型，
支持批量嵌入和自动限流。
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI

from ubmc_rag.config.settings import IndexingConfig
from ubmc_rag.models.code_chunk import CodeChunk

logger = logging.getLogger(__name__)

# DashScope API 配置常量
_API_BATCH_SIZE = 10          # 单次 API 调用最大文本数
_MAX_CHARS = 24000            # 单个文本最大字符数（约 8K tokens）
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_MODEL = "text-embedding-v4"
_MIN_INTERVAL = 0.1           # API 调用最小间隔（秒）


class Embedder:
    """向量嵌入服务，封装 DashScope 嵌入 API 调用。

    提供批量嵌入（用于索引构建）和单条嵌入（用于查询）两种接口，
    内置重试机制和限流控制。

    Attributes:
        config: 索引配置，包含 API 密钥和嵌入维度
    """

    def __init__(self, config: IndexingConfig):
        self.config = config
        self._dimension = config.embedding_dim
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """获取或创建 DashScope OpenAI 兼容客户端（懒加载）。"""
        if self._client is None:
            api_key = self.config.get_dashscope_api_key()
            if not api_key:
                raise ValueError(
                    "DashScope API key not configured. Set it via one of:\n"
                    "  1. .env file:  DASHSCOPE_API_KEY=sk-xxx\n"
                    "  2. YAML config: indexing.dashscope_api_key: sk-xxx\n"
                    "  3. Env var:    export DASHSCOPE_API_KEY=sk-xxx\n"
                    "Get your key from https://dashscope.console.aliyun.com/"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=_DASHSCOPE_BASE_URL,
            )
            logger.info("DashScope client initialized (model=%s)", _DASHSCOPE_MODEL)
        return self._client

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调用 DashScope 嵌入 API，返回嵌入向量列表。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            与输入一一对应的嵌入向量列表
        """
        resp = self.client.embeddings.create(
            model=_DASHSCOPE_MODEL,
            input=texts,
        )
        items = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in items]

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """批量计算代码分块的嵌入向量。

        分批调用 API，每批 _API_BATCH_SIZE 条，失败自动重试一次。
        重试仍失败时填充零向量作为降级处理。

        Args:
            chunks: 待嵌入的代码分块列表

        Returns:
            填充了 embedding 字段的同一列表
        """
        total = len(chunks)
        logger.info("Computing embeddings for %d chunks via DashScope API...", total)

        for i in range(0, total, _API_BATCH_SIZE):
            batch = chunks[i:i + _API_BATCH_SIZE]
            texts = [c.content[:_MAX_CHARS] for c in batch]

            try:
                embeddings = self._call_api(texts)
            except Exception as e:
                logger.error("API call failed at batch %d/%d: %s", i, total, e)
                time.sleep(2)
                try:
                    embeddings = self._call_api(texts)
                except Exception as e2:
                    logger.error("Retry failed: %s", e2)
                    embeddings = [[0.0] * self._dimension for _ in batch]

            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

            done = min(i + len(batch), total)
            if done % 200 == 0 or done == total:
                logger.info("Embedded %d/%d chunks", done, total)

            time.sleep(_MIN_INTERVAL)

        logger.info("Embeddings complete: %d chunks", total)
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """计算单条查询文本的嵌入向量。"""
        result = self._call_api([query])
        return result[0]

    @property
    def dimension(self) -> int:
        """嵌入向量的维度。"""
        return self._dimension
