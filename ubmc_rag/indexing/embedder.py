"""Embedding via DashScope text-embedding-v4 (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import os
import time

from openai import OpenAI

from ubmc_rag.config.settings import IndexingConfig
from ubmc_rag.models.code_chunk import CodeChunk

logger = logging.getLogger(__name__)

_API_BATCH_SIZE = 10
_MAX_CHARS = 24000  # ~8K tokens rough estimate (1 token ≈ 3 chars for code)
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_MODEL = "text-embedding-v4"
_MIN_INTERVAL = 0.1


class Embedder:
    def __init__(self, config: IndexingConfig):
        self.config = config
        self._dimension = config.embedding_dim
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
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
        resp = self.client.embeddings.create(
            model=_DASHSCOPE_MODEL,
            input=texts,
        )
        # Sort by index to ensure order matches input
        items = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in items]

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Compute embeddings via DashScope API in batches."""
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
        """Compute embedding for a single query string."""
        result = self._call_api([query])
        return result[0]

    @property
    def dimension(self) -> int:
        return self._dimension
