"""索引管理器 —— 协调向量索引和 BM25 索引的构建、加载和查询。

统一管理嵌入服务（Embedder）、向量存储（VectorStore）和
BM25 索引（BM25Index）的生命周期，提供分批构建、增量更新
和内存优化等能力。
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.bm25_index import BM25Index
from ubmc_rag.indexing.embedder import Embedder
from ubmc_rag.indexing.vector_store import VectorStore
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


class IndexManager:
    """索引管理器，协调向量索引和 BM25 索引的全生命周期。

    负责索引的构建（含分批嵌入和内存管理）、加载、查询和统计。
    维护一个内存中的分块索引用于结果重建。

    Attributes:
        config: 应用配置
        embedder: 向量嵌入服务
        vector_store: ChromaDB 向量存储
        bm25: BM25 关键词索引
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = Embedder(config.indexing)
        self.vector_store = VectorStore(config.indexing)
        self.bm25 = BM25Index()
        self._chunks_index: dict[str, CodeChunk] = {}
        self._checksums_path = Path(config.indexing.persist_dir) / "checksums.json"

    def build_index(self, chunks: list[CodeChunk], full_rebuild: bool = False) -> None:
        """构建或重建索引，分批处理以控制内存占用。

        处理流程：
        1. 先构建 BM25 索引（无 GPU/内存压力）
        2. 分批计算嵌入并写入 ChromaDB（每批 64 条）
        3. 清除嵌入向量释放内存
        4. 保存文件校验和用于后续增量更新

        Args:
            chunks: 待索引的代码分块列表
            full_rebuild: 是否全量重建（会清空现有向量集合）
        """
        if full_rebuild:
            self.vector_store.reset()

        total = len(chunks)
        logger.info("Building index for %d chunks...", total)

        ensure_dir(self.config.indexing.persist_dir)

        # 构建 BM25 索引
        self.bm25.build(chunks)
        bm25_path = Path(self.config.indexing.persist_dir) / "bm25_index.json"
        self.bm25.save(bm25_path)

        # 分批计算嵌入并写入向量库
        batch_size = 64
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            batch = self.embedder.embed_chunks(batch)
            self.vector_store.add_chunks(batch)

            # 清除嵌入以释放内存，同时写入分块索引
            for c in batch:
                c.embedding = None
                self._chunks_index[c.chunk_id] = c

            logger.info("Indexed %d/%d chunks", min(i + batch_size, total), total)
            gc.collect()

        # 保存文件校验和，供增量更新时比较
        self._save_checksums(chunks)

        logger.info(
            "Index built: %d chunks, ChromaDB has %d total",
            total, self.vector_store.count(),
        )

    def load_index(self) -> bool:
        """从磁盘加载现有索引。

        Returns:
            成功加载返回 True，索引不存在返回 False
        """
        bm25_path = Path(self.config.indexing.persist_dir) / "bm25_index.json"
        loaded = self.bm25.load(bm25_path)

        if self.vector_store.count() > 0:
            logger.info(
                "Loaded existing index: %d chunks in ChromaDB",
                self.vector_store.count(),
            )
            return True

        return loaded

    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """根据 ID 获取单个代码分块。"""
        return self._chunks_index.get(chunk_id)

    def get_all_chunks(self) -> list[CodeChunk]:
        """获取内存索引中的所有代码分块。"""
        return list(self._chunks_index.values())

    def get_stats(self) -> dict:
        """返回索引的统计信息。"""
        return {
            "total_chunks": len(self._chunks_index),
            "chroma_count": self.vector_store.count(),
            "bm25_docs": len(self.bm25.get_chunk_ids()),
        }

    def _save_checksums(self, chunks: list[CodeChunk]) -> None:
        """保存文件内容的 MD5 校验和，用于增量更新时的变更检测。"""
        checksums = {}
        for c in chunks:
            key = f"{c.repo_name}:{c.file_path}"
            checksums[key] = hashlib.md5(c.content.encode()).hexdigest()
        self._checksums_path.parent.mkdir(parents=True, exist_ok=True)
        self._checksums_path.write_text(
            json.dumps(checksums, indent=2), encoding="utf-8"
        )
