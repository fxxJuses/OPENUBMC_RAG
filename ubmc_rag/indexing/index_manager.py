"""索引管理器 —— 协调向量索引和 BM25 索引的构建、加载和查询。

统一管理嵌入服务（Embedder）、向量存储（VectorStore）和
BM25 索引（BM25Index）的生命周期，提供分批构建、增量更新
和内存优化等能力。支持代码和文档两个独立的 collection。
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
    支持代码 (openubmc_code) 和文档 (openubmc_docs) 两个独立 collection。

    Attributes:
        config: 应用配置
        embedder: 向量嵌入服务
        vector_store: ChromaDB 代码向量存储
        docs_vector_store: ChromaDB 文档向量存储
        bm25: 代码 BM25 关键词索引
        docs_bm25: 文档 BM25 关键词索引
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = Embedder(config.indexing)
        # 代码索引
        self.vector_store = VectorStore(config.indexing)
        self.bm25 = BM25Index()
        # 文档索引（使用独立的 collection）
        docs_config = config.indexing.model_copy(
            update={"chroma_collection": config.indexing.docs_collection}
        )
        self.docs_vector_store = VectorStore(docs_config)
        self.docs_bm25 = BM25Index()

        self._chunks_index: dict[str, CodeChunk] = {}
        self._docs_chunks_index: dict[str, CodeChunk] = {}
        self._checksums_path = Path(config.indexing.persist_dir) / "checksums.json"
        self._docs_checksums_path = Path(config.indexing.persist_dir) / "docs_checksums.json"

    def build_index(self, chunks: list[CodeChunk], full_rebuild: bool = False) -> None:
        """构建或重建代码索引，分批处理以控制内存占用。

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
        logger.info("Building code index for %d chunks...", total)

        ensure_dir(self.config.indexing.persist_dir)

        self.bm25.build(chunks)
        bm25_path = Path(self.config.indexing.persist_dir) / "bm25_index.json"
        self.bm25.save(bm25_path)

        self._build_vector_index(chunks, self.vector_store, self._chunks_index)

        self._save_checksums(chunks, self._checksums_path)

        logger.info(
            "Code index built: %d chunks, ChromaDB has %d total",
            total, self.vector_store.count(),
        )

    def build_docs_index(self, chunks: list[CodeChunk], full_rebuild: bool = False) -> None:
        """构建或重建文档索引，独立于代码索引。

        Args:
            chunks: 文档分块列表
            full_rebuild: 是否全量重建
        """
        if full_rebuild:
            self.docs_vector_store.reset()

        total = len(chunks)
        logger.info("Building docs index for %d chunks...", total)

        ensure_dir(self.config.indexing.persist_dir)

        self.docs_bm25.build(chunks)
        docs_bm25_path = Path(self.config.indexing.persist_dir) / "bm25_docs_index.json"
        self.docs_bm25.save(docs_bm25_path)

        self._build_vector_index(chunks, self.docs_vector_store, self._docs_chunks_index)

        self._save_checksums(chunks, self._docs_checksums_path)

        logger.info(
            "Docs index built: %d chunks, ChromaDB has %d total",
            total, self.docs_vector_store.count(),
        )

    def _build_vector_index(
        self, chunks: list[CodeChunk],
        vector_store: VectorStore, chunks_index: dict[str, CodeChunk],
    ) -> None:
        """分批计算嵌入并写入向量库的通用方法。"""
        total = len(chunks)
        batch_size = getattr(self.config.indexing, 'embedding_batch_size', 256) or 256
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            batch = self.embedder.embed_chunks(batch)
            vector_store.add_chunks(batch)

            for c in batch:
                c.embedding = None
                chunks_index[c.chunk_id] = c

            logger.info("Indexed %d/%d chunks", min(i + batch_size, total), total)
            gc.collect()

    def load_index(self) -> bool:
        """从磁盘加载现有代码索引。

        Returns:
            成功加载返回 True，索引不存在返回 False
        """
        bm25_path = Path(self.config.indexing.persist_dir) / "bm25_index.json"
        loaded = self.bm25.load(bm25_path)

        chroma_count = self.vector_store.count()
        if chroma_count > 0:
            self._load_chunks_from_chroma(
                self.vector_store, self._chunks_index
            )
            logger.info(
                "Loaded code index: %d chunks in ChromaDB, %d in memory",
                chroma_count, len(self._chunks_index),
            )
            return True

        return loaded

    def load_docs_index(self) -> bool:
        """从磁盘加载现有文档索引。

        Returns:
            成功加载返回 True，索引不存在返回 False
        """
        docs_bm25_path = Path(self.config.indexing.persist_dir) / "bm25_docs_index.json"
        loaded = self.docs_bm25.load(docs_bm25_path)

        chroma_count = self.docs_vector_store.count()
        if chroma_count > 0:
            self._load_chunks_from_chroma(
                self.docs_vector_store, self._docs_chunks_index
            )
            logger.info(
                "Loaded docs index: %d chunks in ChromaDB, %d in memory",
                chroma_count, len(self._docs_chunks_index),
            )
            return True

        return loaded

    def _load_chunks_from_chroma(
        self, vector_store: VectorStore, chunks_index: dict[str, CodeChunk],
    ) -> None:
        """从 ChromaDB 集合中恢复所有分块到内存索引。"""
        collection = vector_store.collection
        result = collection.get(include=["documents", "metadatas"])
        if not result["ids"]:
            return

        for i, chunk_id in enumerate(result["ids"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            content = result["documents"][i] if result["documents"] else ""
            chunks_index[chunk_id] = CodeChunk.from_chroma_metadata(
                chunk_id=chunk_id, content=content, meta=meta,
            )

    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """根据 ID 获取单个代码分块。"""
        return self._chunks_index.get(chunk_id) or self._docs_chunks_index.get(chunk_id)

    def get_all_chunks(self) -> list[CodeChunk]:
        """获取内存索引中的所有代码分块。"""
        return list(self._chunks_index.values())

    def get_all_docs_chunks(self) -> list[CodeChunk]:
        """获取内存索引中的所有文档分块。"""
        return list(self._docs_chunks_index.values())

    def get_stats(self) -> dict:
        """返回索引的统计信息。"""
        return {
            "code_chunks": len(self._chunks_index),
            "chroma_count": self.vector_store.count(),
            "bm25_docs": len(self.bm25.get_chunk_ids()),
            "docs_chunks": len(self._docs_chunks_index),
            "docs_chroma_count": self.docs_vector_store.count(),
            "docs_bm25_docs": len(self.docs_bm25.get_chunk_ids()),
        }

    def search_docs_vector(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """在文档向量库中搜索。"""
        return self.docs_vector_store.search(query_embedding, top_k=top_k)

    def search_docs_bm25(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """在文档 BM25 索引中搜索。"""
        return self.docs_bm25.search(query, top_k=top_k)

    def _save_checksums(self, chunks: list[CodeChunk], path: Path) -> None:
        """保存文件内容的 MD5 校验和，用于增量更新时的变更检测。"""
        checksums = {}
        for c in chunks:
            key = f"{c.repo_name}:{c.file_path}"
            checksums[key] = hashlib.md5(c.content.encode()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(checksums, indent=2), encoding="utf-8"
        )
