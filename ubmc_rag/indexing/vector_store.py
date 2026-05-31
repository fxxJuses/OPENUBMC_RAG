"""ChromaDB 向量存储管理器。

负责向量数据库的初始化、文档写入、向量相似度搜索和集合管理。
使用 cosine 距离作为相似度度量。
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb

from ubmc_rag.config.settings import IndexingConfig
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储管理器，负责代码分块的向量索引和检索。

    使用 PersistentClient 将数据持久化到磁盘，支持 upsert 操作实现增量更新。

    Attributes:
        config: 索引配置，包含持久化目录和集合名称
    """

    def __init__(self, config: IndexingConfig):
        self.config = config
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """获取或创建 ChromaDB 持久化客户端（懒加载）。"""
        if self._client is None:
            persist_dir = ensure_dir(self.config.persist_dir)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """获取或创建 ChromaDB 集合（使用 cosine 距离度量）。"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """将带有预计算嵌入的代码分块写入 ChromaDB。

        分批写入以避免单次操作过大。如果某批次中部分分块缺少嵌入，
        则跳过整个批次。

        Args:
            chunks: 已计算嵌入的代码分块列表
        """
        if not chunks:
            return

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            embeddings = [c.embedding for c in batch if c.embedding is not None]
            metadatas = [c.to_chroma_metadata() for c in batch]

            if len(embeddings) != len(batch):
                logger.warning("Some chunks missing embeddings, skipping batch")
                continue

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        logger.info("Stored %d chunks in ChromaDB", len(chunks))

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """基于向量相似度搜索最相关的代码分块。

        Args:
            query_embedding: 查询文本的嵌入向量
            top_k: 返回结果数量
            where: ChromaDB 过滤条件，如 {"language": "lua"}

        Returns:
            匹配结果列表，每个元素包含 chunk_id, content, metadata, distance
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()) if self.collection.count() > 0 else top_k,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        items = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                item = {
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
                items.append(item)

        return items

    def count(self) -> int:
        """返回集合中的文档总数。"""
        return self.collection.count()

    def reset(self) -> None:
        """删除并重建集合，用于全量重建索引。"""
        try:
            self.client.delete_collection(self.config.chroma_collection)
        except Exception:
            pass
        self._collection = None
        logger.info("Collection reset")
