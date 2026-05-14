"""ChromaDB vector store manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb

from ubmc_rag.config.settings import IndexingConfig
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config: IndexingConfig):
        self.config = config
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            persist_dir = ensure_dir(self.config.persist_dir)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Add chunks with pre-computed embeddings to ChromaDB."""
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
        """Search for similar chunks using vector similarity."""
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
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.config.chroma_collection)
        except Exception:
            pass
        self._collection = None
        logger.info("Collection reset")
