"""Index manager — orchestrates vector + BM25 indexing with incremental updates."""

from __future__ import annotations

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
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedder = Embedder(config.indexing)
        self.vector_store = VectorStore(config.indexing)
        self.bm25 = BM25Index()
        self._chunks_index: dict[str, CodeChunk] = {}
        self._checksums_path = Path(config.indexing.persist_dir) / "checksums.json"

    def build_index(self, chunks: list[CodeChunk], full_rebuild: bool = False) -> None:
        """Build or rebuild the full index. Embeds and writes in batches to control memory."""
        import gc

        if full_rebuild:
            self.vector_store.reset()

        total = len(chunks)
        logger.info("Building index for %d chunks...", total)

        # Ensure persist directory exists
        ensure_dir(self.config.indexing.persist_dir)

        # Build BM25 index first (no GPU/memory pressure)
        self.bm25.build(chunks)
        bm25_path = Path(self.config.indexing.persist_dir) / "bm25_index.json"
        self.bm25.save(bm25_path)

        # Process embedding + ChromaDB write in batches
        batch_size = 64
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            # Compute embeddings for this batch
            batch = self.embedder.embed_chunks(batch)

            # Immediately write to ChromaDB and free embedding memory
            self.vector_store.add_chunks(batch)

            # Store in chunk index (without embeddings to save memory)
            for c in batch:
                c.embedding = None
                self._chunks_index[c.chunk_id] = c

            logger.info("Indexed %d/%d chunks", min(i + batch_size, total), total)
            gc.collect()

        # Save checksums for incremental updates
        self._save_checksums(chunks)

        logger.info(
            "Index built: %d chunks, ChromaDB has %d total",
            total, self.vector_store.count(),
        )

    def load_index(self) -> bool:
        """Load existing index from disk."""
        bm25_path = Path(self.config.indexing.persist_dir) / "bm25_index.json"
        loaded = self.bm25.load(bm25_path)

        if self.vector_store.count() > 0:
            logger.info("Loaded existing index: %d chunks in ChromaDB", self.vector_store.count())
            return True

        return loaded

    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        return self._chunks_index.get(chunk_id)

    def get_all_chunks(self) -> list[CodeChunk]:
        return list(self._chunks_index.values())

    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self._chunks_index),
            "chroma_count": self.vector_store.count(),
            "bm25_docs": len(self.bm25.get_chunk_ids()),
        }

    def _save_checksums(self, chunks: list[CodeChunk]) -> None:
        checksums = {}
        for c in chunks:
            key = f"{c.repo_name}:{c.file_path}"
            checksums[key] = hashlib.md5(c.content.encode()).hexdigest()
        self._checksums_path.parent.mkdir(parents=True, exist_ok=True)
        self._checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
