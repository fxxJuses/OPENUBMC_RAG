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

    支持代码 (openubmc_code) 和文档 (openubmc_docs) 两个独立 collection。
    文档索引支持增量构建：通过文件级 checksum 对比，只重新嵌入变更部分。

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
        """构建或增量更新文档索引。

        增量逻辑：
        1. 计算新 chunks 的文件级 checksums
        2. 与已保存的 checksums 对比，找出变更/新增/删除的文件
        3. 只对变更文件的 chunks 重新嵌入
        4. 未变更文件的 chunks 从 ChromaDB 复用
        5. BM25 全量重建（纯 CPU 操作，很快）

        Args:
            chunks: 当前解析出的全部文档分块列表
            full_rebuild: 是否全量重建（忽略增量逻辑）
        """
        if full_rebuild:
            self.docs_vector_store.reset()

        ensure_dir(self.config.indexing.persist_dir)

        # 计算当前 chunks 的文件级 checksums
        new_checksums = self._compute_checksums(chunks)

        if not full_rebuild:
            old_checksums = self._load_checksums(self._docs_checksums_path)

            if old_checksums:
                changed, new, deleted, unchanged = self._diff_checksums(
                    old_checksums, new_checksums
                )
                logger.info(
                    "Docs incremental: %d changed, %d new, %d deleted, %d unchanged",
                    len(changed), len(new), len(deleted), len(unchanged),
                )

                if not changed and not new and not deleted:
                    logger.info("Docs index is up to date, skipping")
                    return

                # 分离：只对变更/新增文件的 chunks 重新嵌入
                changed_files = changed | new
                chunks_to_embed = [c for c in chunks if c.file_path in changed_files]

                # 加载已有索引以获取未变更 chunks
                self.load_docs_index()

                # 删除已移除文件的旧 chunks
                self._delete_chunks_by_files(
                    self.docs_vector_store, self._docs_chunks_index, deleted
                )

                # 只嵌入变更部分
                if chunks_to_embed:
                    logger.info(
                        "Incremental embedding: %d chunks (vs %d total)",
                        len(chunks_to_embed), len(chunks),
                    )
                    self._build_vector_index(
                        chunks_to_embed, self.docs_vector_store,
                        self._docs_chunks_index,
                    )

                # BM25 全量重建（需要所有 chunks）
                all_chunks = list(self._docs_chunks_index.values())
                self.docs_bm25.build(all_chunks)
                docs_bm25_path = (
                    Path(self.config.indexing.persist_dir) / "bm25_docs_index.json"
                )
                self.docs_bm25.save(docs_bm25_path)

                # 保存新 checksums
                self._save_checksums_data(new_checksums, self._docs_checksums_path)

                logger.info(
                    "Docs index updated: %d total chunks (%d re-embedded)",
                    len(all_chunks), len(chunks_to_embed),
                )
                return

        # 全量构建（首次或 full_rebuild）
        total = len(chunks)
        logger.info("Building docs index (full) for %d chunks...", total)

        self.docs_bm25.build(chunks)
        docs_bm25_path = Path(self.config.indexing.persist_dir) / "bm25_docs_index.json"
        self.docs_bm25.save(docs_bm25_path)

        self._build_vector_index(chunks, self.docs_vector_store, self._docs_chunks_index)

        self._save_checksums_data(new_checksums, self._docs_checksums_path)

        logger.info(
            "Docs index built: %d chunks, ChromaDB has %d total",
            total, self.docs_vector_store.count(),
        )

    # ── 增量构建辅助方法 ──

    @staticmethod
    def _compute_checksums(chunks: list[CodeChunk]) -> dict[str, str]:
        """计算文件级 checksums：同一文件的所有 chunks 内容拼接后取 MD5。"""
        file_contents: dict[str, list[str]] = {}
        for c in chunks:
            key = f"{c.repo_name}:{c.file_path}"
            file_contents.setdefault(key, []).append(c.content)

        checksums = {}
        for key, contents in file_contents.items():
            combined = "\n".join(contents)
            checksums[key] = hashlib.md5(combined.encode()).hexdigest()
        return checksums

    @staticmethod
    def _diff_checksums(
        old: dict[str, str], new: dict[str, str],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """对比新旧 checksums，返回 (changed, new, deleted, unchanged) 文件集合。"""
        old_keys = set(old.keys())
        new_keys = set(new.keys())

        deleted = old_keys - new_keys
        added = new_keys - old_keys
        common = old_keys & new_keys

        changed = {k for k in common if old[k] != new[k]}
        unchanged = common - changed

        return changed, added, deleted, unchanged

    @staticmethod
    def _load_checksums(path: Path) -> dict[str, str]:
        """从磁盘加载 checksums。"""
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_checksums_data(checksums: dict[str, str], path: Path) -> None:
        """保存 checksums 到磁盘。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    def _delete_chunks_by_files(
        self, vector_store: VectorStore,
        chunks_index: dict[str, CodeChunk], deleted_files: set[str],
    ) -> None:
        """删除指定文件对应的所有 chunks（从 ChromaDB 和内存索引）。"""
        if not deleted_files:
            return

        # 找到属于已删除文件的 chunk IDs
        to_delete = [
            cid for cid, c in chunks_index.items()
            if f"{c.repo_name}:{c.file_path}" in deleted_files
        ]

        if to_delete:
            vector_store.collection.delete(ids=to_delete)
            for cid in to_delete:
                del chunks_index[cid]
            logger.info("Deleted %d chunks from removed files", len(to_delete))

    # ── 通用构建方法 ──

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

    # ── 索引加载 ──

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

    # ── 查询接口 ──

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

    # ── 旧接口兼容 ──

    def _save_checksums(self, chunks: list[CodeChunk], path: Path) -> None:
        """保存代码索引的 checksums（兼容旧接口）。"""
        checksums = self._compute_checksums(chunks)
        self._save_checksums_data(checksums, path)
