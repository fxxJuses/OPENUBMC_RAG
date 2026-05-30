"""LangChain BaseRetriever 适配器，封装 HybridSearchEngine。

提供与 LangChain 生态系统兼容的检索器接口，同时支持
单查询检索和多查询合并检索（用于 LLM 子查询模式）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ubmc_rag.search.hybrid_search import HybridSearchEngine

if TYPE_CHECKING:
    from ubmc_rag.config.settings import AppConfig


class UBMCRetriever(BaseRetriever):
    """openUBMC 代码检索器，委托 HybridSearchEngine 执行搜索。

    兼容 LangChain 的 BaseRetriever 接口，可无缝接入 LangChain Agent。

    Attributes:
        top_k: 单次检索返回的最大结果数
        engine: 底层的混合搜索引擎实例
    """

    top_k: int = 5
    engine: HybridSearchEngine | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """执行单查询检索，返回 LangChain Document 列表。

        Args:
            query: 检索查询文本
            run_manager: LangChain 回调管理器

        Returns:
            匹配的 Document 列表，元数据包含文件路径、分数等信息
        """
        if self.engine is None:
            return []

        results = self.engine.search(query, top_k=self.top_k)
        return self._results_to_docs(results)

    def multi_query_search(self, queries: list[str], top_k: int | None = None) -> list[Document]:
        """执行多查询检索，合并去重后返回结果。

        对多个子查询分别检索，按 chunk_id 去重并保留最高分数，
        最终按分数降序排列。

        Args:
            queries: 子查询文本列表
            top_k: 返回的最大结果数

        Returns:
            合并去重后的 Document 列表
        """
        if self.engine is None:
            return []

        top_k = top_k or self.top_k
        merged: dict[str, tuple] = {}  # chunk_id -> (result, score)

        for q in queries:
            results = self.engine.search(q, top_k=top_k)
            for r in results:
                d = r.to_dict()
                chunk_id = f"{d['repo']}/{d['file_path']}:{d['start_line']}-{d['end_line']}"
                score = d["score"]
                if chunk_id not in merged or score > merged[chunk_id][1]:
                    merged[chunk_id] = (r, score)

        ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return self._results_to_docs([r for r, _ in ranked[:top_k]])

    @staticmethod
    def _results_to_docs(results) -> list[Document]:
        """将 SearchResult 列表转换为 LangChain Document 列表。"""
        docs = []
        for r in results:
            d = r.to_dict()
            docs.append(Document(
                page_content=d["content"],
                metadata={
                    "file_path": d["file_path"],
                    "repo": d["repo"],
                    "language": d["language"],
                    "chunk_type": d["chunk_type"],
                    "start_line": d["start_line"],
                    "end_line": d["end_line"],
                    "score": d["score"],
                },
            ))
        return docs


def create_retriever(config: AppConfig) -> UBMCRetriever:
    """加载索引并创建配置好的检索器实例。

    Args:
        config: 应用配置

    Returns:
        初始化完成的 UBMCRetriever 实例

    Raises:
        RuntimeError: 索引不存在时抛出
    """
    from ubmc_rag.indexing.index_manager import IndexManager

    index_mgr = IndexManager(config)
    if not index_mgr.load_index():
        raise RuntimeError("No index found. Run 'ubmc-rag index' first.")

    chunks = index_mgr.get_all_chunks()
    engine = HybridSearchEngine(
        embedder=index_mgr.embedder,
        vector_store=index_mgr.vector_store,
        bm25=index_mgr.bm25,
        config=config,
    )
    engine.set_chunk_index(chunks)

    return UBMCRetriever(engine=engine, top_k=5)
