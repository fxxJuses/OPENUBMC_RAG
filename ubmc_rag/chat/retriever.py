"""LangChain BaseRetriever wrapping openUBMC HybridSearchEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ubmc_rag.search.hybrid_search import HybridSearchEngine

if TYPE_CHECKING:
    from ubmc_rag.config.settings import AppConfig


class UBMCRetriever(BaseRetriever):
    """Retriever that delegates to our HybridSearchEngine."""

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
        if self.engine is None:
            return []

        results = self.engine.search(query, top_k=self.top_k)

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
    """Load index and create a configured retriever."""
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
