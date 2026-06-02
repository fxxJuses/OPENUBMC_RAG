"""Tests for Reranker with cross-encoder integration (迭代6-P0)."""

import pytest
from ubmc_rag.config.settings import SearchConfig
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.reranker import Reranker


def _make_chunk(
    chunk_id: str,
    content: str,
    file_path: str = "test.lua",
    repo_name: str = "test",
    language: str = "lua",
) -> CodeChunk:
    """Helper: create a minimal CodeChunk for testing."""
    return CodeChunk(
        chunk_id=chunk_id,
        content=content,
        file_path=file_path,
        repo_name=repo_name,
        language=language,
        component_name=repo_name,
        start_line=1,
        end_line=1,
        chunk_type="function",
    )


def _make_result(
    chunk: CodeChunk,
    score: float = 0.5,
    source: str = "hybrid",
) -> SearchResult:
    """Helper: create a SearchResult."""
    return SearchResult(chunk=chunk, score=score, source=source)


@pytest.fixture
def config():
    """Default search config for testing."""
    return SearchConfig(
        rrf_k=60,
        bm25_weight=0.5,
        dense_weight=0.5,
        cross_encoder_enabled=False,
    )




def test_reranker_rrf_fuse(config):
    """Test RRF fusion without cross-encoder."""
    reranker = Reranker(config)

    chunks = [
        _make_chunk("1", "sensor reading function"),
        _make_chunk("2", "firmware update function"),
        _make_chunk("3", "power config function"),
    ]

    dense = [
        _make_result(chunks[0], score=0.9, source="dense"),
        _make_result(chunks[1], score=0.7, source="dense"),
        _make_result(chunks[2], score=0.5, source="dense"),
    ]
    bm25 = [
        _make_result(chunks[1], score=2.5, source="bm25"),
        _make_result(chunks[0], score=2.0, source="bm25"),
        _make_result(chunks[2], score=1.0, source="bm25"),
    ]

    result = reranker.rerank(
        dense_results=dense,
        bm25_results=bm25,
        query="sensor reading",
        top_k=5,
    )

    assert len(result) > 0
    assert all(isinstance(r, SearchResult) for r in result)


def test_reranker_skip_boost(config):
    """Test skip_boost mode produces results without boosting."""
    reranker = Reranker(config)

    chunks = [
        _make_chunk("1", "test content"),
        _make_chunk("2", "other content"),
    ]
    dense = [_make_result(chunks[0], score=0.8, source="dense")]
    bm25 = [_make_result(chunks[1], score=2.0, source="bm25")]

    result = reranker.rerank(
        dense_results=dense,
        bm25_results=bm25,
        query="test",
        top_k=5,
        skip_boost=True,
    )

    assert len(result) > 0


def test_reranker_dashscope_disabled_by_default(config):
    """When dashscope_reranker_enabled=False, DS reranker is not initialized."""
    reranker = Reranker(config)

    chunks = [_make_chunk("1", "test")]
    dense = [_make_result(chunks[0], score=0.8, source="dense")]
    bm25 = [_make_result(chunks[0], score=2.0, source="bm25")]

    result = reranker.rerank(
        dense_results=dense,
        bm25_results=bm25,
        query="test",
        top_k=5,
    )

    assert len(result) > 0
    # DashScope reranker should not have been initialized
    assert reranker._dashscope_reranker is None


def test_reranker_dashscope_enabled():
    """When dashscope_reranker_enabled=True, DS reranker is initialized."""
    config = SearchConfig(
        rrf_k=60,
        bm25_weight=0.5,
        dense_weight=0.5,
        dashscope_reranker_enabled=True,
    )
    reranker = Reranker(config)

    # DashScope reranker should be initialized (even without API key)
    assert reranker._dashscope_reranker is not None


def test_reranker_empty_input(config):
    """Empty input returns empty list."""
    reranker = Reranker(config)
    result = reranker.rerank([], [], query="test")
    assert result == []


def test_reranker_single_source(config):
    """RRF fusion works with only one source (dense or bm25)."""
    reranker = Reranker(config)

    chunks = [
        _make_chunk("1", "test content"),
        _make_chunk("2", "other content"),
    ]
    dense = [
        _make_result(chunks[0], score=0.9, source="dense"),
        _make_result(chunks[1], score=0.5, source="dense"),
    ]

    result = reranker.rerank(
        dense_results=dense,
        bm25_results=[],
        query="test",
        top_k=2,
    )

    assert len(result) <= 2
    assert len(result) > 0


def test_reranker_diversity_cap(config):
    """Results from same file are capped by diversity_max_per_file."""
    config.diversity_max_per_file = 1
    reranker = Reranker(config)

    chunks = [
        _make_chunk("1", "a", file_path="same.lua"),
        _make_chunk("2", "b", file_path="same.lua"),
        _make_chunk("3", "c", file_path="other.lua"),
    ]
    dense = [
        _make_result(chunks[0], score=0.9, source="dense"),
        _make_result(chunks[1], score=0.8, source="dense"),
        _make_result(chunks[2], score=0.7, source="dense"),
    ]

    result = reranker.rerank(
        dense_results=dense,
        bm25_results=[],
        query="test",
        top_k=3,
    )

    # Chunks 0 and 1 share same file; 1 should be penalized
    assert len(result) == 3
    # Verify that chunk 1 (first in same.lua) ranks above chunk 2 (second in same.lua, penalized)
    ids = [r.chunk.chunk_id for r in result]
    idx_1 = ids.index("1")  # first same.lua
    idx_2 = ids.index("2")  # second same.lua (penalized)
    assert idx_1 < idx_2, (
        f"Chunk 1 (first in same.lua) should rank above chunk 2 (penalized); got {ids}"
    )
