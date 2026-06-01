"""Tests for cross-encoder reranker (迭代6-P0)."""

from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.cross_encoder import CrossEncoderReranker


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


def test_cross_encoder_empty_candidates():
    """Empty candidate list returns empty list."""
    ce = CrossEncoderReranker(device="cpu")
    result = ce.rerank("test query", [])
    assert result == []


def test_cross_encoder_heuristic_fallback():
    """Heuristic fallback reranks by keyword overlap + original score."""
    ce = CrossEncoderReranker(device="cpu")

    # Force fallback mode by not loading a model
    ce.fallback = True
    ce.model = None

    chunks = [
        _make_chunk("1", "function getSensorReading() reads IPMI sensor data"),
        _make_chunk("2", "function updateFirmware() handles firmware updates"),
        _make_chunk("3", "function configPower() manages power supply"),
    ]
    candidates = [
        _make_result(chunks[0], score=0.3),
        _make_result(chunks[1], score=0.5),
        _make_result(chunks[2], score=0.4),
    ]

    # Query about sensor data - chunk 0 should rank high
    result = ce.rerank("sensor reading data", candidates)
    assert len(result) == 3
    # All should have cross_encoder_fallback source
    for r in result:
        assert r.source == "cross_encoder_fallback"

    # Results should be sorted by score descending
    for i in range(len(result) - 1):
        assert result[i].score >= result[i + 1].score


def test_cross_encoder_top_k():
    """top_k parameter limits results."""
    ce = CrossEncoderReranker(device="cpu")
    ce.fallback = True
    ce.model = None

    chunks = [
        _make_chunk(str(i), f"content {i}") for i in range(10)
    ]
    candidates = [
        _make_result(c, score=0.1 * i) for i, c in enumerate(chunks)
    ]

    result = ce.rerank("test", candidates, top_k=3)
    assert len(result) == 3


def test_cross_encoder_is_fallback():
    """is_fallback property reflects mode."""
    ce = CrossEncoderReranker(device="cpu")
    ce.fallback = True
    assert ce.is_fallback is True
    ce.fallback = False
    assert ce.is_fallback is False


def test_cross_encoder_single_candidate():
    """Single candidate returns single result."""
    ce = CrossEncoderReranker(device="cpu")
    ce.fallback = True
    ce.model = None

    chunk = _make_chunk("1", "test content")
    candidates = [_make_result(chunk, score=0.5)]

    result = ce.rerank("test", candidates)
    assert len(result) == 1
    assert result[0].chunk.chunk_id == "1"


def test_cross_encoder_preserves_chunk_data():
    """Reranking preserves original chunk data (file_path, repo, etc.)."""
    ce = CrossEncoderReranker(device="cpu")
    ce.fallback = True
    ce.model = None

    chunk = _make_chunk(
        "1", "IPMI sensor reading function",
        file_path="ipmi/sensor.lua",
        repo_name="sensor_mgmt",
    )
    candidates = [_make_result(chunk, score=0.5)]

    result = ce.rerank("IPMI sensor", candidates)
    assert len(result) == 1
    assert result[0].chunk.file_path == "ipmi/sensor.lua"
    assert result[0].chunk.repo_name == "sensor_mgmt"
    assert result[0].chunk.content == "IPMI sensor reading function"
