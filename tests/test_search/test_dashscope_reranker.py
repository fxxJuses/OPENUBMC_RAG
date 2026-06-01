"""Tests for DashScope qwen3-rerank integration (迭代6-B)."""

from unittest.mock import patch, MagicMock

import pytest

from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.dashscope_reranker import DashScopeReranker


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


class TestDashScopeReranker:
    """Tests for DashScopeReranker basic functionality."""

    def test_empty_candidates(self):
        """Empty candidate list returns empty list."""
        reranker = DashScopeReranker(api_key="sk-test")
        result = reranker.rerank("test query", [])
        assert result == []

    def test_not_available_without_key(self):
        """Without API key, available returns False."""
        reranker = DashScopeReranker(api_key="")
        assert reranker.available is False

    def test_available_with_key(self):
        """With API key, available returns True."""
        reranker = DashScopeReranker(api_key="sk-test")
        assert reranker.available is True

    def test_no_api_key_returns_unreranked(self):
        """Without API key, rerank returns candidates unchanged."""
        reranker = DashScopeReranker(api_key="")
        chunks = [
            _make_chunk("1", "sensor reading function"),
            _make_chunk("2", "firmware update function"),
        ]
        candidates = [
            _make_result(chunks[0], score=0.9),
            _make_result(chunks[1], score=0.5),
        ]

        result = reranker.rerank("sensor", candidates)
        assert len(result) == 2
        # Should return in original order (no reranking)
        assert result[0].chunk.chunk_id == "1"
        assert result[1].chunk.chunk_id == "2"

    def test_api_rerank_success(self):
        """Successful API call returns reranked results."""
        reranker = DashScopeReranker(api_key="sk-test")

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

        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 2, "relevance_score": 0.60},
                    {"index": 1, "relevance_score": 0.30},
                ]
            }
        }

        with patch("requests.post", return_value=mock_response):
            result = reranker.rerank("sensor reading", candidates)

        assert len(result) == 3
        # Chunk 1 (index 0) should rank first with highest score
        assert result[0].chunk.chunk_id == "1"
        assert result[0].score == 0.95
        assert result[0].source == "dashscope_rerank"

    def test_api_rerank_top_k(self):
        """top_k parameter limits results."""
        reranker = DashScopeReranker(api_key="sk-test")

        chunks = [_make_chunk(str(i), f"content {i}") for i in range(5)]
        candidates = [
            _make_result(c, score=0.1 * (5 - i)) for i, c in enumerate(chunks)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 2, "relevance_score": 0.7},
                ]
            }
        }

        with patch("requests.post", return_value=mock_response):
            result = reranker.rerank("test", candidates, top_k=3)

        assert len(result) == 3

    def test_api_rerank_preserves_chunk_data(self):
        """Reranking preserves original chunk metadata."""
        reranker = DashScopeReranker(api_key="sk-test")

        chunk = _make_chunk(
            "1", "IPMI sensor reading function",
            file_path="ipmi/sensor.lua",
            repo_name="sensor_mgmt",
        )
        candidates = [_make_result(chunk, score=0.5)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "results": [{"index": 0, "relevance_score": 0.88}]
            }
        }

        with patch("requests.post", return_value=mock_response):
            result = reranker.rerank("IPMI sensor", candidates)

        assert len(result) == 1
        assert result[0].chunk.file_path == "ipmi/sensor.lua"
        assert result[0].chunk.repo_name == "sensor_mgmt"
        assert result[0].chunk.content == "IPMI sensor reading function"
        assert result[0].source == "dashscope_rerank"

    def test_api_rerank_http_error_fallback(self):
        """HTTP error falls back to un-reranked candidates."""
        reranker = DashScopeReranker(api_key="sk-test")

        chunks = [
            _make_chunk("1", "test content"),
            _make_chunk("2", "other content"),
        ]
        candidates = [
            _make_result(chunks[0], score=0.9),
            _make_result(chunks[1], score=0.5),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_response):
            result = reranker.rerank("test", candidates)

        # Should fall back to original order
        assert len(result) == 2
        assert result[0].chunk.chunk_id == "1"

    def test_api_rerank_network_error_fallback(self):
        """Network error falls back to un-reranked candidates."""
        reranker = DashScopeReranker(api_key="sk-test")

        chunk = _make_chunk("1", "test content")
        candidates = [_make_result(chunk, score=0.5)]

        with patch("requests.post", side_effect=Exception("Connection refused")):
            result = reranker.rerank("test", candidates)

        assert len(result) == 1
        assert result[0].chunk.chunk_id == "1"

    def test_api_rerank_empty_results_fallback(self):
        """Empty API results fall back to un-reranked candidates."""
        reranker = DashScopeReranker(api_key="sk-test")

        chunk = _make_chunk("1", "test content")
        candidates = [_make_result(chunk, score=0.5)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"results": []}
        }

        with patch("requests.post", return_value=mock_response):
            result = reranker.rerank("test", candidates)

        assert len(result) == 1
        assert result[0].chunk.chunk_id == "1"

    def test_factory_function(self):
        """Factory function creates a valid reranker."""
        from ubmc_rag.search.dashscope_reranker import create_dashscope_reranker

        reranker = create_dashscope_reranker(
            api_key="sk-test",
            model="qwen3-rerank",
            top_n=15,
        )
        assert reranker.model == "qwen3-rerank"
        assert reranker.top_n == 15
        assert reranker.available is True
