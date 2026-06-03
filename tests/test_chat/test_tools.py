"""Tests for the agent tools module."""

from unittest.mock import MagicMock

import pytest

from ubmc_rag.chat.tools import create_tools
from ubmc_rag.models.code_chunk import CodeChunk, Symbol
from ubmc_rag.models.search_result import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "c1",
    content: str = "local x = 1",
    file_path: str = "src/main.lua",
    repo_name: str = "sensor",
    language: str = "lua",
    start_line: int = 10,
    end_line: int = 20,
    symbols: list[Symbol] | None = None,
) -> CodeChunk:
    """Create a minimal CodeChunk for testing."""
    return CodeChunk(
        chunk_id=chunk_id,
        content=content,
        file_path=file_path,
        repo_name=repo_name,
        language=language,
        component_name=repo_name,
        start_line=start_line,
        end_line=end_line,
        chunk_type="function",
        symbols=symbols or [],
    )


def _make_result(
    chunk: CodeChunk | None = None,
    score: float = 0.85,
    source: str = "hybrid",
) -> SearchResult:
    """Create a SearchResult."""
    if chunk is None:
        chunk = _make_chunk()
    return SearchResult(chunk=chunk, score=score, source=source)


# ---------------------------------------------------------------------------
# TestFormatResults
# ---------------------------------------------------------------------------

class TestFormatResults:
    """Tests for _format_results helper."""

    def test_empty_results(self):
        from ubmc_rag.chat.tools import _format_results

        assert _format_results([]) == "No results found."

    def test_results_with_symbols(self):
        from ubmc_rag.chat.tools import _format_results

        sym = Symbol(name="init", kind="function", line_start=1, line_end=5, language="lua")
        chunk = _make_chunk(symbols=[sym])
        result = _make_result(chunk=chunk, score=0.9234)
        formatted = _format_results([result])

        assert "init(function)" in formatted
        assert "symbols:" in formatted

    def test_results_without_symbols(self):
        from ubmc_rag.chat.tools import _format_results

        chunk = _make_chunk(symbols=[])
        result = _make_result(chunk=chunk)
        formatted = _format_results([result])

        assert "symbols:" not in formatted

    def test_results_format_includes_repo_path_lines_score(self):
        from ubmc_rag.chat.tools import _format_results

        chunk = _make_chunk(
            file_path="src/sensor.lua",
            repo_name="sensor",
            start_line=42,
            end_line=55,
        )
        result = _make_result(chunk=chunk, score=0.7531)
        formatted = _format_results([result])

        assert "sensor/src/sensor.lua:42-55" in formatted
        assert "(score=0.7531)" in formatted
        assert "local x = 1" in formatted

    def test_multiple_results_separated(self):
        from ubmc_rag.chat.tools import _format_results

        r1 = _make_result(
            chunk=_make_chunk(chunk_id="c1", content="aaa"),
            score=0.9,
        )
        r2 = _make_result(
            chunk=_make_chunk(chunk_id="c2", content="bbb"),
            score=0.8,
        )
        formatted = _format_results([r1, r2])

        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "---" in formatted


# ---------------------------------------------------------------------------
# TestSearchCodeIntentHint
# ---------------------------------------------------------------------------

class TestSearchCodeIntentHint:
    """Tests that search_code maps intent_hint to is_code_query correctly."""

    @pytest.fixture()
    def tools(self):
        engine = MagicMock()
        engine.search.return_value = []
        index_mgr = MagicMock()
        return create_tools(engine, index_mgr), engine

    def _get_search_code(self, tools):
        for t in tools:
            if t.name == "search_code":
                return t
        raise AssertionError("search_code tool not found")

    def test_intent_hint_code_passes_is_code_query_true(self, tools):
        tool_list, engine = tools
        search_code = self._get_search_code(tool_list)

        search_code.invoke({"query": "init", "intent_hint": "code"})

        engine.search.assert_called_once()
        call_kwargs = engine.search.call_args
        assert call_kwargs.kwargs.get("is_code_query") is True or (
            "is_code_query" in call_kwargs[1] and call_kwargs[1]["is_code_query"] is True
        )

    def test_intent_hint_semantic_passes_is_code_query_false(self, tools):
        tool_list, engine = tools
        search_code = self._get_search_code(tool_list)

        search_code.invoke({"query": "how does sensor work", "intent_hint": "semantic"})

        call_kwargs = engine.search.call_args
        assert call_kwargs.kwargs.get("is_code_query") is False or (
            "is_code_query" in call_kwargs[1] and call_kwargs[1]["is_code_query"] is False
        )

    def test_intent_hint_none_passes_is_code_query_none(self, tools):
        tool_list, engine = tools
        search_code = self._get_search_code(tool_list)

        search_code.invoke({"query": "test", "intent_hint": None})

        call_kwargs = engine.search.call_args
        assert call_kwargs.kwargs.get("is_code_query") is None or (
            "is_code_query" in call_kwargs[1] and call_kwargs[1]["is_code_query"] is None
        )

    def test_default_no_intent_hint_passes_is_code_query_none(self, tools):
        tool_list, engine = tools
        search_code = self._get_search_code(tool_list)

        search_code.invoke({"query": "test"})

        call_kwargs = engine.search.call_args
        assert call_kwargs.kwargs.get("is_code_query") is None or (
            "is_code_query" in call_kwargs[1] and call_kwargs[1]["is_code_query"] is None
        )


# ---------------------------------------------------------------------------
# TestSearchMulti
# ---------------------------------------------------------------------------

class TestSearchMulti:
    """Tests for search_multi dedup and sorting behavior."""

    def _make_search_multi_tool(self, search_results: list[list[SearchResult]]):
        engine = MagicMock()
        engine.search.side_effect = search_results
        index_mgr = MagicMock()
        tools = create_tools(engine, index_mgr)
        for t in tools:
            if t.name == "search_multi":
                return t
        raise AssertionError("search_multi tool not found")

    def test_deduplicates_same_chunk_keeps_highest_score(self):
        """Two queries return the same chunk with different scores; only highest kept."""
        chunk = _make_chunk(chunk_id="c1", repo_name="sensor", file_path="a.lua", start_line=1)
        r_low = _make_result(chunk=chunk, score=0.3)
        r_high = _make_result(chunk=chunk, score=0.9)

        tool = self._make_search_multi_tool([[r_low], [r_high]])
        result = tool.invoke({"queries": ["q1", "q2"], "top_k": 5})

        # Should contain only one entry for that chunk
        assert result.count("[1]") == 1
        assert "0.9000" in result
        assert "0.3000" not in result

    def test_returns_top_k_sorted_by_score(self):
        """Results should be sorted by score descending and limited to top_k."""
        r1 = _make_result(chunk=_make_chunk(chunk_id="c1", start_line=1), score=0.5)
        r2 = _make_result(chunk=_make_chunk(chunk_id="c2", start_line=2), score=0.9)
        r3 = _make_result(chunk=_make_chunk(chunk_id="c3", start_line=3), score=0.7)

        tool = self._make_search_multi_tool([[r1, r2, r3]])
        result = tool.invoke({"queries": ["q1"], "top_k": 2})

        # Should have exactly 2 results, highest score first
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" not in result
        # Score 0.9 should appear before 0.7
        pos_09 = result.find("0.9000")
        pos_07 = result.find("0.7000")
        assert pos_09 < pos_07

    def test_limits_queries_to_five(self):
        """More than 5 queries should be silently truncated."""
        engine = MagicMock()
        engine.search.return_value = []
        index_mgr = MagicMock()
        tools = create_tools(engine, index_mgr)
        search_multi = None
        for t in tools:
            if t.name == "search_multi":
                search_multi = t
                break

        search_multi.invoke({"queries": ["a", "b", "c", "d", "e", "f", "g"]})

        # engine.search should be called at most 5 times (one per truncated query)
        assert engine.search.call_count == 5


# ---------------------------------------------------------------------------
# TestToolCount
# ---------------------------------------------------------------------------

class TestToolCount:
    """Verify create_tools returns the expected number of tools."""

    def test_create_tools_returns_seven_tools(self):
        engine = MagicMock()
        index_mgr = MagicMock()
        tools = create_tools(engine, index_mgr)

        assert len(tools) == 7

    def test_tool_names(self):
        engine = MagicMock()
        index_mgr = MagicMock()
        tools = create_tools(engine, index_mgr)
        names = {t.name for t in tools}

        expected = {
            "search_code",
            "search_docs",
            "search_multi",
            "find_definitions",
            "find_references",
            "list_components",
            "get_component_deps",
        }
        assert names == expected
