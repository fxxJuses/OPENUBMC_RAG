"""Tests for Lua AST parser."""

from pathlib import Path

from ubmc_rag.ingestion.parsers.lua_parser import LuaParser


def test_parse_lua_functions(lua_sample):
    parser = LuaParser()
    chunks = parser.parse(Path("sensor_app.lua"), lua_sample, "sensor")

    assert len(chunks) > 0

    # openUBMC uses method syntax (obj:method), so chunks are "method" type
    func_chunks = [c for c in chunks if c.chunk_type in ("function", "method")]
    assert len(func_chunks) >= 4  # init, pre_init, get_sensor_data, update_sensor, etc.

    # Check symbol extraction
    all_symbols = []
    for c in chunks:
        all_symbols.extend(c.symbols)

    symbol_names = [s.name for s in all_symbols]
    assert "init" in symbol_names or "SensorApp:init" in symbol_names
    assert "get_sensor_data" in symbol_names or "SensorApp:get_sensor_data" in symbol_names


def test_parse_lua_small_file():
    small_lua = "local x = 1\nreturn x\n"
    parser = LuaParser()
    chunks = parser.parse(Path("small.lua"), small_lua, "test")

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"


def test_lua_chunk_metadata(lua_sample):
    parser = LuaParser()
    chunks = parser.parse(Path("sensor_app.lua"), lua_sample, "sensor")

    for chunk in chunks:
        assert chunk.language == "lua"
        assert chunk.repo_name == "sensor"
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line
        assert chunk.content.strip()
