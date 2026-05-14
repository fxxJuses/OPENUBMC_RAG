"""Tests for C/C++ AST parser."""

from pathlib import Path

from ubmc_rag.ingestion.parsers.c_cpp_parser import CCppParser


def test_parse_c_functions(c_sample):
    parser = CCppParser()
    chunks = parser.parse(Path("ipmi_sensor.c"), c_sample, "libipmi")

    assert len(chunks) > 0

    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) >= 2  # ipmi_get_sensor_reading, ipmi_set_sensor_threshold

    symbol_names = [s.name for c in func_chunks for s in c.symbols]
    assert "ipmi_get_sensor_reading" in symbol_names


def test_parse_c_struct(c_sample):
    parser = CCppParser()
    chunks = parser.parse(Path("ipmi_sensor.c"), c_sample, "libipmi")

    # The struct is inside a typedef, so chunk_type is "typedef"
    typedef_chunks = [c for c in chunks if c.chunk_type in ("class", "typedef")]
    assert len(typedef_chunks) >= 1

    symbol_names = [s.name for c in typedef_chunks for s in c.symbols]
    assert "ipmi_sensor" in symbol_names or "ipmi_sensor_t" in symbol_names


def test_cpp_file_detection():
    parser = CCppParser()
    assert parser._get_language_tag(Path("test.cpp")) == "cpp"
    assert parser._get_language_tag(Path("test.c")) == "c"
    assert parser._get_language_tag(Path("test.hpp")) == "cpp"
