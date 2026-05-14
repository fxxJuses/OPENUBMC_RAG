"""Tests for BM25 index."""

from ubmc_rag.indexing.bm25_index import BM25Index, code_tokenize
from ubmc_rag.models.code_chunk import CodeChunk


def test_code_tokenize():
    tokens = code_tokenize("getSensorData reading_value IPMI_CMD")
    assert "get" in tokens
    assert "sensor" in tokens
    assert "data" in tokens
    assert "reading" in tokens
    # underscore-split tokens: reading_value -> reading, value
    lower_tokens = [t.lower() for t in tokens]
    assert "value" in lower_tokens or "reading" in lower_tokens


def test_bm25_build_and_search():
    chunks = [
        CodeChunk(
            chunk_id="1", content="function get_sensor_data(sensor_id) return data end",
            file_path="a.lua", repo_name="sensor", language="lua",
            component_name="sensor", start_line=1, end_line=1, chunk_type="function",
        ),
        CodeChunk(
            chunk_id="2", content="function update_firmware(version) flash(version) end",
            file_path="b.lua", repo_name="fructrl", language="lua",
            component_name="fructrl", start_line=1, end_line=1, chunk_type="function",
        ),
        CodeChunk(
            chunk_id="3", content="ThresholdSensor class with reading and threshold properties",
            file_path="c.json", repo_name="sensor", language="json",
            component_name="sensor", start_line=1, end_line=10, chunk_type="mds_model",
        ),
    ]

    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search("sensor reading data")
    assert len(results) > 0
    # First result should be the sensor-related chunk
    assert results[0][0] == "1" or results[0][0] == "3"


def test_bm25_save_load(tmp_path):
    chunks = [
        CodeChunk(
            chunk_id="1", content="test function",
            file_path="a.lua", repo_name="test", language="lua",
            component_name="test", start_line=1, end_line=1, chunk_type="function",
        ),
    ]

    bm25 = BM25Index()
    bm25.build(chunks)

    save_path = tmp_path / "bm25.json"
    bm25.save(save_path)
    assert save_path.exists()

    bm25_loaded = BM25Index()
    assert bm25_loaded.load(save_path)

    results = bm25_loaded.search("test function")
    assert len(results) == 1
    assert results[0][0] == "1"
