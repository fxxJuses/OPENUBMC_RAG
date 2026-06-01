"""Tests for BM25 code tokenizer (迭代6-B simplified).

Verifies:
- camelCase splitting (no composite preservation)
- snake_case splitting (no composite preservation)
- Domain dictionary: IPMI, FRU, SEL etc. maintained intact
- Backward compatibility with existing tests
"""

from ubmc_rag.indexing.bm25_index import (
    BM25Index, code_tokenize, _DOMAIN_DICTIONARY,
)
from ubmc_rag.models.code_chunk import CodeChunk


def test_code_tokenize_camelcase():
    """Test camelCase splitting produces sub-tokens, no composite."""
    tokens = code_tokenize("getSensorData")
    token_set = set(tokens)
    # Sub-tokens from splitting
    assert "get" in token_set, f"Expected 'get' in {tokens}"
    assert "sensor" in token_set, f"Expected 'sensor' in {tokens}"
    assert "data" in token_set, f"Expected 'data' in {tokens}"
    # Composite token should NOT be present (去复合 token)
    assert "getsensordata" not in token_set, (
        f"Composite 'getsensordata' should NOT be in {tokens}"
    )


def test_code_tokenize_snakecase():
    """Test snake_case splitting produces sub-tokens, no composite."""
    tokens = code_tokenize("reading_value")
    token_set = set(tokens)
    assert "reading" in token_set, f"Expected 'reading' in {tokens}"
    assert "value" in token_set, f"Expected 'value' in {tokens}"
    # Composite should NOT be preserved
    assert "reading_value" not in token_set, (
        f"Composite 'reading_value' should NOT be in {tokens}"
    )


def test_code_tokenize_domain_term_ipmi():
    """Test domain dictionary terms are recognized, no composite."""
    tokens = code_tokenize("IPMI_CMD GetSensorReading FRU")
    token_set = set(tokens)
    # IPMI sub-tokens from splitting
    assert "ipmi" in token_set, f"Expected 'ipmi' in {tokens}"
    assert "cmd" in token_set, f"Expected 'cmd' in {tokens}"
    # Domain multi-word composite should NOT be preserved (去复合 token)
    assert "ipmi_cmd" not in token_set, (
        f"Composite 'ipmi_cmd' should NOT be in {tokens}"
    )
    # camelCase splitting (sub-tokens only, no composite)
    assert "get" in token_set, f"Expected 'get' in {tokens}"
    assert "reading" in token_set, f"Expected 'reading' in {tokens}"
    assert "getsensorreading" not in token_set, (
        f"Composite 'getsensorreading' should NOT be in {tokens}"
    )
    # Domain term FRU
    assert "fru" in token_set, f"Expected 'fru' in {tokens}"


def test_code_tokenize_domain_dict_terms():
    """Test that all domain dictionary terms are recognized."""
    domain_tokens = {
        "ipmi", "sel", "sdr", "pef", "fru", "vpd", "i2c",
        "sensor", "firmware", "bios", "bmc", "gpio", "pcie",
    }
    # Each term should be in the dictionary
    for term in domain_tokens:
        assert term in _DOMAIN_DICTIONARY, f"'{term}' should be in domain dictionary"


def test_code_tokenize_filter_short():
    """Test that tokens of length <= 1 are filtered out."""
    tokens = code_tokenize("a b c d ab cd")
    # Single chars should be filtered
    assert "a" not in tokens
    assert "b" not in tokens
    assert "c" not in tokens
    assert "d" not in tokens
    # Two-char tokens should remain
    assert "ab" in tokens
    assert "cd" in tokens


def test_code_tokenize_backward_compat():
    """Test backward compatibility: sub-token patterns still work."""
    tokens = code_tokenize("getSensorData reading_value IPMI_CMD")
    lower_tokens = [t.lower() for t in tokens]
    # Original assertion: sub-tokens exist
    assert "get" in lower_tokens
    assert "sensor" in lower_tokens
    assert "data" in lower_tokens
    assert "reading" in lower_tokens
    assert "ipmi" in lower_tokens
    assert "cmd" in lower_tokens
    # 迭代6-B: composites should NOT be present (去复合 token)
    assert "getsensordata" not in lower_tokens
    assert "reading_value" not in lower_tokens
    assert "ipmi_cmd" not in lower_tokens


def test_code_tokenize_no_composites():
    """Original no-composite test still passes with simplified tokenizer."""
    tokens = code_tokenize("getSensorData")
    # Should have sub-tokens
    assert "get" in tokens
    assert "sensor" in tokens
    assert "data" in tokens
    # Should NOT have composite
    assert "getsensordata" not in tokens


def test_bm25_build_and_search():
    """Original test: BM25 build and search still works with simplified tokenizer."""
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
    """Original test: BM25 save/load with simplified tokenizer."""
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


def test_bm25_with_simplified_tokens():
    """Test BM25 search with simplified tokenizer (sub-token matching)."""
    chunks = [
        CodeChunk(
            chunk_id="1", content="function getSensorReading() reads IPMI sensor data",
            file_path="ipmi.lua", repo_name="sensor", language="lua",
            component_name="sensor", start_line=1, end_line=1, chunk_type="function",
        ),
        CodeChunk(
            chunk_id="2", content="function updateFirmware() handles firmware updates",
            file_path="firmware.lua", repo_name="fructrl", language="lua",
            component_name="fructrl", start_line=1, end_line=1, chunk_type="function",
        ),
    ]

    bm25 = BM25Index()
    bm25.build(chunks)

    # Search for sub-tokens of getSensorReading - should match chunk 1
    results = bm25.search("get sensor reading")
    assert len(results) > 0
    # Both chunks contain matched tokens; check at least one result
    matched_ids = [r[0] for r in results]
    assert "1" in matched_ids, f"Chunk 1 should be in results: {matched_ids}"

    # Search for "sensor reading" - sub-tokens should match chunk 1
    results2 = bm25.search("sensor reading")
    assert len(results2) > 0
    assert results2[0][0] == "1"

    # Search for "IPMI" - domain term should match chunk 1
    results3 = bm25.search("IPMI")
    assert len(results3) > 0
    assert results3[0][0] == "1"
