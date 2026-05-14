"""Tests for JSON Schema parser."""

from pathlib import Path

from ubmc_rag.ingestion.parsers.json_parser import JsonParser


def test_parse_service_json(mds_service_json):
    parser = JsonParser()
    chunks = parser.parse(Path("service.json"), mds_service_json, "sensor")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "mds_service"
    assert "mdb_interface" in chunk.metadata.get("dependencies", [])
    assert "bmc.kepler.Sensors" in chunk.metadata.get("required_interfaces", [])


def test_parse_model_json(mds_model_json):
    parser = JsonParser()
    chunks = parser.parse(Path("model.json"), mds_model_json, "sensor")

    assert len(chunks) == 2  # ThresholdSensor and DiscreteSensor

    names = [c.metadata.get("mds_class") for c in chunks]
    assert "ThresholdSensor" in names
    assert "DiscreteSensor" in names

    # Check symbol extraction
    all_symbols = []
    for c in chunks:
        all_symbols.extend(c.symbols)
    symbol_names = [s.name for s in all_symbols]
    assert "ThresholdSensor" in symbol_names
    assert "Reading" in symbol_names


def test_parse_ipmi_json(mds_ipmi_json):
    parser = JsonParser()
    chunks = parser.parse(Path("ipmi.json"), mds_ipmi_json, "sensor")

    assert len(chunks) == 2  # GetSensorReading, SetSensorThresholds

    symbol_names = [s.name for c in chunks for s in c.symbols]
    assert "GetSensorReading" in symbol_names
    assert "SetSensorThresholds" in symbol_names


def test_parse_invalid_json():
    parser = JsonParser()
    chunks = parser.parse(Path("bad.json"), "not valid json {", "test")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "config_block"
