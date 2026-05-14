"""FastMCP server for openUBMC Code RAG — exposes search and analysis tools."""

from __future__ import annotations

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.embedder import Embedder
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.models.component_info import ComponentInfo
from ubmc_rag.search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)

# Global state (initialized on server start)
_config: Optional[AppConfig] = None
_index_mgr: Optional[IndexManager] = None
_engine: Optional[HybridSearchEngine] = None


def _ensure_initialized() -> tuple[AppConfig, IndexManager, HybridSearchEngine]:
    if _engine is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return _config, _index_mgr, _engine


def init_server(config: AppConfig) -> None:
    """Initialize the server state by loading the index."""
    global _config, _index_mgr, _engine

    _config = config
    _index_mgr = IndexManager(config)
    _index_mgr.load_index()

    chunks = _index_mgr.get_all_chunks()
    _engine = HybridSearchEngine(
        embedder=_index_mgr.embedder,
        vector_store=_index_mgr.vector_store,
        bm25=_index_mgr.bm25,
        config=config,
    )
    _engine.set_chunk_index(chunks)
    logger.info("MCP server initialized with %d chunks", len(chunks))


def create_server(config: AppConfig) -> FastMCP:
    """Create and configure the FastMCP server."""
    init_server(config)

    mcp = FastMCP(
        name="openubmc-code-rag",
        version="0.1.0",
        instructions=(
            "Code RAG server for the openUBMC project. "
            "Provides semantic and keyword search over openUBMC's Lua, C/C++, Python, JSON, "
            "and documentation codebase. Use search_code for general queries, "
            "find_definitions to locate symbol definitions, and get_component_deps "
            "to understand component dependencies."
        ),
    )

    @mcp.tool()
    def search_code(
        query: str,
        language: Optional[str] = None,
        repo: Optional[str] = None,
        chunk_type: Optional[str] = None,
        top_k: int = 10,
    ) -> str:
        """Search openUBMC codebase using hybrid semantic + keyword search.

        Args:
            query: Natural language or code snippet search query
            language: Filter by language (lua, c, cpp, python, json, markdown)
            repo: Filter by repository/component name (e.g., "sensor", "libipmi")
            chunk_type: Filter by chunk type (function, class, method, mds_model, mds_ipmi_cmd, csr_object, section)
            top_k: Number of results to return (default 10, max 50)
        """
        _, _, engine = _ensure_initialized()
        results = engine.search(
            query=query,
            top_k=min(top_k, 50),
            language=language,
            repo=repo,
            chunk_type=chunk_type,
        )
        return json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)

    @mcp.tool()
    def find_definitions(
        symbol_name: str,
        language: Optional[str] = None,
    ) -> str:
        """Find all definitions of a symbol (function, class, variable, interface) across openUBMC.

        Args:
            symbol_name: The symbol name to search for (e.g., "ThresholdSensor", "init", "get_sensor_data")
            language: Optional language filter (lua, c, cpp, python, json)
        """
        _, index_mgr, engine = _ensure_initialized()
        results = engine.search(
            query=symbol_name,
            top_k=20,
            language=language,
            is_code_query=True,
        )

        # Filter to only definition-like results (functions, classes, mds_models)
        definitions = []
        for r in results:
            sym_names = [s.name for s in r.chunk.symbols]
            if symbol_name in sym_names:
                definitions.append(r.to_dict())

        if not definitions:
            # Fallback: return top results mentioning the symbol
            definitions = [r.to_dict() for r in results[:5]]

        return json.dumps(definitions, indent=2, ensure_ascii=False)

    @mcp.tool()
    def find_references(symbol_name: str) -> str:
        """Find all references to a named symbol across the openUBMC codebase.

        Args:
            symbol_name: The symbol name to search references for
        """
        _, _, engine = _ensure_initialized()
        results = engine.search(
            query=symbol_name,
            top_k=30,
            is_code_query=True,
        )
        return json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)

    @mcp.tool()
    def list_components() -> str:
        """List all openUBMC components discovered in the indexed codebase.

        Returns component name, languages, file count, function count, and class count.
        """
        _, index_mgr, _ = _ensure_initialized()
        from collections import defaultdict

        chunks = index_mgr.get_all_chunks()
        comp_data: dict[str, dict] = defaultdict(lambda: {
            "files": set(), "functions": 0, "classes": 0, "languages": set(),
        })

        for chunk in chunks:
            comp = chunk.component_name or chunk.repo_name
            data = comp_data[comp]
            data["files"].add(chunk.file_path)
            data["languages"].add(chunk.language)
            for sym in chunk.symbols:
                if sym.kind == "function":
                    data["functions"] += 1
                elif sym.kind == "class":
                    data["classes"] += 1

        components = []
        for name, data in sorted(comp_data.items()):
            components.append({
                "name": name,
                "languages": sorted(data["languages"]),
                "file_count": len(data["files"]),
                "function_count": data["functions"],
                "class_count": data["classes"],
            })

        return json.dumps(components, indent=2, ensure_ascii=False)

    @mcp.tool()
    def get_component_deps(component_name: str) -> str:
        """Get dependencies and interfaces for a specific openUBMC component.

        Parses service.json for build dependencies and required interfaces.

        Args:
            component_name: The component name (e.g., "sensor", "devmon", "vpd")
        """
        _, index_mgr, _ = _ensure_initialized()
        chunks = index_mgr.get_all_chunks()

        deps = {
            "component": component_name,
            "dependencies": [],
            "required_interfaces": [],
            "provided_interfaces": [],
            "mds_classes": [],
            "ipmi_commands": [],
        }

        for chunk in chunks:
            if chunk.repo_name != component_name:
                continue

            # From service.json metadata
            if chunk.chunk_type == "mds_service":
                deps["dependencies"] = chunk.metadata.get("dependencies", [])
                deps["required_interfaces"] = chunk.metadata.get("required_interfaces", [])

            # From model.json
            if chunk.chunk_type == "mds_model":
                mds_class = chunk.metadata.get("mds_class", "")
                if mds_class:
                    deps["mds_classes"].append(mds_class)

            # From ipmi.json
            if chunk.chunk_type == "mds_ipmi_cmd":
                for sym in chunk.symbols:
                    if sym.kind == "ipmi_command":
                        deps["ipmi_commands"].append(sym.name)

        return json.dumps(deps, indent=2, ensure_ascii=False)

    # Resources
    @mcp.resource("ubmc://component/{component_name}/info")
    def component_info(component_name: str) -> str:
        """Get component metadata."""
        return get_component_deps(component_name)

    @mcp.resource("ubmc://mds/{component_name}/models")
    def mds_models(component_name: str) -> str:
        """Get MDS model definitions for a component."""
        _, index_mgr, _ = _ensure_initialized()
        models = []
        for chunk in index_mgr.get_all_chunks():
            if chunk.repo_name == component_name and chunk.chunk_type == "mds_model":
                models.append({
                    "class_name": chunk.metadata.get("mds_class", ""),
                    "content": chunk.content,
                })
        return json.dumps(models, indent=2, ensure_ascii=False)

    @mcp.resource("ubmc://mds/{component_name}/ipmi")
    def mds_ipmi(component_name: str) -> str:
        """Get IPMI command definitions for a component."""
        _, index_mgr, _ = _ensure_initialized()
        commands = []
        for chunk in index_mgr.get_all_chunks():
            if chunk.repo_name == component_name and chunk.chunk_type == "mds_ipmi_cmd":
                commands.append({
                    "command": chunk.metadata.get("cmd", ""),
                    "netfn": chunk.metadata.get("netfn", ""),
                    "content": chunk.content,
                })
        return json.dumps(commands, indent=2, ensure_ascii=False)

    return mcp
