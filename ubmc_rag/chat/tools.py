"""LangChain Tools for openUBMC RAG Agent — wrapping HybridSearchEngine and IndexManager."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Optional

from langchain_core.tools import tool

from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


def _format_results(results) -> str:
    """Format search results into a human-readable string with source markers."""
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        d = r.to_dict()
        header = (
            f"[{i}] {d['repo']}/{d['file_path']}:{d['start_line']}-{d['end_line']} "
            f"(score={d['score']:.4f})"
        )
        parts.append(f"{header}\n{d['content']}")
    return "\n\n---\n\n".join(parts)


def create_tools(engine: HybridSearchEngine, index_mgr: IndexManager) -> list:
    """Create LangChain tools for the RAG agent."""

    @tool
    def search_code(
        query: str,
        language: Optional[str] = None,
        repo: Optional[str] = None,
        chunk_type: Optional[str] = None,
        top_k: int = 8,
    ) -> str:
        """在 openUBMC 代码库中进行混合语义+关键词搜索。适用于：理解代码逻辑、查找代码片段、探索架构细节。"""
        results = engine.search(
            query=query,
            top_k=min(top_k, 50),
            language=language,
            repo=repo,
            chunk_type=chunk_type,
        )
        return _format_results(results)

    @tool
    def find_definitions(
        symbol_name: str,
        language: Optional[str] = None,
    ) -> str:
        """查找函数、类、变量的定义位置。适用于：需要知道某个符号在哪里定义的。"""
        results = engine.search(
            query=symbol_name,
            top_k=20,
            language=language,
            is_code_query=True,
        )
        definitions = []
        for r in results:
            sym_names = [s.name for s in r.chunk.symbols]
            if symbol_name in sym_names:
                definitions.append(r)

        if not definitions:
            definitions = results[:5]

        return _format_results(definitions)

    @tool
    def find_references(symbol_name: str) -> str:
        """查找符号的所有引用位置。适用于：需要知道某个函数/类在哪里被调用或使用。"""
        results = engine.search(
            query=symbol_name,
            top_k=30,
            is_code_query=True,
        )
        return _format_results(results)

    @tool
    def list_components() -> str:
        """列出所有已索引的 openUBMC 微组件及其统计信息。"""
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

        lines = []
        for name, data in sorted(comp_data.items()):
            lines.append(
                f"- {name}: {len(data['files'])} files, "
                f"{data['functions']} functions, {data['classes']} classes "
                f"({', '.join(sorted(data['languages']))})"
            )
        return "\n".join(lines)

    @tool
    def get_component_deps(component_name: str) -> str:
        """获取指定组件的依赖关系、接口、MDS 类和 IPMI 命令。适用于：理解组件间的依赖和交互关系。"""
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
            if chunk.chunk_type == "mds_service":
                deps["dependencies"] = chunk.metadata.get("dependencies", [])
                deps["required_interfaces"] = chunk.metadata.get("required_interfaces", [])
            if chunk.chunk_type == "mds_model":
                mds_class = chunk.metadata.get("mds_class", "")
                if mds_class:
                    deps["mds_classes"].append(mds_class)
            if chunk.chunk_type == "mds_ipmi_cmd":
                for sym in chunk.symbols:
                    if sym.kind == "ipmi_command":
                        deps["ipmi_commands"].append(sym.name)

        return json.dumps(deps, indent=2, ensure_ascii=False)

    return [search_code, find_definitions, find_references, list_components, get_component_deps]
