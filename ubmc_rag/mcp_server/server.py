"""FastMCP 服务器，将 openUBMC 代码检索能力暴露为 MCP 工具。

通过 MCP (Model Context Protocol) 协议提供以下工具：
- search_code: 混合语义+关键词代码搜索
- find_definitions: 符号定义查找
- find_references: 符号引用查找
- list_components: 组件列表
- get_component_deps: 组件依赖分析

同时提供 MCP Resource 接口用于获取组件的 MDS 模型和 IPMI 命令定义。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)

# 全局状态（服务器启动时初始化）
_config: Optional[AppConfig] = None
_index_mgr: Optional[IndexManager] = None
_engine: Optional[HybridSearchEngine] = None


def _ensure_initialized() -> tuple[AppConfig, IndexManager, HybridSearchEngine]:
    """确保服务器已初始化，返回全局状态元组。"""
    if _engine is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return _config, _index_mgr, _engine


def init_server(config: AppConfig) -> None:
    """初始化服务器状态，加载索引并构建搜索引擎。

    Args:
        config: 应用配置
    """
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
    """创建并配置 FastMCP 服务器实例。

    注册所有工具函数和资源端点，然后返回配置好的 FastMCP 实例。

    Args:
        config: 应用配置

    Returns:
        配置完成的 FastMCP 服务器
    """
    init_server(config)

    mcp = FastMCP(
        name="openubmc-code-rag",
        version="0.1.0",
        instructions=(
            "Code RAG server for the openUBMC project. "
            "Provides semantic and keyword search over openUBMC's Lua, C/C++, "
            "Python, JSON, and documentation codebase."
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
        """混合语义+关键词代码搜索。

        Args:
            query: 自然语言或代码片段查询
            language: 语言过滤（lua, c, cpp, python, json, markdown）
            repo: 仓库/组件名过滤（如 "sensor", "libipmi"）
            chunk_type: 分块类型过滤（function, class, mds_model 等）
            top_k: 返回结果数量（默认 10，最大 50）
        """
        _, _, engine = _ensure_initialized()
        results = engine.search(
            query=query,
            top_k=min(top_k, 50),
            language=language,
            repo=repo,
            chunk_type=chunk_type,
        )
        return json.dumps(
            [r.to_dict() for r in results], indent=2, ensure_ascii=False
        )

    @mcp.tool()
    def find_definitions(
        symbol_name: str,
        language: Optional[str] = None,
    ) -> str:
        """查找符号定义位置（函数、类、变量、接口）。

        Args:
            symbol_name: 符号名称（如 "ThresholdSensor", "init"）
            language: 可选的语言过滤
        """
        _, index_mgr, engine = _ensure_initialized()
        results = engine.search(
            query=symbol_name,
            top_k=20,
            language=language,
            is_code_query=True,
        )

        # 优先返回符号名精确匹配的定义
        definitions = []
        for r in results:
            sym_names = [s.name for s in r.chunk.symbols]
            if symbol_name in sym_names:
                definitions.append(r.to_dict())

        if not definitions:
            definitions = [r.to_dict() for r in results[:5]]

        return json.dumps(definitions, indent=2, ensure_ascii=False)

    @mcp.tool()
    def find_references(symbol_name: str) -> str:
        """查找符号的所有引用位置。

        Args:
            symbol_name: 待查找引用的符号名称
        """
        _, _, engine = _ensure_initialized()
        results = engine.search(
            query=symbol_name,
            top_k=30,
            is_code_query=True,
        )
        return json.dumps(
            [r.to_dict() for r in results], indent=2, ensure_ascii=False
        )

    @mcp.tool()
    def list_components() -> str:
        """列出所有已索引的 openUBMC 组件及其统计信息。"""
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
        """获取组件的依赖关系、接口、MDS 类和 IPMI 命令。

        解析 service.json 获取构建依赖和所需接口。

        Args:
            component_name: 组件名称（如 "sensor", "devmon"）
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
            if chunk.chunk_type == "mds_service":
                deps["dependencies"] = chunk.metadata.get("dependencies", [])
                deps["required_interfaces"] = chunk.metadata.get(
                    "required_interfaces", []
                )
            if chunk.chunk_type == "mds_model":
                mds_class = chunk.metadata.get("mds_class", "")
                if mds_class:
                    deps["mds_classes"].append(mds_class)
            if chunk.chunk_type == "mds_ipmi_cmd":
                for sym in chunk.symbols:
                    if sym.kind == "ipmi_command":
                        deps["ipmi_commands"].append(sym.name)

        return json.dumps(deps, indent=2, ensure_ascii=False)

    # ---- MCP Resource 端点 ----

    @mcp.resource("ubmc://component/{component_name}/info")
    def component_info(component_name: str) -> str:
        """获取组件的完整元数据。"""
        return get_component_deps(component_name)

    @mcp.resource("ubmc://mds/{component_name}/models")
    def mds_models(component_name: str) -> str:
        """获取组件的 MDS 模型定义。"""
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
        """获取组件的 IPMI 命令定义。"""
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
