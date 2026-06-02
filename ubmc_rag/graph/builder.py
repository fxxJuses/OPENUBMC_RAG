"""两阶段 AST 图构建器 —— 从 CodeChunk 列表构建代码知识图。

Pass 1: 从所有 chunks 中提取节点（component, file, entity, interface）
Pass 2: 解析关系边（DEPENDS_ON, IMPORTS, CALLS, REQUIRES_IFACE 等）
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ubmc_rag.graph.schema import (
    DEFAULT_EDGE_WEIGHTS,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    make_node_id,
)
from ubmc_rag.graph.store import KnowledgeGraph
from ubmc_rag.models.code_chunk import CodeChunk

logger = logging.getLogger(__name__)

# Lua require() 模式: require("xxx") 或 require('xxx')
_LUA_REQUIRE_RE = re.compile(r'''require\s*\(\s*["']([^"']+)["']\s*\)''')

# C/C++ #include 模式: #include "xxx.h"
_C_INCLUDE_RE = re.compile(r'''#include\s*"([^"]+)"''')

# Lua 函数调用: obj:method() 或 func()
_LUA_CALL_RE = re.compile(r'(?:\w+)\s*:\s*(\w+)\s*\(|(\w+)\s*\(')

# C 函数调用: func_name(
_C_CALL_RE = re.compile(r'(\w+)\s*\(')

# 需要排除的 C 关键字，不算函数调用
_C_KEYWORDS = frozenset({
    "if", "else", "while", "for", "switch", "case", "return", "sizeof",
    "typeof", "typedef", "struct", "enum", "union", "class", "do",
    "break", "continue", "goto", "default", "NULL",
})


class GraphBuilder:
    """从 CodeChunk 列表构建代码知识图。

    使用两阶段构建：
    1. Pass 1: 遍历所有 chunks，注册 component/file/entity/interface 节点
    2. Pass 2: 在完整节点表基础上解析 require()/#include/调用目标等边
    """

    def build(self, chunks: list[CodeChunk]) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        node_registry: dict[str, GraphNode] = {}

        # Pass 1: 发现所有节点
        self._pass1_discover_nodes(chunks, kg, node_registry)

        # Pass 2: 解析所有边
        self._pass2_resolve_edges(chunks, kg, node_registry)

        logger.info(
            "Graph built: %d nodes (%s), %d edges (%s)",
            kg.node_count(), kg.node_type_counts(),
            kg.edge_count(), kg.edge_type_counts(),
        )
        return kg

    def _pass1_discover_nodes(
        self,
        chunks: list[CodeChunk],
        kg: KnowledgeGraph,
        registry: dict[str, GraphNode],
    ) -> None:
        seen_components: set[str] = set()
        seen_files: set[str] = set()

        for chunk in chunks:
            # Component 节点
            comp_id = make_node_id(NodeType.COMPONENT, chunk.repo_name)
            if chunk.repo_name not in seen_components:
                self._add_node(registry, kg, GraphNode(
                    node_id=comp_id,
                    node_type=NodeType.COMPONENT,
                    properties={"name": chunk.repo_name},
                ))
                seen_components.add(chunk.repo_name)

            # File 节点
            file_id = make_node_id(NodeType.FILE, chunk.repo_name, chunk.file_path)
            file_key = f"{chunk.repo_name}:{chunk.file_path}"
            if file_key not in seen_files:
                self._add_node(registry, kg, GraphNode(
                    node_id=file_id,
                    node_type=NodeType.FILE,
                    properties={
                        "path": chunk.file_path,
                        "language": chunk.language,
                        "repo_name": chunk.repo_name,
                    },
                ))
                seen_files.add(file_key)

                # BELONGS_TO: file -> component
                kg.add_edge(GraphEdge(
                    source_id=file_id, target_id=comp_id,
                    edge_type=EdgeType.BELONGS_TO,
                    weight=DEFAULT_EDGE_WEIGHTS[EdgeType.BELONGS_TO],
                ))

            # Entity 节点（从 symbols 提取）
            for sym in chunk.symbols:
                if sym.kind in ("function", "class", "method", "ipmi_command"):
                    entity_id = make_node_id(
                        NodeType.ENTITY, chunk.repo_name, chunk.file_path, sym.name,
                    )
                    self._add_node(registry, kg, GraphNode(
                        node_id=entity_id,
                        node_type=NodeType.ENTITY,
                        properties={
                            "name": sym.name,
                            "kind": sym.kind,
                            "language": chunk.language,
                            "file_path": chunk.file_path,
                            "repo_name": chunk.repo_name,
                            "chunk_id": chunk.chunk_id,
                            "signature": sym.signature or "",
                        },
                    ))

                    # CONTAINS: file -> entity
                    kg.add_edge(GraphEdge(
                        source_id=file_id, target_id=entity_id,
                        edge_type=EdgeType.CONTAINS,
                        weight=DEFAULT_EDGE_WEIGHTS[EdgeType.CONTAINS],
                    ))

                    # DEFINES: component -> entity
                    kg.add_edge(GraphEdge(
                        source_id=comp_id, target_id=entity_id,
                        edge_type=EdgeType.DEFINES,
                        weight=DEFAULT_EDGE_WEIGHTS[EdgeType.DEFINES],
                    ))

            # Interface 节点（从 service.json 提取）
            if chunk.chunk_type == "mds_service":
                self._extract_service_interfaces(chunk, kg, registry, comp_id)

    def _pass2_resolve_edges(
        self,
        chunks: list[CodeChunk],
        kg: KnowledgeGraph,
        registry: dict[str, GraphNode],
    ) -> None:
        # 构建 entity name -> node_id 索引（用于调用匹配）
        entity_index = self._build_entity_index(registry)
        # 构建 module path -> component 映射（用于 require() 解析）
        module_index = self._build_module_index(registry)

        for chunk in chunks:
            comp_id = make_node_id(NodeType.COMPONENT, chunk.repo_name)

            # service.json 依赖
            if chunk.chunk_type == "mds_service":
                self._resolve_service_deps(chunk, kg, comp_id)

            # Lua require() 依赖
            if chunk.language == "lua":
                self._resolve_lua_imports(chunk, kg, comp_id, module_index)
                self._resolve_lua_calls(chunk, kg, registry, entity_index)

            # C/C++ #include 依赖
            if chunk.language in ("c", "cpp"):
                self._resolve_c_imports(chunk, kg, comp_id, module_index)
                self._resolve_c_calls(chunk, kg, registry, entity_index)

    def _add_node(self, registry: dict[str, GraphNode], kg: KnowledgeGraph, node: GraphNode) -> None:
        if node.node_id not in registry:
            registry[node.node_id] = node
            kg.add_node(node)

    def _build_entity_index(self, registry: dict[str, GraphNode]) -> dict[str, list[str]]:
        """构建 symbol name -> [entity_node_ids] 的索引。"""
        index: dict[str, list[str]] = {}
        for nid, node in registry.items():
            if node.node_type == NodeType.ENTITY:
                name = node.properties.get("name", "")
                if name:
                    index.setdefault(name, []).append(nid)
        return index

    def _build_module_index(self, registry: dict[str, GraphNode]) -> dict[str, str]:
        """构建 file path -> component_id 的映射。"""
        index: dict[str, str] = {}
        for nid, node in registry.items():
            if node.node_type == NodeType.FILE:
                path = node.properties.get("path", "")
                repo = node.properties.get("repo_name", "")
                if path and repo:
                    index[path] = repo
        return index

    # ── service.json 关系提取 ──

    def _extract_service_interfaces(
        self, chunk: CodeChunk, kg: KnowledgeGraph,
        registry: dict[str, GraphNode], comp_id: str,
    ) -> None:
        for iface_name in chunk.metadata.get("required_interfaces", []):
            iface_id = make_node_id(NodeType.INTERFACE, iface_name)
            self._add_node(registry, kg, GraphNode(
                node_id=iface_id,
                node_type=NodeType.INTERFACE,
                properties={"name": iface_name},
            ))
            kg.add_edge(GraphEdge(
                source_id=comp_id, target_id=iface_id,
                edge_type=EdgeType.REQUIRES_IFACE,
                weight=DEFAULT_EDGE_WEIGHTS[EdgeType.REQUIRES_IFACE],
            ))

    def _resolve_service_deps(
        self, chunk: CodeChunk, kg: KnowledgeGraph, comp_id: str,
    ) -> None:
        for dep_name in chunk.metadata.get("dependencies", []):
            # Conan 格式: "name/[version]" -> 取 name
            clean_name = dep_name.split("/")[0].split("[")[0].strip()
            if not clean_name:
                continue
            target_comp_id = make_node_id(NodeType.COMPONENT, clean_name)
            kg.add_edge(GraphEdge(
                source_id=comp_id, target_id=target_comp_id,
                edge_type=EdgeType.DEPENDS_ON,
                weight=DEFAULT_EDGE_WEIGHTS[EdgeType.DEPENDS_ON],
            ))

    # ── Lua 关系提取 ──

    def _resolve_lua_imports(
        self, chunk: CodeChunk, kg: KnowledgeGraph,
        comp_id: str, module_index: dict[str, str],
    ) -> None:
        for match in _LUA_REQUIRE_RE.finditer(chunk.content):
            module_path = match.group(1)
            target_comp = module_index.get(module_path)
            if not target_comp:
                # 尝试从路径推断组件名: "component.module" -> "component"
                target_comp = module_path.split(".")[0]
            target_comp_id = make_node_id(NodeType.COMPONENT, target_comp)
            kg.add_edge(GraphEdge(
                source_id=comp_id, target_id=target_comp_id,
                edge_type=EdgeType.IMPORTS,
                weight=DEFAULT_EDGE_WEIGHTS[EdgeType.IMPORTS],
            ))

    def _resolve_lua_calls(
        self, chunk: CodeChunk, kg: KnowledgeGraph,
        registry: dict[str, GraphNode], entity_index: dict[str, list[str]],
    ) -> None:
        caller_ids = self._get_chunk_entity_ids(chunk, registry)
        if not caller_ids:
            return

        for match in _LUA_CALL_RE.finditer(chunk.content):
            callee_name = match.group(1) or match.group(2)
            if not callee_name:
                continue
            target_ids = entity_index.get(callee_name, [])
            for caller_id in caller_ids:
                for target_id in target_ids:
                    if caller_id != target_id:
                        kg.add_edge(GraphEdge(
                            source_id=caller_id, target_id=target_id,
                            edge_type=EdgeType.CALLS,
                            weight=DEFAULT_EDGE_WEIGHTS[EdgeType.CALLS],
                        ))

    # ── C/C++ 关系提取 ──

    def _resolve_c_imports(
        self, chunk: CodeChunk, kg: KnowledgeGraph,
        comp_id: str, module_index: dict[str, str],
    ) -> None:
        for match in _C_INCLUDE_RE.finditer(chunk.content):
            include_path = match.group(1)
            target_comp = module_index.get(include_path)
            if not target_comp:
                # 尝试从路径推断: "include/i2c/smbus.h" -> 从已知组件匹配
                continue
            target_comp_id = make_node_id(NodeType.COMPONENT, target_comp)
            kg.add_edge(GraphEdge(
                source_id=comp_id, target_id=target_comp_id,
                edge_type=EdgeType.IMPORTS,
                weight=DEFAULT_EDGE_WEIGHTS[EdgeType.IMPORTS],
            ))

    def _resolve_c_calls(
        self, chunk: CodeChunk, kg: KnowledgeGraph,
        registry: dict[str, GraphNode], entity_index: dict[str, list[str]],
    ) -> None:
        caller_ids = self._get_chunk_entity_ids(chunk, registry)
        if not caller_ids:
            return

        for match in _C_CALL_RE.finditer(chunk.content):
            callee_name = match.group(1)
            if callee_name in _C_KEYWORDS:
                continue
            target_ids = entity_index.get(callee_name, [])
            for caller_id in caller_ids:
                for target_id in target_ids:
                    if caller_id != target_id:
                        kg.add_edge(GraphEdge(
                            source_id=caller_id, target_id=target_id,
                            edge_type=EdgeType.CALLS,
                            weight=DEFAULT_EDGE_WEIGHTS[EdgeType.CALLS],
                        ))

    # ── 工具方法 ──

    def _get_chunk_entity_ids(
        self, chunk: CodeChunk, registry: dict[str, GraphNode],
    ) -> list[str]:
        """获取 chunk 中 symbols 对应的 entity node IDs。"""
        ids = []
        for sym in chunk.symbols:
            if sym.kind in ("function", "class", "method"):
                eid = make_node_id(NodeType.ENTITY, chunk.repo_name, chunk.file_path, sym.name)
                if eid in registry:
                    ids.append(eid)
        return ids
