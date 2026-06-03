"""LangChain Agent 工具集，封装 HybridSearchEngine 和 IndexManager 的能力。

定义了 Agent 可调用的工具函数：
- search_code: 混合语义+关键词代码搜索
- search_docs: openUBMC 文档知识库搜索
- find_definitions: 符号定义查找
- find_references: 符号引用查找
- list_components: 组件列表和统计
- get_component_deps: 组件依赖分析
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Optional

from langchain_core.tools import tool

from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.models.search_result import SearchResult
from ubmc_rag.search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


def _format_results(results) -> str:
    """将搜索结果格式化为带来源标记的可读文本。

    每条结果标注序号、文件路径、行号范围和分数，
    Agent 可据此在回答中引用来源。
    """
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        d = r.to_dict()
        symbols = d.get("symbols", [])
        sym_str = ""
        if symbols:
            names = [f"{s['name']}({s['kind']})" for s in symbols[:3]]
            sym_str = " symbols: " + ", ".join(names)
        header = (
            f"[{i}] {d['repo']}/{d['file_path']}:{d['start_line']}-{d['end_line']} "
            f"(score={d['score']:.4f}){sym_str}"
        )
        parts.append(f"{header}\n{d['content']}")
    return "\n\n---\n\n".join(parts)


def _format_docs_results(chunks_and_scores: list[tuple]) -> str:
    """将文档搜索结果格式化为可读文本。"""
    if not chunks_and_scores:
        return "No documentation found."

    parts = []
    for i, (chunk, score) in enumerate(chunks_and_scores, 1):
        symbols = chunk.symbols
        sym_str = ""
        if symbols:
            names = [s.name for s in symbols[:2]]
            sym_str = f" section: {', '.join(names)}"
        header = (
            f"[{i}] docs/{chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"(score={score:.4f}){sym_str}"
        )
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def create_tools(
    engine: HybridSearchEngine, index_mgr: IndexManager,
    docs_engine: Optional[HybridSearchEngine] = None,
) -> list:
    """创建 RAG Agent 可使用的 LangChain 工具列表。

    Args:
        engine: 代码混合搜索引擎实例
        index_mgr: 索引管理器实例
        docs_engine: 文档混合搜索引擎实例（可选）

    Returns:
        LangChain @tool 装饰的工具函数列表
    """

    @tool
    def search_code(
        query: str,
        language: Optional[str] = None,
        repo: Optional[str] = None,
        chunk_type: Optional[str] = None,
        top_k: int = 8,
        intent_hint: Optional[str] = None,
    ) -> str:
        """在 openUBMC 代码库中进行混合语义+关键词搜索。
        适用于：理解代码逻辑、查找代码片段、探索架构细节。

        Args:
            intent_hint: 搜索意图提示，可选值：
                "code" — 查找精确代码/函数名（提升关键词匹配权重）
                "semantic" — 理解概念和逻辑（提升语义匹配权重）
                None — 自动检测（默认）
        """
        is_code_query = None
        if intent_hint == "code":
            is_code_query = True
        elif intent_hint == "semantic":
            is_code_query = False

        results = engine.search(
            query=query,
            top_k=min(top_k, 50),
            language=language,
            repo=repo,
            chunk_type=chunk_type,
            is_code_query=is_code_query,
        )
        return _format_results(results)

    @tool
    def search_docs(
        query: str,
        top_k: int = 5,
    ) -> str:
        """搜索 openUBMC 官方文档知识库。
        适用于：查找架构说明、开发指南、API 参考、配置规范、FAQ 等。
        当需要理解 openUBMC 的概念、框架设计、开发流程时优先使用此工具。

        示例查询：
        - "MDS 框架是什么"
        - "CSR 配置方法"
        - "组件如何开发"
        - "IPMI 命令规范"
        """
        if docs_engine is not None:
            results = docs_engine.search(query=query, top_k=min(top_k, 20))
            return _format_results(results)

        # 降级：直接使用 BM25 + 向量搜索
        from ubmc_rag.indexing.embedder import Embedder

        embedder = Embedder(index_mgr.config.indexing)
        query_embedding = embedder.embed_query(query)

        # BM25 搜索
        bm25_results = index_mgr.search_docs_bm25(query, top_k=top_k * 2)
        bm25_scores = {cid: score for cid, score in bm25_results}

        # 向量搜索
        vector_results = index_mgr.search_docs_vector(query_embedding, top_k=top_k * 2)
        vector_scores = {r["chunk_id"]: 1.0 - r["distance"] for r in vector_results}

        # RRF 融合
        all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        k = index_mgr.config.search.rrf_k
        rrf_results = []
        for cid in all_ids:
            bm25_rank = sorted(
                bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True
            ).index(cid) + 1 if cid in bm25_scores else len(bm25_scores) + 1
            vector_rank = sorted(
                vector_scores.keys(), key=lambda x: vector_scores[x], reverse=True
            ).index(cid) + 1 if cid in vector_scores else len(vector_scores) + 1

            rrf_score = 1.0 / (k + bm25_rank) + 1.0 / (k + vector_rank)
            chunk = index_mgr._docs_chunks_index.get(cid)
            if chunk:
                rrf_results.append((chunk, rrf_score))

        rrf_results.sort(key=lambda x: x[1], reverse=True)
        return _format_docs_results(rrf_results[:top_k])

    @tool
    def find_definitions(
        symbol_name: str,
        language: Optional[str] = None,
    ) -> str:
        """查找函数、类、变量的定义位置。
        适用于：需要知道某个符号在哪里定义的。"""
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
        """查找符号的所有引用位置。
        适用于：需要知道某个函数/类在哪里被调用或使用。"""
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
        """获取指定组件的依赖关系、接口、MDS 类和 IPMI 命令。
        适用于：理解组件间的依赖和交互关系。"""
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

    @tool
    def search_multi(
        queries: list[str],
        top_k: int = 10,
    ) -> str:
        """使用多个查询词进行多角度搜索，合并去重后返回结果。
        适用于：复杂问题需要从不同角度检索代码。
        例如查询"sensor和power的关系"时，可以用
        ["sensor require power", "power_mgmt sensor reading",
         "sensor power_monitor"] 三个查询交叉检索。

        Args:
            queries: 2-5 个不同角度的搜索查询
            top_k: 最终返回的结果数量
        """
        queries = queries[:5]
        seen: dict[str, SearchResult] = {}
        for q in queries:
            for r in engine.search(query=q, top_k=top_k * 2):
                cid = f"{r.chunk.repo_name}/{r.chunk.file_path}:{r.chunk.start_line}"
                if cid not in seen or r.score > seen[cid].score:
                    seen[cid] = r
        results = sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]
        return _format_results(results)

    return [
        search_code, search_docs, search_multi, find_definitions,
        find_references, list_components, get_component_deps,
    ]
