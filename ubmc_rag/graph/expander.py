"""图遍历扩展器 —— 从检索 seed 节点双向扩展获取相关 chunk。

支持三种扩展策略：
1. 双向扩展：从命中节点的 successors（下游依赖）和 predecessors（上游被依赖）扩展
2. 接口消费者扩展：发现实现同一接口的其他组件（对等方）
3. 跨组件 CALLS 定向扩展：从已访问 entity 沿 CALLS 边找到跨组件被调用函数（实际代码 chunk）
"""

from __future__ import annotations

import logging
from typing import Optional

from ubmc_rag.config.settings import GraphConfig
from ubmc_rag.graph.schema import EdgeType, NodeType, make_node_id
from ubmc_rag.graph.store import KnowledgeGraph
from ubmc_rag.models.code_chunk import CodeChunk
from ubmc_rag.models.search_result import SearchResult

logger = logging.getLogger(__name__)

# 用于 BFS 扩展的关系边类型
_EXPANSION_EDGES = [
    EdgeType.CALLS,
    EdgeType.DEPENDS_ON,
    EdgeType.IMPORTS,
    EdgeType.REQUIRES_IFACE,
    EdgeType.PROVIDES_IFACE,
]


class GraphExpander:
    """从检索结果中提取 seed 节点，通过图遍历扩展相关 chunk。"""

    def __init__(
        self,
        graph: KnowledgeGraph,
        chunk_index: dict[str, CodeChunk],
        config: GraphConfig,
    ) -> None:
        self.graph = graph
        self.chunk_index = chunk_index
        self.config = config

        # 构建 entity_name -> [entity_node_ids] 索引
        self._entity_name_index: dict[str, list[str]] = {}
        for nid, data in graph.graph.nodes(data=True):
            if data.get("node_type") == NodeType.ENTITY.value:
                name = data.get("name", "")
                if name:
                    self._entity_name_index.setdefault(name, []).append(nid)

    def expand(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int = 30,
    ) -> list[SearchResult]:
        """从检索结果出发，图遍历扩展相关 chunk。"""
        # 1. 提取 seed 节点（带分数）
        seed_scores = self._extract_seeds(results, query)
        if not seed_scores:
            return []

        seed_ids = list(seed_scores.keys())

        # 2. 双向扩展
        expanded = self._expand_from_seeds(seed_scores)

        # 3. 接口消费者扩展
        expanded = self._expand_interface_consumers(seed_ids, expanded)

        # 4. 收集 chunk 并评分
        graph_results = self._to_search_results(expanded, results, top_k)

        logger.debug(
            "Graph expansion: %d seeds -> %d expanded nodes -> %d results",
            len(seed_ids), len(expanded), len(graph_results),
        )
        return graph_results

    def _expand_from_seeds(self, seed_scores: dict[str, float]) -> dict[str, float]:
        """从带分数的 seed 出发，沿图边扩展。

        BFS 沿 _EXPANSION_EDGES 边常规扩展，完成后通过
        _expand_cross_component_calls 做跨组件 CALLS 定向扩展。
        """
        visited: dict[str, float] = {}
        frontier: list[tuple[str, int]] = []

        for sid, score in seed_scores.items():
            if sid in self.graph.graph:
                visited[sid] = score
                frontier.append((sid, 0))

        while frontier:
            next_frontier = []
            for nid, hops in frontier:
                if hops >= self.config.max_hops:
                    continue
                for neighbor in self.graph.get_neighbors(nid, "both", _EXPANSION_EDGES):
                    parent_score = visited.get(nid, 0)
                    decay = parent_score * 0.5
                    if decay < 0.05:
                        continue
                    if neighbor not in visited or visited[neighbor] < decay:
                        visited[neighbor] = decay
                        next_frontier.append((neighbor, hops + 1))

            frontier = next_frontier

        # BFS 完成后，做跨组件 CALLS 定向扩展
        visited = self._expand_cross_component_calls(seed_scores, visited)

        return visited

    def _expand_cross_component_calls(
        self,
        seed_scores: dict[str, float],
        visited: dict[str, float],
    ) -> dict[str, float]:
        """从 seed entity 及其同文件 entity 沿 CALLS 边发现跨组件代码。

        两步扩展：
        1. 对 seed entity，先通过 BELONGS_TO 找到同文件的所有 entity
           （因为 seed 可能只是 ctor 这类无 CALLS 边的函数）
        2. 对同文件所有 entity 检查出 CALLS 边，找到跨组件目标
        3. 对跨组件目标，通过 CONTAINS 拉入同文件的其他 entity（文件级关联）
        """
        cross_call_results: dict[str, float] = {}

        # 步骤 1：收集所有 seed entity 及其同文件 entity
        call_sources: dict[str, float] = {}  # entity_id -> score
        for nid, score in visited.items():
            node = self.graph.get_node(nid)
            if node is None or node.node_type != NodeType.ENTITY:
                continue
            call_sources[nid] = score

            # 从 entity ID 解析文件路径，通过 CONTAINS 找同文件 entity
            parts = nid.split(":", 2)
            if len(parts) < 3:
                continue
            comp, file_path, _ = parts
            file_nid = f"{comp}:{file_path}"
            file_node = self.graph.get_node(file_nid)
            if file_node is None or file_node.node_type != NodeType.FILE:
                continue
            siblings = self.graph.get_neighbors(file_nid, "out", [EdgeType.CONTAINS])
            for sib_id in siblings:
                if sib_id not in visited and sib_id not in call_sources:
                    call_sources[sib_id] = score * 0.3

        # 步骤 2：对所有 call_sources 检查跨组件 CALLS 边
        for nid, score in call_sources.items():
            source_comp = nid.split(":")[0]
            targets = self.graph.get_neighbors(nid, "out", [EdgeType.CALLS])
            for target_id in targets:
                target_comp = target_id.split(":")[0]
                if target_comp == source_comp:
                    continue
                if target_id in visited:
                    continue
                target_node = self.graph.get_node(target_id)
                if target_node is None or not target_node.properties.get("chunk_id"):
                    continue
                call_score = score * 0.6
                if target_id not in cross_call_results or cross_call_results[target_id] < call_score:
                    cross_call_results[target_id] = call_score

        # 步骤 3：文件级关联——拉入 CALLS 目标同文件 + 同组件主入口文件 entity
        file_associated: dict[str, float] = {}
        # 收集所有涉及的目标组件
        target_comps: dict[str, float] = {}  # comp -> max_call_score
        for target_id, call_score in cross_call_results.items():
            parts = target_id.split(":", 2)
            if len(parts) < 3:
                continue
            target_comp, file_path, _ = parts

            # 3a: 同文件 entity
            file_nid = f"{target_comp}:{file_path}"
            file_node = self.graph.get_node(file_nid)
            if file_node is not None and file_node.node_type == NodeType.FILE:
                siblings = self.graph.get_neighbors(file_nid, "out", [EdgeType.CONTAINS])
                for sibling_id in siblings:
                    if sibling_id == target_id:
                        continue
                    if sibling_id in visited or sibling_id in cross_call_results:
                        continue
                    sib_node = self.graph.get_node(sibling_id)
                    if sib_node is None or not sib_node.properties.get("chunk_id"):
                        continue
                    file_score = call_score * 0.4
                    if sibling_id not in file_associated or file_associated[sibling_id] < file_score:
                        file_associated[sibling_id] = file_score

            # 记录目标组件及其最高分数
            if target_comp not in target_comps or target_comps[target_comp] < call_score:
                target_comps[target_comp] = call_score

        # 3b: 组件主入口文件关联——拉入 *_app.lua 中的 entity
        for target_comp, comp_score in target_comps.items():
            # 找到该组件的 CONTAINS 文件中名为 *_app.lua 的
            comp_nid = target_comp  # component node ID
            comp_files = self.graph.get_neighbors(comp_nid, "out", [EdgeType.CONTAINS])
            for file_nid in comp_files:
                if not file_nid.endswith("_app.lua"):
                    continue
                file_node = self.graph.get_node(file_nid)
                if file_node is None or file_node.node_type != NodeType.FILE:
                    continue
                app_entities = self.graph.get_neighbors(file_nid, "out", [EdgeType.CONTAINS])
                for eid in app_entities[:5]:  # 每个主入口文件最多 5 个 entity
                    if eid in visited or eid in cross_call_results or eid in file_associated:
                        continue
                    e_node = self.graph.get_node(eid)
                    if e_node is None or not e_node.properties.get("chunk_id"):
                        continue
                    app_score = comp_score * 0.3
                    if eid not in file_associated or file_associated[eid] < app_score:
                        file_associated[eid] = app_score

        cross_call_results.update(file_associated)

        # 合并到 visited
        for target_id, score in cross_call_results.items():
            if target_id not in visited or visited[target_id] < score:
                visited[target_id] = score

        if cross_call_results:
            logger.debug(
                "Cross-component CALLS expansion: added %d entities (+ %d file-associated)",
                len(cross_call_results) - len(file_associated),
                len(file_associated),
            )

        return visited

    def _extract_seeds(
        self, results: list[SearchResult], query: str,
    ) -> dict[str, float]:
        """从检索结果和查询文本中提取图 seed 节点。

        Returns:
            {node_id: seed_score} — 高分 seed 优先
        """
        seeds: dict[str, float] = {}

        # 1. 从 Top-N 检索结果中提取 component 和 entity seeds
        top_results = results[:10]
        for i, r in enumerate(top_results):
            relevance = 1.0 / (i + 1)  # 排名衰减

            comp_id = make_node_id(NodeType.COMPONENT, r.chunk.repo_name)
            if comp_id in self.graph.graph:
                seeds[comp_id] = max(seeds.get(comp_id, 0), relevance * 0.8)

            # 从结果的所有 symbol 中提取 entity seeds（最多 3 个）
            for sym in r.chunk.symbols[:3]:
                eid = make_node_id(
                    NodeType.ENTITY, r.chunk.repo_name,
                    r.chunk.file_path, sym.name,
                )
                if eid in self.graph.graph:
                    seeds[eid] = max(seeds.get(eid, 0), relevance * 0.5)

        # 2. 从查询文本匹配 component 名称（高优先级）
        query_lower = query.lower()
        import re
        for token in re.findall(r'[a-zA-Z_]\w{2,}', query_lower):
            comp_id = make_node_id(NodeType.COMPONENT, token)
            if comp_id in self.graph.graph:
                seeds[comp_id] = max(seeds.get(comp_id, 0), 0.9)

        # 3. 从查询文本匹配 entity 名称（低优先级，限制数量）
        entity_count = 0
        for token in re.findall(r'[a-zA-Z_]\w{3,}', query_lower):
            for eid in self._entity_name_index.get(token, [])[:2]:
                if entity_count >= 10:
                    break
                seeds[eid] = max(seeds.get(eid, 0), 0.3)
                entity_count += 1

        return seeds

    def _expand_interface_consumers(
        self,
        seed_ids: list[str],
        expanded: dict[str, float],
    ) -> dict[str, float]:
        """接口消费者扩展：发现使用同一接口的其他组件。"""
        all_seeds = set(seed_ids) | set(expanded.keys())

        for nid in list(all_seeds):
            # 沿 REQUIRES_IFACE 边找到接口
            iface_neighbors = self.graph.get_neighbors(
                nid, direction="out", edge_types=[EdgeType.REQUIRES_IFACE],
            )
            for iface_id in iface_neighbors:
                # 找到所有消费该接口的组件（对等方）
                consumers = self.graph.get_neighbors(
                    iface_id, direction="in", edge_types=[EdgeType.REQUIRES_IFACE],
                )
                for consumer_id in consumers:
                    if consumer_id not in expanded:
                        expanded[consumer_id] = 0.3  # 较低的初始分数

        return expanded

    def _to_search_results(
        self,
        expanded: dict[str, float],
        original_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """将扩展节点转换为 SearchResult 列表。"""
        # 排除原始结果中已有的 chunk
        existing_chunk_ids = {r.chunk.chunk_id for r in original_results}

        results = []
        for nid, score in expanded.items():
            node = self.graph.get_node(nid)
            if node is None:
                continue

            # 从 entity 节点获取 chunk_id
            chunk_id = node.properties.get("chunk_id")
            if not chunk_id or chunk_id in existing_chunk_ids:
                continue

            chunk = self.chunk_index.get(chunk_id)
            if chunk is None:
                continue

            results.append(SearchResult(
                chunk=chunk,
                score=score,
                source="graph",
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
