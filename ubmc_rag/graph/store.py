"""基于 NetworkX 的代码知识图存储。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx

from ubmc_rag.graph.schema import EdgeType, GraphEdge, GraphNode, NodeType

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """内存中的代码知识图，基于 NetworkX DiGraph。

    支持节点/边的增删改查、双向遍历和持久化。
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def add_node(self, node: GraphNode) -> None:
        self._graph.add_node(node.node_id, node_type=node.node_type.value, **node.properties)

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source_id not in self._graph:
            logger.debug("Edge source missing: %s", edge.source_id)
            return
        if edge.target_id not in self._graph:
            logger.debug("Edge target missing: %s", edge.target_id)
            return
        self._graph.add_edge(
            edge.source_id, edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            **edge.properties,
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        if node_id not in self._graph:
            return None
        data = self._graph.nodes[node_id]
        return GraphNode(
            node_id=node_id,
            node_type=NodeType(data.get("node_type", "entity")),
            properties={k: v for k, v in data.items() if k != "node_type"},
        )

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: Optional[list[EdgeType]] = None,
    ) -> list[str]:
        if node_id not in self._graph:
            return []
        neighbors = []
        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                if edge_types is None or EdgeType(data.get("edge_type", "")) in edge_types:
                    neighbors.append(target)
        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if edge_types is None or EdgeType(data.get("edge_type", "")) in edge_types:
                    neighbors.append(source)
        return list(dict.fromkeys(neighbors))

    def expand(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        edge_types: Optional[list[EdgeType]] = None,
    ) -> dict[str, float]:
        """从 seed 节点双向扩展，返回 {node_id: score}，score 基于跳数衰减。"""
        visited: dict[str, float] = {}
        frontier: list[tuple[str, int]] = [(sid, 0) for sid in seed_ids if sid in self._graph]
        for sid, _ in frontier:
            visited[sid] = 1.0

        while frontier:
            next_frontier = []
            for nid, hops in frontier:
                if hops >= max_hops:
                    continue
                for neighbor in self.get_neighbors(nid, "both", edge_types):
                    decay = 1.0 / (hops + 2)
                    if neighbor not in visited or visited[neighbor] < decay:
                        visited[neighbor] = decay
                        next_frontier.append((neighbor, hops + 1))
            frontier = next_frontier
        return visited

    def get_chunk_ids(self, node_ids: list[str]) -> list[tuple[str, float]]:
        """从节点 ID 列表中提取 (chunk_id, score) 对。"""
        results = []
        for nid in node_ids:
            node = self.get_node(nid)
            if node is None:
                continue
            chunk_id = node.properties.get("chunk_id")
            if chunk_id:
                score = self._graph.nodes[nid].get("_score", 1.0)
                results.append((chunk_id, score))
        return results

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def node_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            counts[nt] = counts.get(nt, 0) + 1
        return counts

    def edge_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            counts[et] = counts.get(et, 0) + 1
        return counts

    def save(self, path: Path) -> None:
        data = nx.node_link_data(self._graph)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Knowledge graph saved: %d nodes, %d edges -> %s", self.node_count(), self.edge_count(), path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self._graph = nx.node_link_graph(data, directed=True)
        logger.info("Knowledge graph loaded: %d nodes, %d edges", self.node_count(), self.edge_count())
        return True

    def summary(self) -> dict:
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "node_types": self.node_type_counts(),
            "edge_types": self.edge_type_counts(),
        }
