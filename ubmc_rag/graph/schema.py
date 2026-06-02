"""代码知识图的节点和边类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    COMPONENT = "component"
    FILE = "file"
    ENTITY = "entity"
    INTERFACE = "interface"


class EdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"
    DEFINES = "defines"
    IMPORTS = "imports"
    CALLS = "calls"
    REQUIRES_IFACE = "requires_iface"
    PROVIDES_IFACE = "provides_iface"
    USES_MDS_MODEL = "uses_mds_model"


DEFAULT_EDGE_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.DEPENDS_ON: 1.0,
    EdgeType.IMPORTS: 0.8,
    EdgeType.CALLS: 0.7,
    EdgeType.DEFINES: 0.5,
    EdgeType.REQUIRES_IFACE: 0.9,
    EdgeType.PROVIDES_IFACE: 0.9,
    EdgeType.CONTAINS: 0.3,
    EdgeType.BELONGS_TO: 0.3,
    EdgeType.USES_MDS_MODEL: 0.6,
}


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    properties: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.properties.get("name", self.node_id.rsplit(":", 1)[-1])


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: dict = field(default_factory=dict)


def make_node_id(node_type: NodeType, *parts: str) -> str:
    if node_type == NodeType.COMPONENT:
        return parts[0]
    if node_type == NodeType.INTERFACE:
        return f"iface:{parts[0]}"
    return ":".join(parts)
