"""组件元数据模型。

描述 openUBMC 微组件的结构信息，包括依赖关系、接口定义、
MDS 模型类和 IPMI 命令等组件级聚合数据。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ComponentInfo:
    """openUBMC 微组件的元数据信息。

    通过聚合同一组件下所有 CodeChunk 的符号和元数据生成，
    用于组件列表展示和依赖分析。

    Attributes:
        name: 组件名称，如 "sensor", "devmon"
        repo_name: 所属仓库名称
        language: 组件使用的编程语言（逗号分隔）
        description: 组件功能描述
        file_count: 组件包含的源文件数量
        function_count: 组件中的函数总数
        class_count: 组件中的类总数
        dependencies: 构建依赖列表（来自 service.json）
        required_interfaces: 组件需要的接口列表
        provided_interfaces: 组件提供的接口列表
        ipmi_commands: 组件定义的 IPMI 命令列表
        mds_classes: 组件中 MDS 模型定义的类名列表
    """

    name: str
    repo_name: str
    language: str = ""
    description: str = ""
    file_count: int = 0
    function_count: int = 0
    class_count: int = 0
    dependencies: list[str] = field(default_factory=list)
    required_interfaces: list[str] = field(default_factory=list)
    provided_interfaces: list[str] = field(default_factory=list)
    ipmi_commands: list[str] = field(default_factory=list)
    mds_classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """将组件信息转换为字典，用于 JSON 序列化。"""
        return asdict(self)
