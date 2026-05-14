"""Component metadata model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentInfo:
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
        return {
            "name": self.name,
            "repo_name": self.repo_name,
            "language": self.language,
            "description": self.description,
            "file_count": self.file_count,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "dependencies": self.dependencies,
            "required_interfaces": self.required_interfaces,
            "provided_interfaces": self.provided_interfaces,
            "ipmi_commands": self.ipmi_commands,
            "mds_classes": self.mds_classes,
        }
