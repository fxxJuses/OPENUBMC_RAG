"""Core data models for code chunks and symbols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Symbol:
    name: str
    kind: str  # "function", "class", "method", "variable", "interface", "ipmi_command"
    line_start: int
    line_end: int
    language: str
    signature: Optional[str] = None


@dataclass
class CodeChunk:
    chunk_id: str
    content: str
    file_path: str
    repo_name: str
    language: str  # "lua", "c", "cpp", "python", "json", "markdown"
    component_name: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "method", "mds_model", "mds_ipmi_cmd", "csr_object", "section"
    symbols: list[Symbol] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    @property
    def symbol_names(self) -> list[str]:
        return [s.name for s in self.symbols]

    def to_chroma_metadata(self) -> dict:
        return {
            "file_path": self.file_path,
            "repo_name": self.repo_name,
            "language": self.language,
            "component_name": self.component_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "symbol_names": ",".join(self.symbol_names),
        }
