"""代码分块和符号的核心数据模型。

定义了代码检索系统中两个最基础的数据结构：
- Symbol: 表示代码中的符号（函数、类、变量等）
- CodeChunk: 表示代码分块，是索引和检索的基本单元
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _parse_symbols_from_metadata(meta: dict) -> list[Symbol]:
    """从 ChromaDB metadata 恢复 Symbol 列表。

    ChromaDB 不支持 list 类型，symbol_names 以逗号分隔字符串存储。
    """
    names = meta.get("symbol_names", "")
    if not names:
        return []
    language = meta.get("language", "")
    symbols = []
    for name in names.split(","):
        name = name.strip()
        if name:
            symbols.append(Symbol(
                name=name, kind="unknown", line_start=0, line_end=0,
                language=language,
            ))
    return symbols


@dataclass
class Symbol:
    """代码符号，表示代码中的一个命名实体。

    Attributes:
        name: 符号名称，如函数名 "init"、类名 "ThresholdSensor"
        kind: 符号类型，可选值包括 "function", "class", "method",
              "variable", "interface", "ipmi_command", "dependency", "section"
        line_start: 符号在源文件中的起始行号（1-based）
        line_end: 符号在源文件中的结束行号（1-based）
        language: 所属编程语言，如 "lua", "c", "python", "json"
        signature: 符号签名，如函数的第一行声明（可选）
    """

    name: str
    kind: str
    line_start: int
    line_end: int
    language: str
    signature: Optional[str] = None


@dataclass
class CodeChunk:
    """代码分块，是索引和检索的基本单元。

    每个分块代表源文件中的一个语义片段，可以是函数、类、MDS 模型定义、
    IPMI 命令等。分块会被送入向量数据库（ChromaDB）和关键词索引（BM25）进行检索。

    Attributes:
        chunk_id: 唯一标识符（UUID）
        content: 分块的源代码文本内容
        file_path: 源文件路径
        repo_name: 所属仓库名称（如 "sensor", "libipmi"）
        language: 编程语言，可选值："lua", "c", "cpp", "python", "json", "markdown"
        component_name: 所属组件名称，通常与 repo_name 相同
        start_line: 分块在源文件中的起始行号（1-based）
        end_line: 分块在源文件中的结束行号（1-based）
        chunk_type: 分块类型，可选值："function", "class", "method",
                    "mds_model", "mds_ipmi_cmd", "csr_object", "section", "file", "block"
        symbols: 分块中包含的符号列表
        metadata: 额外的元数据字典，存储类型特定的信息
        embedding: 向量嵌入表示，索引构建后会被清空以节省内存
    """

    chunk_id: str
    content: str
    file_path: str
    repo_name: str
    language: str
    component_name: str
    start_line: int
    end_line: int
    chunk_type: str
    symbols: list[Symbol] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    @property
    def symbol_names(self) -> list[str]:
        """获取分块中所有符号的名称列表。"""
        return [s.name for s in self.symbols]

    @classmethod
    def from_chroma_metadata(cls, chunk_id: str, content: str, meta: dict) -> CodeChunk:
        """从 ChromaDB 存储的元数据字典恢复 CodeChunk。

        Args:
            chunk_id: 分块唯一标识符
            content: 分块文本内容
            meta: ChromaDB metadata 字典

        Returns:
            重建的 CodeChunk 对象
        """
        return cls(
            chunk_id=chunk_id,
            content=content,
            file_path=meta.get("file_path", ""),
            repo_name=meta.get("repo_name", ""),
            language=meta.get("language", ""),
            component_name=meta.get("component_name", ""),
            start_line=meta.get("start_line", 0),
            end_line=meta.get("end_line", 0),
            chunk_type=meta.get("chunk_type", ""),
            symbols=_parse_symbols_from_metadata(meta),
        )

    def to_chroma_metadata(self) -> dict:
        """将分块转换为 ChromaDB 可存储的元数据字典。

        注意：ChromaDB 的 metadata 不支持列表类型，因此 symbol_names
        会用逗号拼接为字符串存储。
        """
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
