"""解析器抽象基类，定义代码解析的通用接口和公共方法。

所有语言解析器（Lua、C/C++、Python、JSON、Markdown）都继承此基类，
实现 language、supported_extensions 和 parse 三个抽象属性/方法。
基类还提供了 AST 遍历、分块转换和按行分割等通用工具方法。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from tree_sitter import Node

from ubmc_rag.models.code_chunk import CodeChunk


class BaseParser(ABC):
    """代码解析器抽象基类。

    子类必须实现：
        - language: 返回解析器对应的语言标识符
        - supported_extensions: 返回支持的文件扩展名列表
        - parse(): 解析单个文件，返回 CodeChunk 列表
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """解析器对应的语言标识符，如 "lua", "c", "python"。

        对于支持多种语言的解析器（如 CCppParser），返回主要语言标识。
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """支持的文件扩展名列表，如 [".lua"], [".c", ".h", ".cpp"]。"""
        ...

    @abstractmethod
    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        """解析单个文件内容，提取语义分块。

        Args:
            file_path: 源文件的完整路径
            content: 文件的文本内容
            repo_name: 所属仓库名称

        Returns:
            解析得到的代码分块列表
        """
        ...

    def can_parse(self, file_path: Path) -> bool:
        """判断解析器是否支持该文件类型（基于文件扩展名）。"""
        return file_path.suffix.lower() in self.supported_extensions

    def _walk(self, node: Node):
        """深度优先遍历 AST 节点树。

        Yields:
            AST 中的每个节点，按深度优先顺序
        """
        yield node
        for child in node.children:
            yield from self._walk(child)

    def _get_first_line(self, node: Node, source: bytes) -> str | None:
        """获取 AST 节点对应源码的第一行，用作函数签名。

        Args:
            node: AST 节点
            source: 完整的源文件字节内容

        Returns:
            第一行文本（截断到 200 字符），失败返回 None
        """
        end = source.find(b"\n", node.start_byte)
        if end == -1:
            end = node.end_byte
        sig = source[node.start_byte:end].decode("utf-8", errors="replace").strip()
        return sig[:200] if sig else None

    def _node_to_chunk(
        self,
        node: Node,
        source: bytes,
        rel_path: str,
        repo_name: str,
        language: str,
        chunk_type: str,
        symbols=None,
    ) -> CodeChunk | None:
        """将 AST 节点转换为 CodeChunk。

        Args:
            node: AST 节点
            source: 完整的源文件字节内容
            rel_path: 文件的相对路径
            repo_name: 所属仓库名称
            language: 语言标识符
            chunk_type: 分块类型，如 "function", "class"
            symbols: 预提取的符号列表（可选，为 None 时使用空列表）

        Returns:
            CodeChunk 实例，如果节点内容为空则返回 None
        """
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            return None

        return CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            file_path=rel_path,
            repo_name=repo_name,
            language=language,
            component_name=repo_name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            chunk_type=chunk_type,
            symbols=symbols or [],
        )

    def _split_by_lines(
        self,
        content: str,
        rel_path: str,
        repo_name: str,
        language: str,
        chunk_size: int = 150,
        overlap: int = 5,
    ) -> list[CodeChunk]:
        """按固定行数分割大文件，作为 AST 解析的降级方案。

        当文件无法通过 AST 提取有意义的结构时，使用简单的按行分割，
        相邻分块之间保留 overlap 行重叠以保持上下文连贯性。

        Args:
            content: 文件文本内容
            rel_path: 文件相对路径
            repo_name: 仓库名称
            language: 语言标识符
            chunk_size: 每个分块的最大行数
            overlap: 相邻分块的重叠行数

        Returns:
            分割后的代码分块列表
        """
        lines = content.splitlines()
        chunks = []
        for i in range(0, len(lines), chunk_size - overlap):
            end = min(i + chunk_size, len(lines))
            text = "\n".join(lines[i:end])
            if text.strip():
                chunks.append(CodeChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=text,
                    file_path=rel_path,
                    repo_name=repo_name,
                    language=language,
                    component_name=repo_name,
                    start_line=i + 1,
                    end_line=end,
                    chunk_type="block",
                ))
        return chunks
