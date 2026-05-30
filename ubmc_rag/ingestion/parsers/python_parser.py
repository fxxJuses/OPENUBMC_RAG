"""Python AST 解析器，使用 Tree-sitter 解析 openUBMC 构建工具（bingo CLI）代码。

提取函数定义和类定义作为独立分块。
对于较大的类，会保留完整的类定义分块，不做方法拆分。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol

PYTHON_LANGUAGE = Language(tspython.language())


class PythonParser(BaseParser):
    """Python 代码解析器。

    解析策略：
    1. 小文件（≤80行）：整体作为一个分块
    2. 大文件：按函数和类定义分别提取
    3. 类定义：若行数≤200则整体保留（含方法），否则仅保留类声明
    """

    def __init__(self):
        self._parser = Parser(PYTHON_LANGUAGE)

    @property
    def language(self) -> str:
        return "python"

    @property
    def supported_extensions(self) -> list[str]:
        return [".py"]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        """解析 Python 文件，提取函数和类定义。"""
        tree = self._parser.parse(content.encode("utf-8"))
        source = content.encode("utf-8")
        rel_path = str(file_path)
        lines = content.splitlines()

        # 小文件整体作为一个分块
        if len(lines) <= 80:
            symbols = self._extract_symbols(tree.root_node, source)
            return [CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                file_path=rel_path,
                repo_name=repo_name,
                language="python",
                component_name=repo_name,
                start_line=1,
                end_line=len(lines),
                chunk_type="file",
                symbols=symbols,
            )]

        chunks = []
        visited = set()

        for node in self._walk(tree.root_node):
            if node.id in visited:
                continue

            if node.type == "function_definition":
                symbols = self._extract_symbols(node, source)
                chunk = self._node_to_chunk(
                    node, source, rel_path, repo_name, "python", "function",
                    symbols=symbols,
                )
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)

            elif node.type == "class_definition":
                line_count = node.end_point[0] - node.start_point[0]
                symbols = self._extract_symbols(node, source)
                chunk = self._node_to_chunk(
                    node, source, rel_path, repo_name, "python", "class",
                    symbols=symbols,
                )
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)
                    # 小类（≤200行）整体保留，标记子方法已处理
                    if line_count <= 200:
                        for child in self._walk(node):
                            if child.type == "function_definition":
                                visited.add(child.id)

        # 降级：按行分割
        if not chunks:
            chunks = self._split_by_lines(content, rel_path, repo_name, "python")

        return chunks

    def _extract_symbols(self, node: Node, source: bytes) -> list[Symbol]:
        """从 AST 节点中提取函数和类符号。"""
        symbols = []
        for child in self._walk(node):
            if child.type == "function_definition":
                name = self._get_name(child, source)
                if name:
                    sig = self._get_first_line(child, source)
                    symbols.append(Symbol(
                        name=name, kind="function",
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language="python", signature=sig,
                    ))
            elif child.type == "class_definition":
                name = self._get_name(child, source)
                if name:
                    symbols.append(Symbol(
                        name=name, kind="class",
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language="python",
                    ))
        return symbols

    def _get_name(self, node: Node, source: bytes) -> str | None:
        """从函数/类定义节点中提取名称。"""
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None
