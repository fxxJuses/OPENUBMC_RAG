"""C/C++ AST 解析器，使用 Tree-sitter 解析 openUBMC 原生驱动和 libipmi。

提取函数定义、结构体/类定义和 typedef 定义，
自动区分 C 和 C++ 文件并使用对应的 Tree-sitter 语法。
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())

# C++ 文件扩展名集合，用于区分 C 和 C++ 语法
_CPP_EXTENSIONS = frozenset({".cpp", ".hpp", ".cc", ".cxx"})


class CCppParser(BaseParser):
    """C/C++ 代码解析器，自动检测语言并使用对应语法树。

    根据文件扩展名自动选择 C 或 C++ 的 Tree-sitter 解析器，
    提取函数、结构体、类和 typedef 作为独立分块。
    """

    def __init__(self):
        self._c_parser = Parser(C_LANGUAGE)
        self._cpp_parser = Parser(CPP_LANGUAGE)

    @property
    def language(self) -> str:
        """返回主语言标识。实际语言由文件扩展名动态决定。"""
        return "c"

    @property
    def supported_extensions(self) -> list[str]:
        return [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]

    def _get_language_tag(self, file_path: Path) -> str:
        """根据文件扩展名确定语言标签。"""
        return "cpp" if file_path.suffix.lower() in _CPP_EXTENSIONS else "c"

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        """解析 C/C++ 文件，提取函数、结构体和类型定义。"""
        lang_tag = self._get_language_tag(file_path)
        parser = self._cpp_parser if lang_tag == "cpp" else self._c_parser
        tree = parser.parse(content.encode("utf-8"))
        source = content.encode("utf-8")
        rel_path = str(file_path)
        lines = content.splitlines()

        chunks = []
        visited = set()

        for node in self._walk(tree.root_node):
            if node.id in visited:
                continue

            if node.type == "function_definition":
                symbols = self._extract_symbols(node, source, lang_tag)
                chunk = self._node_to_chunk(
                    node, source, rel_path, repo_name, lang_tag, "function",
                    symbols=symbols,
                )
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)

            elif node.type in ("struct_specifier", "class_specifier"):
                # 跳过 typedef 内嵌套的 struct/class（由 type_definition 统一处理）
                if node.parent and node.parent.type != "type_definition":
                    symbols = self._extract_symbols(node, source, lang_tag)
                    chunk = self._node_to_chunk(
                        node, source, rel_path, repo_name, lang_tag, "class",
                        symbols=symbols,
                    )
                    if chunk:
                        chunks.append(chunk)
                        visited.add(node.id)

            elif node.type == "type_definition":
                symbols = self._extract_symbols(node, source, lang_tag)
                chunk = self._node_to_chunk(
                    node, source, rel_path, repo_name, lang_tag, "typedef",
                    symbols=symbols,
                )
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)
                    # 标记嵌套的 struct/class 已处理
                    for child in node.children:
                        if child.type in ("struct_specifier", "class_specifier"):
                            visited.add(child.id)

        # 降级：未提取到分块时，整个文件作为一个分块
        if not chunks:
            symbols = self._extract_symbols(tree.root_node, source, lang_tag)
            chunks = [CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                file_path=rel_path,
                repo_name=repo_name,
                language=lang_tag,
                component_name=repo_name,
                start_line=1,
                end_line=len(lines),
                chunk_type="file",
                symbols=symbols,
            )]

        return chunks

    def _extract_symbols(self, node: Node, source: bytes, lang_tag: str) -> list[Symbol]:
        """从 AST 节点中提取函数和结构体/类符号。"""
        symbols = []
        for child in self._walk(node):
            if child.type == "function_definition":
                name = self._get_func_name(child, source)
                if name:
                    sig = self._get_first_line(child, source)
                    symbols.append(Symbol(
                        name=name, kind="function",
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language=lang_tag, signature=sig,
                    ))
            elif child.type in ("struct_specifier", "class_specifier"):
                name = self._get_struct_name(child, source)
                if name:
                    symbols.append(Symbol(
                        name=name, kind="class",
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language=lang_tag,
                    ))
        return symbols

    def _get_func_name(self, node: Node, source: bytes) -> str | None:
        """从函数定义节点中提取函数名。"""
        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        return source[sub.start_byte:sub.end_byte].decode("utf-8")
            elif child.type in ("identifier",):
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_struct_name(self, node: Node, source: bytes) -> str | None:
        """从结构体/类定义节点中提取类型名。"""
        for child in node.children:
            if child.type == "type_identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None
