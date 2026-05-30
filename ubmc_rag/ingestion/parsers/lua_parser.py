"""Lua AST 解析器，使用 Tree-sitter 解析 openUBMC 组件代码。

识别 openUBMC 特有的代码模式，包括：
- class() / singleton() 组件定义
- function init() 入口函数
- function obj:method() 成员方法
- require() 模块依赖
"""

from __future__ import annotations

import uuid
from pathlib import Path

import tree_sitter_lua as tslua
from tree_sitter import Language, Node, Parser

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol

LUA_LANGUAGE = Language(tslua.language())

# openUBMC Lua 特有的模式关键字
_CLASS_PATTERN = b"class"
_SINGLETON_PATTERN = b"singleton"


class LuaParser(BaseParser):
    """Lua 代码解析器，针对 openUBMC 组件开发模式优化。

    解析策略：
    1. 小文件（≤20行）：整体作为一个分块
    2. 中等/大文件：按 AST 提取函数和类定义作为独立分块
    3. 无法提取结构时：按固定行数分割
    """

    def __init__(self):
        self._parser = Parser(LUA_LANGUAGE)

    @property
    def language(self) -> str:
        return "lua"

    @property
    def supported_extensions(self) -> list[str]:
        return [".lua"]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        """解析 Lua 文件，提取函数和组件定义。"""
        tree = self._parser.parse(content.encode("utf-8"))
        source = content.encode("utf-8")
        rel_path = str(file_path)
        lines = content.splitlines()
        total_lines = len(lines)

        # 小文件整体作为一个分块，附带所有符号
        if total_lines <= 20:
            symbols = self._extract_all_symbols(tree.root_node, source)
            return [CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                file_path=rel_path,
                repo_name=repo_name,
                language="lua",
                component_name=repo_name,
                start_line=1,
                end_line=total_lines,
                chunk_type="file",
                symbols=symbols,
            )]

        chunks = []
        visited = set()

        for node in self._walk(tree.root_node):
            if node.id in visited:
                continue

            if node.type == "function_declaration":
                name = self._get_function_name(node, source)
                is_method = ":" in (
                    source[node.start_byte:node.end_byte][:200]
                    .decode("utf-8", errors="replace")
                )
                chunk_type = "method" if is_method else "function"
                symbols = self._extract_all_symbols(node, source)
                chunk = self._node_to_chunk(
                    node, source, rel_path, repo_name, "lua", chunk_type,
                    symbols=symbols,
                )
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)

            elif node.type == "variable_declaration" and node.id not in visited:
                # 检测 class() 或 singleton() 模式：local X = class(...)
                if self._is_class_decl(node, source):
                    symbols = self._extract_all_symbols(node, source)
                    chunk = self._node_to_chunk(
                        node, source, rel_path, repo_name, "lua", "class",
                        symbols=symbols,
                    )
                    if chunk:
                        chunks.append(chunk)
                        visited.add(node.id)

        # 降级：未提取到分块时，按大小决定整体还是按行分割
        if not chunks:
            if total_lines <= 80:
                symbols = self._extract_all_symbols(tree.root_node, source)
                chunks = [CodeChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=content,
                    file_path=rel_path,
                    repo_name=repo_name,
                    language="lua",
                    component_name=repo_name,
                    start_line=1,
                    end_line=total_lines,
                    chunk_type="file",
                    symbols=symbols,
                )]
            else:
                chunks = self._split_by_lines(content, rel_path, repo_name, "lua")

        return chunks

    def _extract_all_symbols(self, node: Node, source: bytes) -> list[Symbol]:
        """从 AST 节点中提取所有函数和类符号。"""
        symbols = []
        for child in self._walk(node):
            if child.type == "function_declaration":
                name = self._get_function_name(child, source)
                if name:
                    sig = self._get_first_line(child, source)
                    symbols.append(Symbol(
                        name=name, kind="function",
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language="lua", signature=sig,
                    ))
            elif child.type == "variable_declaration":
                if self._is_class_decl(child, source):
                    name = self._get_local_name(child, source)
                    if name:
                        symbols.append(Symbol(
                            name=name, kind="class",
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                            language="lua",
                        ))
        return symbols

    def _is_class_decl(self, node: Node, source: bytes) -> bool:
        """检查变量声明是否为 class() 或 singleton() 调用。"""
        for child in node.children:
            if child.type in ("function_call", "expression_list"):
                func_text = source[child.start_byte:child.end_byte]
                if _CLASS_PATTERN in func_text or _SINGLETON_PATTERN in func_text:
                    return True
            # 深层遍历：local X = class(...) 的嵌套结构
            for sub in child.children:
                if sub.type == "function_call":
                    func_text = source[sub.start_byte:sub.end_byte]
                    if _CLASS_PATTERN in func_text or _SINGLETON_PATTERN in func_text:
                        return True
        return False

    def _get_function_name(self, node: Node, source: bytes) -> str | None:
        """从函数声明节点中提取函数名。

        处理普通函数 function init() 和方法 function Obj:method() 两种模式。
        """
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return source[child.start_byte:child.end_byte].decode("utf-8")
            if child.type == "method_index_expression":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_local_name(self, node: Node, source: bytes) -> str | None:
        """从 local 声明中获取变量名。"""
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None
