"""Lua AST parser using Tree-sitter for openUBMC component code."""

from __future__ import annotations

import uuid
from pathlib import Path

import tree_sitter_lua as tslua
from tree_sitter import Language, Node, Parser

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol

LUA_LANGUAGE = Language(tslua.language())

# openUBMC Lua patterns
_CLASS_PATTERN = b"class"
_SINGLETON_PATTERN = b"singleton"
_REQUIRE_PREFIX = b"require"


class LuaParser(BaseParser):
    def __init__(self):
        self._parser = Parser(LUA_LANGUAGE)

    @property
    def language(self) -> str:
        return "lua"

    @property
    def supported_extensions(self) -> list[str]:
        return [".lua"]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        tree = self._parser.parse(content.encode("utf-8"))
        source = content.encode("utf-8")
        rel_path = str(file_path)
        lines = content.splitlines()
        total_lines = len(lines)

        # Very small files: single chunk with symbols
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

            # function init() or function obj:method() — tree-sitter-lua uses "function_declaration"
            if node.type == "function_declaration":
                name = self._get_function_name(node, source)
                is_method = ":" in (source[node.start_byte:node.end_byte][:200].decode("utf-8", errors="replace"))
                chunk_type = "method" if is_method else "function"
                chunk = self._node_to_chunk(node, source, rel_path, repo_name, chunk_type)
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)

            elif node.type == "variable_declaration" and node.id not in visited:
                # Detect class() or singleton() patterns: local X = class(...)
                if self._is_class_decl(node, source):
                    chunk = self._node_to_chunk(node, source, rel_path, repo_name, "class")
                    if chunk:
                        chunks.append(chunk)
                        visited.add(node.id)

        # Fallback: if no chunks found, use whole file or split by lines
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
                chunks = self._split_by_lines(content, rel_path, repo_name)

        return chunks

    def _extract_all_symbols(self, node: Node, source: bytes) -> list[Symbol]:
        symbols = []
        for child in self._walk(node):
            if child.type == "function_declaration":
                name = self._get_function_name(child, source)
                if name:
                    start_line = child.start_point[0] + 1
                    end_line = child.end_point[0] + 1
                    sig = self._get_signature(child, source)
                    symbols.append(Symbol(
                        name=name, kind="function",
                        line_start=start_line, line_end=end_line,
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

    def _node_to_chunk(
        self, node: Node, source: bytes, rel_path: str, repo_name: str, chunk_type: str
    ) -> CodeChunk | None:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

        if not text.strip():
            return None

        symbols = self._extract_all_symbols(node, source)
        return CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            file_path=rel_path,
            repo_name=repo_name,
            language="lua",
            component_name=repo_name,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            symbols=symbols,
        )

    def _is_class_decl(self, node: Node, source: bytes) -> bool:
        """Check if a variable_declaration is a class() or singleton() call."""
        for child in node.children:
            if child.type in ("function_call", "expression_list"):
                func_text = source[child.start_byte:child.end_byte]
                if _CLASS_PATTERN in func_text or _SINGLETON_PATTERN in func_text:
                    return True
            # Walk deeper for variable_declaration -> assignment_statement -> expression_list -> function_call
            for sub in child.children:
                if sub.type == "function_call":
                    func_text = source[sub.start_byte:sub.end_byte]
                    if _CLASS_PATTERN in func_text or _SINGLETON_PATTERN in func_text:
                        return True
        return False

    def _get_function_name(self, node: Node, source: bytes) -> str | None:
        """Extract function name from a function_declaration node."""
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return source[child.start_byte:child.end_byte].decode("utf-8")
            if child.type == "method_index_expression":
                # function Obj:method() — get the method name after ':'
                for sub in child.children:
                    if sub.type == "identifier":
                        # Last identifier after ':' is the method name
                        pass
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_local_name(self, node: Node, source: bytes) -> str | None:
        """Get variable name from local_declaration."""
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_signature(self, node: Node, source: bytes) -> str | None:
        """Get the first line of a function as its signature."""
        first_line_end = source.find(b"\n", node.start_byte)
        if first_line_end == -1:
            first_line_end = node.end_byte
        sig = source[node.start_byte:first_line_end].decode("utf-8").strip()
        return sig[:200] if sig else None

    def _split_by_lines(self, content: str, rel_path: str, repo_name: str) -> list[CodeChunk]:
        """Fallback: split large files by line ranges."""
        lines = content.splitlines()
        chunk_size = 150
        overlap = 5
        chunks = []

        for i in range(0, len(lines), chunk_size - overlap):
            start = i
            end = min(i + chunk_size, len(lines))
            text = "\n".join(lines[start:end])
            if text.strip():
                chunks.append(CodeChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=text,
                    file_path=rel_path,
                    repo_name=repo_name,
                    language="lua",
                    component_name=repo_name,
                    start_line=start + 1,
                    end_line=end,
                    chunk_type="block",
                ))

        return chunks

    def _walk(self, node: Node):
        """Depth-first traversal of AST."""
        yield node
        for child in node.children:
            yield from self._walk(child)
