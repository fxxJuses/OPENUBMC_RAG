"""C/C++ AST parser using Tree-sitter for openUBMC native drivers and libipmi."""

from __future__ import annotations

import uuid
from pathlib import Path

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())


class CCppParser(BaseParser):
    def __init__(self):
        self._c_parser = Parser(C_LANGUAGE)
        self._cpp_parser = Parser(CPP_LANGUAGE)

    @property
    def language(self) -> str:
        return "c"

    @property
    def supported_extensions(self) -> list[str]:
        return [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]

    def _get_language_tag(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        return "cpp" if ext in (".cpp", ".hpp", ".cc", ".cxx") else "c"

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
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
                chunk = self._node_to_chunk(node, source, rel_path, repo_name, lang_tag, "function")
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)

            elif node.type in ("struct_specifier", "class_specifier"):
                # Only top-level struct/class (not nested inside typedef already handled)
                if node.parent and node.parent.type != "type_definition":
                    chunk = self._node_to_chunk(node, source, rel_path, repo_name, lang_tag, "class")
                    if chunk:
                        chunks.append(chunk)
                        visited.add(node.id)

            elif node.type == "type_definition":
                chunk = self._node_to_chunk(node, source, rel_path, repo_name, lang_tag, "typedef")
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)
                    # Mark nested struct as visited
                    for child in node.children:
                        if child.type in ("struct_specifier", "class_specifier"):
                            visited.add(child.id)

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
        """Extract function name from function_definition."""
        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        return source[sub.start_byte:sub.end_byte].decode("utf-8")
            elif child.type in ("identifier",):
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_struct_name(self, node: Node, source: bytes) -> str | None:
        for child in node.children:
            if child.type == "type_identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_first_line(self, node: Node, source: bytes) -> str | None:
        end = source.find(b"\n", node.start_byte)
        if end == -1:
            end = node.end_byte
        sig = source[node.start_byte:end].decode("utf-8", errors="replace").strip()
        return sig[:200] if sig else None

    def _node_to_chunk(
        self, node: Node, source: bytes, rel_path: str, repo_name: str,
        lang_tag: str, chunk_type: str,
    ) -> CodeChunk | None:
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            return None

        symbols = self._extract_symbols(node, source, lang_tag)
        return CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            file_path=rel_path,
            repo_name=repo_name,
            language=lang_tag,
            component_name=repo_name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            chunk_type=chunk_type,
            symbols=symbols,
        )

    def _split_by_lines(
        self, content: str, rel_path: str, repo_name: str, lang_tag: str
    ) -> list[CodeChunk]:
        lines = content.splitlines()
        chunk_size = 150
        overlap = 5
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
                    language=lang_tag,
                    component_name=repo_name,
                    start_line=i + 1,
                    end_line=end,
                    chunk_type="block",
                ))
        return chunks

    def _walk(self, node: Node):
        yield node
        for child in node.children:
            yield from self._walk(child)
