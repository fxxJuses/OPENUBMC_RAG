"""Python AST parser using Tree-sitter for openUBMC build tooling (bingo CLI)."""

from __future__ import annotations

import uuid
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol

PYTHON_LANGUAGE = Language(tspython.language())


class PythonParser(BaseParser):
    def __init__(self):
        self._parser = Parser(PYTHON_LANGUAGE)

    @property
    def language(self) -> str:
        return "python"

    @property
    def supported_extensions(self) -> list[str]:
        return [".py"]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        tree = self._parser.parse(content.encode("utf-8"))
        source = content.encode("utf-8")
        rel_path = str(file_path)
        lines = content.splitlines()

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
                chunk = self._node_to_chunk(node, source, rel_path, repo_name, "function")
                if chunk:
                    chunks.append(chunk)
                    visited.add(node.id)

            elif node.type == "class_definition":
                # If class is small enough, include all methods
                line_count = node.end_point[0] - node.start_point[0]
                if line_count <= 200:
                    chunk = self._node_to_chunk(node, source, rel_path, repo_name, "class")
                    if chunk:
                        chunks.append(chunk)
                        visited.add(node.id)
                        # Mark child methods as visited
                        for child in self._walk(node):
                            if child.type == "function_definition":
                                visited.add(child.id)
                else:
                    # Split: class declaration as one chunk, methods separately
                    chunk = self._node_to_chunk(node, source, rel_path, repo_name, "class")
                    if chunk:
                        chunks.append(chunk)
                        visited.add(node.id)

        if not chunks:
            chunks = self._split_by_lines(content, rel_path, repo_name)

        return chunks

    def _extract_symbols(self, node: Node, source: bytes) -> list[Symbol]:
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
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _get_first_line(self, node: Node, source: bytes) -> str | None:
        end = source.find(b"\n", node.start_byte)
        if end == -1:
            end = node.end_byte
        sig = source[node.start_byte:end].decode("utf-8", errors="replace").strip()
        return sig[:200] if sig else None

    def _node_to_chunk(
        self, node: Node, source: bytes, rel_path: str, repo_name: str, chunk_type: str
    ) -> CodeChunk | None:
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            return None

        symbols = self._extract_symbols(node, source)
        return CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            file_path=rel_path,
            repo_name=repo_name,
            language="python",
            component_name=repo_name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            chunk_type=chunk_type,
            symbols=symbols,
        )

    def _split_by_lines(self, content: str, rel_path: str, repo_name: str) -> list[CodeChunk]:
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
                    language="python",
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
