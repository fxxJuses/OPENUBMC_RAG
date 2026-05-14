"""Chunking coordinator — orchestrates parsers and produces CodeChunks."""

from __future__ import annotations

import logging
from pathlib import Path

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.ingestion.file_filter import FileFilter
from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.ingestion.parsers.c_cpp_parser import CCppParser
from ubmc_rag.ingestion.parsers.json_parser import JsonParser
from ubmc_rag.ingestion.parsers.lua_parser import LuaParser
from ubmc_rag.ingestion.parsers.markdown_parser import MarkdownParser
from ubmc_rag.ingestion.parsers.python_parser import PythonParser
from ubmc_rag.models.code_chunk import CodeChunk

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, config: AppConfig):
        self.config = config
        self.file_filter = FileFilter(config)
        self._parsers: list[BaseParser] = [
            LuaParser(),
            CCppParser(),
            PythonParser(),
            JsonParser(),
            MarkdownParser(),
        ]

    def _get_parser(self, file_path: Path, language: str) -> BaseParser | None:
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def parse_file(self, file_path: Path, language: str, repo_name: str) -> list[CodeChunk]:
        parser = self._get_parser(file_path, language)
        if parser is None:
            return []

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                return []
            return parser.parse(file_path, content, repo_name)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", file_path, e)
            return []

    def parse_repo(self, repo_path: Path) -> list[CodeChunk]:
        """Parse all processable files in a repo."""
        files = self.file_filter.walk_repo(repo_path)
        all_chunks: list[CodeChunk] = []
        repo_name = repo_path.name

        for file_path, language in files:
            chunks = self.parse_file(file_path, language, repo_name)
            all_chunks.extend(chunks)

        logger.info(
            "Parsed [bold]%s[/bold]: %d files -> %d chunks",
            repo_name, len(files), len(all_chunks),
        )
        return all_chunks

    def parse_repos(self, repo_paths: list[Path]) -> list[CodeChunk]:
        """Parse multiple repos."""
        all_chunks: list[CodeChunk] = []
        for repo_path in repo_paths:
            all_chunks.extend(self.parse_repo(repo_path))
        logger.info("Total: %d chunks from %d repos", len(all_chunks), len(repo_paths))
        return all_chunks
