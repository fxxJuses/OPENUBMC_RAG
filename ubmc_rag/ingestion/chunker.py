"""代码分块协调器 —— 调度各语言解析器，将源文件转换为 CodeChunk 列表。

负责管理解析器实例、根据文件类型选择解析器，
并协调仓库级别的批量解析流程。
"""

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
    """代码分块协调器，管理多语言解析器并协调解析流程。

    根据文件扩展名自动选择对应的解析器，对仓库中的所有可处理文件
    进行解析和分块。

    Attributes:
        config: 应用配置
        file_filter: 文件过滤器，决定哪些文件需要处理
    """

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
        """根据文件扩展名选择对应的解析器。"""
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def parse_file(self, file_path: Path, language: str, repo_name: str) -> list[CodeChunk]:
        """解析单个文件，返回代码分块列表。

        Args:
            file_path: 源文件路径
            language: 检测到的编程语言
            repo_name: 所属仓库名称

        Returns:
            解析得到的代码分块列表，解析失败返回空列表
        """
        parser = self._get_parser(file_path, language)
        if parser is None:
            return []

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                return []
            # 使用相对于仓库根目录的路径，去掉 data/repos/{repo}/ 前缀
            return parser.parse(file_path, content, repo_name)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", file_path, e)
            return []

    def parse_repo(self, repo_path: Path) -> list[CodeChunk]:
        """解析仓库中所有可处理的文件。

        Args:
            repo_path: 仓库根目录路径

        Returns:
            所有文件的代码分块汇总列表
        """
        files = self.file_filter.walk_repo(repo_path)
        all_chunks: list[CodeChunk] = []
        repo_name = repo_path.name

        for file_path, language in files:
            chunks = self.parse_file(file_path, language, repo_name)
            # 归一化 file_path：去掉 data/repos/{repo}/ 前缀，只保留相对仓库的路径
            rel_prefix = f"data/repos/{repo_name}/"
            for c in chunks:
                if c.file_path.startswith(rel_prefix):
                    c.file_path = c.file_path[len(rel_prefix):]
                elif c.file_path.startswith(str(repo_path)):
                    c.file_path = str(Path(c.file_path).relative_to(repo_path))
            all_chunks.extend(chunks)

        logger.info(
            "Parsed [bold]%s[/bold]: %d files -> %d chunks",
            repo_name, len(files), len(all_chunks),
        )
        return all_chunks

    def parse_repos(self, repo_paths: list[Path]) -> list[CodeChunk]:
        """批量解析多个仓库。"""
        all_chunks: list[CodeChunk] = []
        for repo_path in repo_paths:
            all_chunks.extend(self.parse_repo(repo_path))
        logger.info("Total: %d chunks from %d repos", len(all_chunks), len(repo_paths))
        return all_chunks
