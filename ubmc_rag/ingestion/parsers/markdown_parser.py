"""Markdown 文档解析器，按标题层级分割文档。

将 Markdown 文件按标题（# ~ ######）切分为独立章节，
每个章节作为独立的 CodeChunk 用于文档检索。
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from ubmc_rag.ingestion.parsers.base_parser import BaseParser
from ubmc_rag.models.code_chunk import CodeChunk, Symbol


class MarkdownParser(BaseParser):
    """Markdown 文档解析器，按标题层级切分为语义章节。

    解析策略：
    1. 有标题的文档：按同级或更高级标题切分
    2. 无标题的文档：整体作为一个分块
    """

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @property
    def language(self) -> str:
        return "markdown"

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".mdx"]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        """解析 Markdown 文件，按标题层级提取章节分块。"""
        rel_path = str(file_path)
        lines = content.splitlines()

        # 收集所有标题的位置、层级和文本
        headings = []
        for i, line in enumerate(lines):
            m = self._HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                headings.append((i, level, title))

        # 无标题时，整体作为一个分块
        if not headings:
            return [CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                file_path=rel_path,
                repo_name=repo_name,
                language="markdown",
                component_name=repo_name,
                start_line=1,
                end_line=len(lines),
                chunk_type="section",
                symbols=[],
            )]

        # 按标题切分：每个章节从当前标题到下一个同级/更高级标题之前
        chunks = []
        for idx, (start, level, title) in enumerate(headings):
            end = len(lines)
            for next_start, next_level, _ in headings[idx + 1:]:
                if next_level <= level:
                    end = next_start
                    break

            section_text = "\n".join(lines[start:end])
            if not section_text.strip():
                continue

            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                content=section_text,
                file_path=rel_path,
                repo_name=repo_name,
                language="markdown",
                component_name=repo_name,
                start_line=start + 1,
                end_line=end,
                chunk_type="section",
                symbols=[Symbol(
                    name=title, kind="section",
                    line_start=start + 1, line_end=end, language="markdown",
                )],
            ))

        return chunks
