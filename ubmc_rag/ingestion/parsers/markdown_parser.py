"""Markdown 文档解析器，按标题层级分割文档。

将 Markdown 文件按标题（# ~ ######）切分为独立章节，
每个章节作为独立的 CodeChunk 用于文档检索。支持 frontmatter 提取
和代码块完整性保护。
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
    1. 提取 YAML frontmatter (title, date) 作为元数据
    2. 按标题 (h1-h4) 切分章节
    3. 大章节(>200行)按段落二次切分，保留代码块完整性
    4. 无标题文档整体作为一个分块
    """

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    _FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    _MAX_SECTION_LINES = 200

    @property
    def language(self) -> str:
        return "markdown"

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".mdx"]

    def _extract_frontmatter(self, content: str) -> tuple[dict, str]:
        """提取 YAML frontmatter，返回 (元数据, 去除frontmatter的内容)。"""
        m = self._FRONTMATTER_RE.match(content)
        if not m:
            return {}, content

        meta = {}
        for line in m.group(1).splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip().strip("\"'")
        return meta, content[m.end():]

    def _split_preserving_code_blocks(self, lines: list[str], max_lines: int) -> list[list[str]]:
        """按段落切分行列表，保持代码块完整性。

        切分点仅在空行处，且不在代码块内部。
        """
        if len(lines) <= max_lines:
            return [lines]

        segments = []
        current = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            current.append(line)

            if not in_code_block and not line.strip() and len(current) >= max_lines:
                segments.append(current)
                current = []

        if current:
            if segments and len(current) < 10:
                segments[-1].extend(current)
            else:
                segments.append(current)

        return segments if segments else [lines]

    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        """解析 Markdown 文件，按标题层级提取章节分块。"""
        rel_path = str(file_path)

        # 提取 frontmatter
        fm_meta, body = self._extract_frontmatter(content)
        title = fm_meta.get("title", "")

        lines = body.splitlines()

        # 收集所有标题的位置、层级和文本
        headings = []
        for i, line in enumerate(lines):
            m = self._HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                heading_title = m.group(2).strip()
                headings.append((i, level, heading_title))

        # 无标题时，整体作为一个分块
        if not headings:
            return [self._make_chunk(
                content=body, file_path=rel_path, repo_name=repo_name,
                start_line=1, end_line=len(lines),
                chunk_type="doc_section",
                symbol_name=title or Path(file_path).stem,
                fm_meta=fm_meta,
            )]

        # 按标题切分
        chunks = []
        for idx, (start, level, heading_title) in enumerate(headings):
            if level > 4:
                continue

            end = len(lines)
            for next_start, next_level, _ in headings[idx + 1:]:
                if next_level <= level:
                    end = next_start
                    break

            section_lines = lines[start:end]
            if not "".join(section_lines).strip():
                continue

            # 大章节二次切分
            if len(section_lines) > self._MAX_SECTION_LINES:
                segments = self._split_preserving_code_blocks(
                    section_lines, self._MAX_SECTION_LINES
                )
                for seg_idx, segment in enumerate(segments):
                    seg_text = "\n".join(segment)
                    seg_start = start + sum(len(s) for s in segments[:seg_idx])
                    chunks.append(self._make_chunk(
                        content=seg_text, file_path=rel_path, repo_name=repo_name,
                        start_line=seg_start + 1,
                        end_line=seg_start + len(segment),
                        chunk_type="doc_section",
                        symbol_name=(
                            f"{heading_title} ({seg_idx+1}/{len(segments)})"
                            if len(segments) > 1 else heading_title
                        ),
                        fm_meta=fm_meta,
                    ))
            else:
                section_text = "\n".join(section_lines)
                chunks.append(self._make_chunk(
                    content=section_text, file_path=rel_path, repo_name=repo_name,
                    start_line=start + 1, end_line=end,
                    chunk_type="doc_section",
                    symbol_name=heading_title,
                    fm_meta=fm_meta,
                ))

        return chunks

    def _make_chunk(
        self, content: str, file_path: str, repo_name: str,
        start_line: int, end_line: int, chunk_type: str,
        symbol_name: str, fm_meta: dict,
    ) -> CodeChunk:
        """创建文档分块的辅助方法。"""
        metadata = {}
        if fm_meta.get("title"):
            metadata["doc_title"] = fm_meta["title"]
        if fm_meta.get("date"):
            metadata["doc_date"] = fm_meta["date"]

        return CodeChunk(
            chunk_id=str(uuid.uuid4()),
            content=content,
            file_path=file_path,
            repo_name=repo_name,
            language="markdown",
            component_name=repo_name,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            symbols=[Symbol(
                name=symbol_name, kind="doc_section",
                line_start=start_line, line_end=end_line, language="markdown",
            )],
            metadata=metadata,
        )
