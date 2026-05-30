"""文件过滤器，根据语言、扩展名和 .gitignore 规则筛选文件。

在仓库遍历时过滤掉不需要处理的文件（如构建产物、测试文件），
并自动加载仓库的 .gitignore 规则。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pathspec

from ubmc_rag.config.settings import AppConfig

logger = logging.getLogger(__name__)


class FileFilter:
    """文件过滤器，根据配置和 .gitignore 规则筛选可处理文件。

    维护扩展名到语言标识的映射，并缓存每个仓库的 .gitignore 规则。

    Attributes:
        config: 应用配置
    """

    def __init__(self, config: AppConfig):
        self.config = config
        # 构建扩展名 -> 语言标识的映射
        self._ext_lang_map: dict[str, str] = {}
        for lang, lang_conf in config.ingestion.languages.items():
            if lang_conf.enabled:
                for ext in lang_conf.extensions:
                    self._ext_lang_map[ext] = lang

        self._exclude_patterns = config.ingestion.exclude_paths
        # 仓库路径 -> .gitignore PathSpec 缓存
        self._gitignore_specs: dict[str, pathspec.PathSpec] = {}

    def _load_gitignore(self, repo_path: Path) -> pathspec.PathSpec:
        """加载并缓存仓库的 .gitignore 规则。"""
        if str(repo_path) not in self._gitignore_specs:
            gitignore = repo_path / ".gitignore"
            patterns = []
            if gitignore.exists():
                patterns = gitignore.read_text().splitlines()
            self._gitignore_specs[str(repo_path)] = pathspec.PathSpec.from_lines(
                "gitwildmatch", patterns
            )
        return self._gitignore_specs[str(repo_path)]

    def get_language(self, file_path: Path) -> Optional[str]:
        """根据文件扩展名获取对应的语言标识。"""
        return self._ext_lang_map.get(file_path.suffix.lower())

    def is_excluded(self, rel_path: str) -> bool:
        """检查相对路径是否匹配排除模式（如 build/, .git/）。"""
        for pattern in self._exclude_patterns:
            if rel_path.startswith(pattern) or f"/{pattern}" in rel_path:
                return True
        return False

    def should_process(self, file_path: Path, repo_path: Path) -> Optional[str]:
        """判断文件是否应被处理，返回语言标识或 None。

        综合考虑：排除目录、.gitignore 规则、文件扩展名。
        """
        rel = str(file_path.relative_to(repo_path))
        if self.is_excluded(rel):
            return None

        gitignore = self._load_gitignore(repo_path)
        if gitignore.match_file(rel):
            return None

        return self.get_language(file_path)

    def walk_repo(self, repo_path: Path) -> list[tuple[Path, str]]:
        """遍历仓库，返回所有可处理文件的 (路径, 语言) 列表。"""
        results = []
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            lang = self.should_process(file_path, repo_path)
            if lang is not None:
                results.append((file_path, lang))

        logger.info("Found %d processable files in %s", len(results), repo_path.name)
        return results
