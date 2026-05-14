"""File filtering based on language, extension, and .gitignore rules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pathspec

from ubmc_rag.config.settings import AppConfig

logger = logging.getLogger(__name__)


class FileFilter:
    def __init__(self, config: AppConfig):
        self.config = config
        self._ext_lang_map: dict[str, str] = {}
        for lang, lang_conf in config.ingestion.languages.items():
            if lang_conf.enabled:
                for ext in lang_conf.extensions:
                    self._ext_lang_map[ext] = lang

        self._exclude_patterns = config.ingestion.exclude_paths
        self._gitignore_specs: dict[str, pathspec.PathSpec] = {}

    def _load_gitignore(self, repo_path: Path) -> pathspec.PathSpec:
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
        """Get the language for a file based on its extension."""
        return self._ext_lang_map.get(file_path.suffix.lower())

    def is_excluded(self, rel_path: str) -> bool:
        """Check if a relative path matches exclusion patterns."""
        for pattern in self._exclude_patterns:
            if rel_path.startswith(pattern) or f"/{pattern}" in rel_path:
                return True
        return False

    def should_process(self, file_path: Path, repo_path: Path) -> Optional[str]:
        """Check if a file should be processed. Returns language or None."""
        rel = str(file_path.relative_to(repo_path))
        if self.is_excluded(rel):
            return None

        gitignore = self._load_gitignore(repo_path)
        if gitignore.match_file(rel):
            return None

        return self.get_language(file_path)

    def walk_repo(self, repo_path: Path) -> list[tuple[Path, str]]:
        """Walk a repo and return (file_path, language) pairs for processable files."""
        results = []
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            lang = self.should_process(file_path, repo_path)
            if lang is not None:
                results.append((file_path, lang))

        logger.info("Found %d processable files in %s", len(results), repo_path.name)
        return results
