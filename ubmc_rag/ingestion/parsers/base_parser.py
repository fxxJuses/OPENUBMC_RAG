"""Abstract base parser for all language parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ubmc_rag.models.code_chunk import CodeChunk


class BaseParser(ABC):
    @property
    @abstractmethod
    def language(self) -> str:
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        ...

    @abstractmethod
    def parse(self, file_path: Path, content: str, repo_name: str) -> list[CodeChunk]:
        ...

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
