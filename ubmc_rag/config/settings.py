"""Pydantic V2 configuration management with YAML loading and .env support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


def _load_dotenv(env_path: str | Path | None = None) -> None:
    """Load .env file into os.environ if it exists. Does not override existing values."""
    if env_path is None:
        # Walk up from CWD to find .env
        p = Path.cwd() / ".env"
    else:
        p = Path(env_path)

    if not p.exists():
        return

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


class RepoConfig(BaseModel):
    name: str
    description: str = ""
    language_hint: Optional[str] = None


class GitConfig(BaseModel):
    base_url: str = "https://gitcode.com/openUBMC"
    clone_dir: str = "./data/repos"
    branch: str = "main"
    auth_token: str = ""
    repos: list[RepoConfig] = Field(default_factory=list)


class LanguageConfig(BaseModel):
    extensions: list[str]
    enabled: bool = True
    patterns: list[str] = Field(default_factory=list)


class IngestionConfig(BaseModel):
    languages: dict[str, LanguageConfig] = Field(default_factory=lambda: {
        "lua": LanguageConfig(extensions=[".lua"], enabled=True),
        "c": LanguageConfig(extensions=[".c", ".h"], enabled=True),
        "cpp": LanguageConfig(extensions=[".cpp", ".hpp", ".cc", ".cxx"], enabled=True),
        "python": LanguageConfig(extensions=[".py"], enabled=True),
        "json": LanguageConfig(
            extensions=[".json"],
            patterns=["mds/service.json", "mds/model.json", "mds/ipmi.json", "mds/types.json", "**/*.sr"],
            enabled=True,
        ),
        "markdown": LanguageConfig(extensions=[".md"], enabled=True),
    })
    exclude_paths: list[str] = Field(default_factory=lambda: [
        "build/", "gen/", ".git/", "test/", "tests/", "dt_test/", "third_party/",
    ])
    max_chunk_lines: int = 200
    min_chunk_lines: int = 5
    overlap_lines: int = 5


class IndexingConfig(BaseModel):
    persist_dir: str = "./data/index"
    embedding_provider: str = "dashscope"
    embedding_dim: int = 1024
    dashscope_api_key: str = ""  # Loaded from .env or env var; empty = auto-detect
    chroma_collection: str = "openubmc_code"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def get_dashscope_api_key(self) -> str:
        """Resolve API key: explicit config > env var > error."""
        if self.dashscope_api_key:
            return self.dashscope_api_key
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        return key


class SearchConfig(BaseModel):
    rrf_k: int = 60
    default_top_k: int = 10
    max_top_k: int = 50
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    code_query_bm25_boost: float = 0.2
    symbol_match_boost: float = 1.5
    filepath_match_boost: float = 1.3
    mds_model_match_boost: float = 2.0
    diversity_max_per_file: int = 3


class MCPConfig(BaseModel):
    transport: str = "stdio"
    host: str = "localhost"
    port: int = 8080


class AppConfig(BaseModel):
    git: GitConfig = Field(default_factory=GitConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        _load_dotenv()
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
