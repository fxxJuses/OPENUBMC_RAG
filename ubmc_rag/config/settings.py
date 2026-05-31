"""Pydantic V2 配置管理模块，支持 YAML 文件加载和 .env 环境变量。

提供系统运行所需的全部配置项，包括 Git 仓库、文件解析、
向量索引、搜索参数和 MCP 服务器等配置。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


def _load_dotenv(env_path: str | Path | None = None) -> None:
    """加载 .env 文件到环境变量，不覆盖已存在的值。

    从当前工作目录向上查找 .env 文件，解析 key=value 格式的行，
    忽略空行和以 # 开头的注释行。
    """
    if env_path is None:
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
    """单个 Git 仓库配置。

    Attributes:
        name: 仓库名称，对应 GitCode 上的仓库名
        description: 仓库描述
        language_hint: 语言提示（可选），用于指导解析器
    """

    name: str
    description: str = ""
    language_hint: Optional[str] = None


class GitConfig(BaseModel):
    """Git 仓库同步配置。

    Attributes:
        base_url: GitCode 仓库的基础 URL
        clone_dir: 本地克隆目录路径
        branch: 默认分支名
        auth_token: 认证令牌（私有仓库需要）
        repos: 需要索引的仓库列表
    """

    base_url: str = "https://gitcode.com/openUBMC"
    clone_dir: str = "./data/repos"
    branch: str = "main"
    auth_token: str = ""
    repos: list[RepoConfig] = Field(default_factory=list)


class LanguageConfig(BaseModel):
    """单种编程语言的解析配置。

    Attributes:
        extensions: 该语言的文件扩展名列表
        enabled: 是否启用该语言的解析
        patterns: 额外的文件匹配模式（如 MDS JSON 文件路径）
    """

    extensions: list[str]
    enabled: bool = True
    patterns: list[str] = Field(default_factory=list)


class IngestionConfig(BaseModel):
    """代码文件解析和分块配置。

    控制哪些文件会被解析，以及分块的大小和重叠策略。

    Attributes:
        languages: 各语言的解析配置，键为语言名
        exclude_paths: 排除的目录路径列表
        max_chunk_lines: 分块的最大行数
        min_chunk_lines: 分块的最小行数（低于此值的可能被合并）
        overlap_lines: 分块之间的重叠行数，确保跨分块的代码不丢失上下文
    """

    languages: dict[str, LanguageConfig] = Field(default_factory=lambda: {
        "lua": LanguageConfig(extensions=[".lua"], enabled=True),
        "c": LanguageConfig(extensions=[".c", ".h"], enabled=True),
        "cpp": LanguageConfig(extensions=[".cpp", ".hpp", ".cc", ".cxx"], enabled=True),
        "python": LanguageConfig(extensions=[".py"], enabled=True),
        "json": LanguageConfig(
            extensions=[".json"],
            patterns=[
                "mds/service.json", "mds/model.json",
                "mds/ipmi.json", "mds/types.json",
            ],
            enabled=True,
        ),
        "markdown": LanguageConfig(extensions=[".md"], enabled=False),
    })
    exclude_paths: list[str] = Field(default_factory=lambda: [
        "build/", "gen/", ".git/", "test/", "tests/",
        "dt_test/", "third_party/",
    ])
    max_chunk_lines: int = 200
    min_chunk_lines: int = 5
    overlap_lines: int = 5


class IndexingConfig(BaseModel):
    """向量索引和嵌入模型配置。

    控制向量数据库（ChromaDB）、嵌入模型（DashScope）和
    BM25 关键词索引的参数。

    Attributes:
        persist_dir: 索引持久化存储目录
        embedding_provider: 嵌入服务提供方，目前仅支持 "dashscope"
        embedding_dim: 嵌入向量的维度
        dashscope_api_key: DashScope API 密钥，为空则从环境变量读取
        chroma_collection: ChromaDB 中的集合名称
        bm25_k1: BM25 的词频饱和参数 k1
        bm25_b: BM25 的文档长度归一化参数 b
    """

    persist_dir: str = "./data/index"
    embedding_provider: str = "dashscope"
    embedding_dim: int = 1024
    dashscope_api_key: str = ""
    chroma_collection: str = "openubmc_code"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def get_dashscope_api_key(self) -> str:
        """获取 DashScope API 密钥，优先使用配置值，否则从环境变量读取。"""
        if self.dashscope_api_key:
            return self.dashscope_api_key
        return os.environ.get("DASHSCOPE_API_KEY", "")


class SearchConfig(BaseModel):
    """混合搜索和重排序配置。

    控制 BM25 + Dense 双路检索的 RRF 融合参数、
    结果重排序的提升规则和多样性过滤。

    Attributes:
        rrf_k: Reciprocal Rank Fusion 的平滑参数 k
        default_top_k: 默认返回的结果数量
        max_top_k: 最大返回的结果数量上限
        bm25_weight: BM25 检索结果的基础权重
        dense_weight: 向量检索结果的基础权重
        code_query_bm25_boost: 代码类查询时 BM25 的额外提升
        symbol_match_boost: 符号名精确匹配的分数提升倍数
        filepath_match_boost: 文件路径匹配的分数提升倍数
        mds_model_match_boost: MDS 模型类名匹配的分数提升倍数
        diversity_max_per_file: 同一文件在结果中的最大出现次数
    """

    rrf_k: int = 60
    default_top_k: int = 10
    max_top_k: int = 50
    bm25_weight: float = 0.60
    dense_weight: float = 0.40
    code_query_bm25_boost: float = 0.15
    symbol_match_boost: float = 1.5
    filepath_match_boost: float = 1.3
    mds_model_match_boost: float = 2.0
    content_keyword_boost: float = 1.2
    diversity_max_per_file: int = 3


class MCPConfig(BaseModel):
    """MCP 服务器传输配置。

    Attributes:
        transport: 传输协议，可选 "stdio" 或 "sse"
        host: SSE 模式下的监听地址
        port: SSE 模式下的监听端口
    """

    transport: str = "stdio"
    host: str = "localhost"
    port: int = 8080


class AppConfig(BaseModel):
    """应用总配置，聚合所有子模块配置项。

    支持从 YAML 文件加载配置，同时自动加载 .env 环境变量。

    Attributes:
        git: Git 仓库同步配置
        ingestion: 代码解析和分块配置
        indexing: 向量索引配置
        search: 搜索和重排序配置
        mcp: MCP 服务器配置
    """

    git: GitConfig = Field(default_factory=GitConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """从 YAML 文件加载配置，文件不存在时返回默认配置。

        加载前会自动读取 .env 文件中的环境变量。
        """
        _load_dotenv()
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
