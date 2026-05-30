"""CLI 命令：ubmc-rag index —— 构建或更新搜索索引。

完整流程：同步仓库 → 解析代码 → 生成分块 → 构建向量索引 + BM25 索引。
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.ingestion.chunker import Chunker
from ubmc_rag.ingestion.git_sync import GitSync
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.utils.logging import setup_logging

console = Console()


def register(app: typer.Typer):
    """注册 index 子命令到 Typer 应用。"""

    @app.command()
    def index(
        config_path: str = typer.Option(
            "config/default_config.yaml", "--config", "-c",
            help="配置文件 YAML 路径",
        ),
        repos: Optional[str] = typer.Option(
            None, "--repos", "-r",
            help="指定仓库名（逗号分隔）",
        ),
        full_rebuild: bool = typer.Option(
            False, "--full-rebuild",
            help="强制全量重建索引",
        ),
        clone_missing: bool = typer.Option(
            False, "--clone-missing",
            help="自动克隆本地缺失的仓库",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """构建或更新搜索索引。"""
        setup_logging("DEBUG" if verbose else "INFO")
        config = AppConfig.from_yaml(config_path)

        # 同步仓库
        git_sync = GitSync(config)
        repo_names = repos.split(",") if repos else None
        repo_paths = git_sync.sync_all(
            repos=repo_names, clone_missing=clone_missing
        )

        if not repo_paths:
            # 尝试使用已克隆的仓库
            repo_paths = git_sync.list_cloned_repos()
            if repo_names:
                repo_paths = [p for p in repo_paths if p.name in repo_names]

        if not repo_paths:
            console.print(
                "[red]No repos found. Use --clone-missing to clone.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"Found {len(repo_paths)} repos to index")

        # 解析代码并生成分块
        chunker = Chunker(config)
        all_chunks = chunker.parse_repos(repo_paths)

        if not all_chunks:
            console.print(
                "[red]No chunks produced. Check file filters.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"Produced {len(all_chunks)} chunks")

        # 构建索引
        index_mgr = IndexManager(config)
        index_mgr.build_index(all_chunks, full_rebuild=full_rebuild)

        stats = index_mgr.get_stats()
        console.print("[green]Index built successfully![/green]")
        console.print(f"  Chunks: {stats['total_chunks']}")
        console.print(f"  ChromaDB: {stats['chroma_count']}")
        console.print(f"  BM25 docs: {stats['bm25_docs']}")
