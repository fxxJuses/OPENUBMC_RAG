"""CLI 命令：ubmc-rag index —— 构建或更新搜索索引。

完整流程：同步仓库 → 解析代码 → 生成分块 → 构建向量索引 + BM25 索引。
代码仓库和文档仓库分别索引到独立的 collection。
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.ingestion.chunker import Chunker
from ubmc_rag.ingestion.git_sync import GitSync
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
            repo_paths = git_sync.list_cloned_repos()
            if repo_names:
                repo_paths = [p for p in repo_paths if p.name in repo_names]

        if not repo_paths:
            console.print(
                "[red]No repos found. Use --clone-missing to clone.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"Found {len(repo_paths)} repos to index")

        # 分离代码仓库和文档仓库
        code_paths = [p for p in repo_paths if p.name != "docs"]
        docs_paths = [p for p in repo_paths if p.name == "docs"]

        index_mgr = IndexManager(config)

        # 索引代码仓库
        if code_paths:
            chunker = Chunker(config)
            all_chunks = chunker.parse_repos(code_paths)

            if not all_chunks:
                console.print(
                    "[yellow]No code chunks produced. Check file filters.[/yellow]"
                )
            else:
                console.print(f"Produced {len(all_chunks)} code chunks")
                index_mgr.build_index(all_chunks, full_rebuild=full_rebuild)

        # 索引文档仓库（只解析 docs/ 子目录，排除项目根目录配置文件）
        if docs_paths:
            console.print("\n[bold cyan]Indexing documentation...[/bold cyan]")
            chunker = Chunker(config)
            # 只解析 docs 仓库的 docs/ 子目录（VitePress 文档内容）
            docs_subdirs = [p / "docs" for p in docs_paths if (p / "docs").exists()]
            if not docs_subdirs:
                docs_subdirs = docs_paths
            docs_chunks = chunker.parse_repos(docs_subdirs)

            if not docs_chunks:
                console.print(
                    "[yellow]No doc chunks produced. Check file filters.[/yellow]"
                )
            else:
                console.print(f"Produced {len(docs_chunks)} doc chunks")
                index_mgr.build_docs_index(docs_chunks, full_rebuild=full_rebuild)

        stats = index_mgr.get_stats()
        console.print("\n[green]Index built successfully![/green]")
        console.print(f"  Code chunks: {stats['code_chunks']}")
        console.print(f"  Code ChromaDB: {stats['chroma_count']}")
        console.print(f"  Code BM25 docs: {stats['bm25_docs']}")
        console.print(f"  Doc chunks: {stats['docs_chunks']}")
        console.print(f"  Doc ChromaDB: {stats['docs_chroma_count']}")
        console.print(f"  Doc BM25 docs: {stats['docs_bm25_docs']}")
