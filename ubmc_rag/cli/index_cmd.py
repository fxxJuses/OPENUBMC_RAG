"""CLI index command — build or update the search index."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.ingestion.chunker import Chunker
from ubmc_rag.ingestion.git_sync import GitSync
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.utils.logging import setup_logging

console = Console()


def register(app: typer.Typer):
    @app.command()
    def index(
        config_path: str = typer.Option("config/default_config.yaml", "--config", "-c", help="Config YAML path"),
        repos: Optional[str] = typer.Option(None, "--repos", "-r", help="Comma-separated repo names"),
        full_rebuild: bool = typer.Option(False, "--full-rebuild", help="Force full re-index"),
        clone_missing: bool = typer.Option(False, "--clone-missing", help="Clone missing repos"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Build or update the search index."""
        setup_logging("DEBUG" if verbose else "INFO")
        config = AppConfig.from_yaml(config_path)

        # Sync repos
        git_sync = GitSync(config)
        repo_names = repos.split(",") if repos else None
        repo_paths = git_sync.sync_all(repos=repo_names, clone_missing=clone_missing)

        if not repo_paths:
            # Try already-cloned repos
            repo_paths = git_sync.list_cloned_repos()
            if repo_names:
                repo_paths = [p for p in repo_paths if p.name in repo_names]

        if not repo_paths:
            console.print("[red]No repos found. Use --clone-missing to clone.[/red]")
            raise typer.Exit(1)

        console.print(f"Found {len(repo_paths)} repos to index")

        # Parse and chunk
        chunker = Chunker(config)
        all_chunks = chunker.parse_repos(repo_paths)

        if not all_chunks:
            console.print("[red]No chunks produced. Check file filters.[/red]")
            raise typer.Exit(1)

        console.print(f"Produced {len(all_chunks)} chunks")

        # Build index
        index_mgr = IndexManager(config)
        index_mgr.build_index(all_chunks, full_rebuild=full_rebuild)

        stats = index_mgr.get_stats()
        console.print(f"[green]Index built successfully![/green]")
        console.print(f"  Chunks: {stats['total_chunks']}")
        console.print(f"  ChromaDB: {stats['chroma_count']}")
        console.print(f"  BM25 docs: {stats['bm25_docs']}")
