"""CLI search command — search the indexed codebase."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.embedder import Embedder
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.search.hybrid_search import HybridSearchEngine
from ubmc_rag.utils.logging import setup_logging

console = Console()


def register(app: typer.Typer):
    @app.command()
    def search(
        query: str = typer.Argument(help="Search query"),
        config_path: str = typer.Option("config/default_config.yaml", "--config", "-c"),
        language: Optional[str] = typer.Option(None, "--language", "-l", help="Filter by language"),
        repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Filter by repo"),
        chunk_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by chunk type"),
        top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
        code: bool = typer.Option(False, "--code", help="Treat as code snippet query"),
        format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, plain"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Search the indexed openUBMC codebase."""
        setup_logging("DEBUG" if verbose else "WARNING")
        config = AppConfig.from_yaml(config_path)

        # Load index
        index_mgr = IndexManager(config)
        if not index_mgr.load_index():
            console.print("[red]No index found. Run 'ubmc-rag index' first.[/red]")
            raise typer.Exit(1)

        chunks = index_mgr.get_all_chunks()

        # Build search engine
        engine = HybridSearchEngine(
            embedder=index_mgr.embedder,
            vector_store=index_mgr.vector_store,
            bm25=index_mgr.bm25,
            config=config,
        )
        engine.set_chunk_index(chunks)

        # Search
        results = engine.search(
            query=query,
            top_k=top_k,
            language=language,
            repo=repo,
            chunk_type=chunk_type,
            is_code_query=code if code else None,
        )

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        if format == "json":
            print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        elif format == "plain":
            for r in results:
                d = r.to_dict()
                print(f"--- {d['file_path']}:{d['start_line']}-{d['end_line']} (score: {d['score']}) ---")
                preview = d["content"][:300]
                print(preview)
                if len(d["content"]) > 300:
                    print("...")
                print()
        else:
            table = Table(title=f"Search results for: {query}")
            table.add_column("File", style="cyan", max_width=40)
            table.add_column("Lines", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Score", style="magenta")
            table.add_column("Preview", max_width=60)

            for r in results:
                d = r.to_dict()
                preview = d["content"][:80].replace("\n", " ")
                table.add_row(
                    d["file_path"],
                    f"{d['start_line']}-{d['end_line']}",
                    d["chunk_type"],
                    f"{d['score']:.4f}",
                    preview,
                )

            console.print(table)
