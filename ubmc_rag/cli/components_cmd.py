"""CLI components command — list indexed components and their info."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.models.component_info import ComponentInfo
from ubmc_rag.utils.logging import setup_logging

console = Console()


def register(app: typer.Typer):
    @app.command()
    def components(
        config_path: str = typer.Option("config/default_config.yaml", "--config", "-c"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed stats"),
        format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
    ):
        """List indexed openUBMC components."""
        setup_logging("WARNING")
        config = AppConfig.from_yaml(config_path)

        index_mgr = IndexManager(config)
        if not index_mgr.load_index():
            console.print("[red]No index found. Run 'ubmc-rag index' first.[/red]")
            raise typer.Exit(1)

        chunks = index_mgr.get_all_chunks()

        # Aggregate by component
        comp_data: dict[str, dict] = defaultdict(lambda: {
            "files": set(), "functions": 0, "classes": 0,
            "languages": set(), "symbols": [],
        })

        for chunk in chunks:
            comp = chunk.component_name or chunk.repo_name
            data = comp_data[comp]
            data["files"].add(chunk.file_path)
            data["languages"].add(chunk.language)
            for sym in chunk.symbols:
                if sym.kind == "function":
                    data["functions"] += 1
                elif sym.kind == "class":
                    data["classes"] += 1
                data["symbols"].append(sym.name)

        components = []
        for name, data in sorted(comp_data.items()):
            info = ComponentInfo(
                name=name,
                repo_name=name,
                language=", ".join(sorted(data["languages"])),
                file_count=len(data["files"]),
                function_count=data["functions"],
                class_count=data["classes"],
            )
            components.append(info)

        if format == "json":
            print(json.dumps([c.to_dict() for c in components], indent=2, ensure_ascii=False))
        else:
            table = Table(title="openUBMC Components")
            table.add_column("Component", style="cyan")
            table.add_column("Languages", style="green")
            table.add_column("Files", style="yellow", justify="right")
            table.add_column("Functions", style="magenta", justify="right")
            table.add_column("Classes", style="blue", justify="right")

            if verbose:
                table.add_column("Top Symbols", max_width=50)

            for c in components:
                row = [c.name, c.language, str(c.file_count), str(c.function_count), str(c.class_count)]
                if verbose:
                    symbols = comp_data[c.name]["symbols"][:10]
                    row.append(", ".join(symbols))
                table.add_row(*row)

            console.print(table)
            console.print(f"\nTotal: {len(components)} components, {sum(c.file_count for c in components)} files")
