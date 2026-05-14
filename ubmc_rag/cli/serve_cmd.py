"""CLI serve command — start the MCP server."""

from __future__ import annotations

import typer
from rich.console import Console

from ubmc_rag.utils.logging import setup_logging

console = Console()


def register(app: typer.Typer):
    @app.command()
    def serve(
        config_path: str = typer.Option("config/default_config.yaml", "--config", "-c"),
        transport: str = typer.Option("stdio", "--transport", help="Transport: stdio or sse"),
        host: str = typer.Option("localhost", "--host", help="Host for SSE transport"),
        port: int = typer.Option(8080, "--port", help="Port for SSE transport"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Start the openUBMC Code RAG MCP server."""
        setup_logging("DEBUG" if verbose else "INFO")

        from ubmc_rag.mcp_server.server import create_server
        from ubmc_rag.config.settings import AppConfig

        config = AppConfig.from_yaml(config_path)
        config.mcp.transport = transport
        config.mcp.host = host
        config.mcp.port = port

        server = create_server(config)

        if transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport="sse", host=host, port=port)
