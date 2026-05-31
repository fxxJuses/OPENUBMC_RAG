"""CLI 命令：ubmc-rag serve —— 启动 MCP 服务器。

支持 stdio 和 SSE 两种传输协议，将代码检索能力
通过 MCP 协议暴露给外部 AI 客户端。
"""

from __future__ import annotations

import typer
from rich.console import Console

from ubmc_rag.utils.logging import setup_logging

console = Console()


def register(app: typer.Typer):
    """注册 serve 子命令到 Typer 应用。"""

    @app.command()
    def serve(
        config_path: str = typer.Option(
            "config/default_config.yaml", "--config", "-c",
        ),
        transport: str = typer.Option(
            "stdio", "--transport",
            help="传输协议: stdio 或 sse",
        ),
        host: str = typer.Option(
            "localhost", "--host",
            help="SSE 模式的监听地址",
        ),
        port: int = typer.Option(
            8080, "--port",
            help="SSE 模式的监听端口",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """启动 openUBMC Code RAG MCP 服务器。"""
        setup_logging("DEBUG" if verbose else "INFO")

        from ubmc_rag.config.settings import AppConfig
        from ubmc_rag.mcp_server.server import create_server

        config = AppConfig.from_yaml(config_path)
        config.mcp.transport = transport
        config.mcp.host = host
        config.mcp.port = port

        server = create_server(config)

        if transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport="sse", host=host, port=port)
