"""CLI command: ubmc-rag chat — interactive RAG-powered Q&A."""

from __future__ import annotations

import typer

from ubmc_rag.config.settings import AppConfig


def register(app: typer.Typer) -> None:
    @app.command()
    def chat(
        config_path: str = typer.Option(
            "config/default_config.yaml", "--config", "-c"
        ),
        model: str = typer.Option(
            "qwen-plus", "--model", "-m", help="DashScope model name"
        ),
        api_key: str = typer.Option(
            "", "--api-key", help="DashScope API key (or set DASHSCOPE_API_KEY)"
        ),
        debug: bool = typer.Option(
            False, "--debug", "-d", help="Show LLM prompts, responses and tracing info"
        ),
    ) -> None:
        """Start interactive chat with the code RAG assistant."""
        from ubmc_rag.chat.chain import run_chat

        config = AppConfig.from_yaml(config_path)
        run_chat(config, api_key=api_key or None, model=model, debug=debug)
