"""CLI 命令：ubmc-rag chat —— 启动交互式 RAG 对话。

启动基于 DashScope Qwen LLM 的 ReAct Agent，
支持代码检索工具调用和带证据引用的问答。
"""

from __future__ import annotations

import typer

from ubmc_rag.config.settings import AppConfig


def register(app: typer.Typer) -> None:
    """注册 chat 子命令到 Typer 应用。"""

    @app.command()
    def chat(
        config_path: str = typer.Option(
            "config/default_config.yaml", "--config", "-c",
        ),
        model: str = typer.Option(
            "qwen-plus", "--model", "-m",
            help="DashScope 模型名称",
        ),
        api_key: str = typer.Option(
            "", "--api-key",
            help="DashScope API 密钥（或设置 DASHSCOPE_API_KEY 环境变量）",
        ),
        debug: bool = typer.Option(
            False, "--debug", "-d",
            help="显示 LLM 提示词、响应和工具调用追踪",
        ),
    ) -> None:
        """启动交互式代码 RAG 助手对话。"""
        from ubmc_rag.chat.chain import run_chat

        config = AppConfig.from_yaml(config_path)
        run_chat(config, api_key=api_key or None, model=model, debug=debug)
