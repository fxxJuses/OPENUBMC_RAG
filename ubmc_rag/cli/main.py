"""CLI 入口点和命令注册。

使用 Typer 框架构建命令行应用，注册所有子命令：
index, search, components, serve, chat。
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="ubmc-rag",
    help="Code RAG system for openUBMC micro-component architecture",
    no_args_is_help=True,
)


@app.command()
def version():
    """显示当前版本号。"""
    from ubmc_rag import __version__
    typer.echo(f"ubmc-rag {__version__}")


# 注册子命令
from ubmc_rag.cli.index_cmd import register as register_index  # noqa: E402
from ubmc_rag.cli.search_cmd import register as register_search  # noqa: E402
from ubmc_rag.cli.components_cmd import register as register_components  # noqa: E402
from ubmc_rag.cli.serve_cmd import register as register_serve  # noqa: E402
from ubmc_rag.cli.chat_cmd import register as register_chat  # noqa: E402

register_index(app)
register_search(app)
register_components(app)
register_serve(app)
register_chat(app)

# 注册评估命令
from evaluation.eval_cmd import register as register_eval  # noqa: E402
register_eval(app)

if __name__ == "__main__":
    app()
