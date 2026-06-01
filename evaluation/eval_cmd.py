"""CLI 命令：ubmc-rag eval —— 评估检索质量和 Agent 回答质量。

提供三个子命令：
- `ubmc-rag eval retrieval` — 检索指标评估（支持四种模式对比）
- `ubmc-rag eval agent` — Agent 回答质量评估（LLM-as-Judge）
- `ubmc-rag eval all` — 运行完整评估套件
"""

from __future__ import annotations

import typer
from rich.console import Console

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.utils.logging import setup_logging

console = Console()

_DATASET_DEFAULT = "evaluation/datasets/regression_v1.yaml"


def register(app: typer.Typer):
    """注册 eval 子命令到 Typer 应用。"""

    eval_app = typer.Typer(
        name="eval",
        help="Evaluation commands for retrieval and agent quality",
        no_args_is_help=True,
    )
    app.add_typer(eval_app, name="eval")

    # ------------------------------------------------------------------
    # ubmc-rag eval retrieval
    # ------------------------------------------------------------------

    @eval_app.command()
    def retrieval(
        config_path: str = typer.Option(
            "config/default_config.yaml",
            "--config",
            "-c",
        ),
        dataset: str = typer.Option(
            _DATASET_DEFAULT,
            "--dataset",
            "-d",
        ),
        mode: str = typer.Option(
            "hybrid_reranked",
            "--mode",
            "-m",
            help="Search mode: all, bm25_only, dense_only, hybrid, hybrid_reranked, hybrid_cross_encoder",
        ),
        top_k: int = typer.Option(10, "--top-k", "-k"),
        output: str = typer.Option("", "--output", "-o", help="Save results to JSON"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Run retrieval benchmarks against regression dataset."""
        setup_logging("DEBUG" if verbose else "WARNING")
        config = AppConfig.from_yaml(config_path)

        from evaluation.datasets.loader import load_dataset
        from evaluation.report import print_comparison_table, print_retrieval_table, save_json
        from evaluation.retrieval.evaluator import RetrievalEvaluator
        from evaluation.retrieval.runner import RetrievalRunner

        ds = load_dataset(dataset)
        console.print(
            f"[bold cyan]Loaded dataset:[/bold cyan] {ds.name} ({len(ds.test_cases)} cases)"
        )

        if mode == "all":
            runner = RetrievalRunner()
            result = runner.run_comparison(config, ds, top_k=top_k)
            print_comparison_table(result, console)
            if output:
                save_json(result, output)
                console.print(f"[green]Results saved to {output}[/green]")
        else:
            evaluator = RetrievalEvaluator(config)
            metrics = evaluator.evaluate(ds, top_k=top_k, search_mode=mode)
            print_retrieval_table(metrics, console)
            if output:
                save_json(metrics, output)
                console.print(f"[green]Results saved to {output}[/green]")

    # ------------------------------------------------------------------
    # ubmc-rag eval agent
    # ------------------------------------------------------------------

    @eval_app.command()
    def agent(
        config_path: str = typer.Option(
            "config/default_config.yaml",
            "--config",
            "-c",
        ),
        dataset: str = typer.Option(
            _DATASET_DEFAULT,
            "--dataset",
            "-d",
        ),
        judge_model: str = typer.Option(
            "glm-4-flash",
            "--judge-model",
            help="Model for judging (must differ from answer model)",
        ),
        answer_model: str = typer.Option(
            "qwen-plus",
            "--answer-model",
            help="Model for answering",
        ),
        max_cases: int = typer.Option(
            0,
            "--max-cases",
            help="Limit number of test cases (0 = all)",
        ),
        output: str = typer.Option("", "--output", "-o"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Run agent answer benchmarks with LLM-as-judge."""
        setup_logging("DEBUG" if verbose else "WARNING")
        config = AppConfig.from_yaml(config_path)

        from evaluation.agent.evaluator import AgentEvaluator
        from evaluation.datasets.loader import load_dataset
        from evaluation.report import save_json

        ds = load_dataset(dataset)
        if max_cases > 0:
            ds.test_cases = ds.test_cases[:max_cases]

        console.print(
            f"[bold cyan]Agent Evaluation[/bold cyan]\n"
            f"  Dataset: {ds.name} ({len(ds.test_cases)} cases)\n"
            f"  Answer model: {answer_model}\n"
            f"  Judge model:  {judge_model}"
        )

        evaluator = AgentEvaluator(
            config,
            judge_model=judge_model,
            answer_model=answer_model,
        )
        metrics = evaluator.evaluate_batch(ds)

        # 打印汇总
        console.print("\n[bold]Agent Evaluation Results[/bold]")
        console.print(f"  Solution Quality:  {metrics.avg_solution_quality:.2f}/10")
        console.print(f"  Localization:      {metrics.avg_localization:.2f}/10")
        console.print(f"  Completeness:      {metrics.avg_completeness:.2f}/10")
        console.print(f"  Evidence:          {metrics.avg_evidence_reliability:.2f}/10")
        console.print(f"  Overall Score:     {metrics.avg_overall_score:.2f}/10")
        console.print(f"  Pass Rate:         {metrics.pass_rate:.1%}")
        console.print(f"  Avg Tool Calls:    {metrics.avg_tool_calls:.1f}")
        console.print(f"  Hallucination Rate:{metrics.hallucination_rate:.1%}")

        if output:
            save_json(metrics, output)
            console.print(f"[green]Results saved to {output}[/green]")

    # ------------------------------------------------------------------
    # ubmc-rag eval all
    # ------------------------------------------------------------------

    @eval_app.command("all")
    def eval_all(
        config_path: str = typer.Option(
            "config/default_config.yaml",
            "--config",
            "-c",
        ),
        dataset: str = typer.Option(
            _DATASET_DEFAULT,
            "--dataset",
            "-d",
        ),
        output: str = typer.Option("", "--output", "-o"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ):
        """Run full evaluation suite (retrieval comparison + agent eval)."""
        setup_logging("DEBUG" if verbose else "WARNING")
        config = AppConfig.from_yaml(config_path)

        from evaluation.datasets.loader import load_dataset
        from evaluation.report import print_comparison_table, save_json
        from evaluation.retrieval.runner import RetrievalRunner

        ds = load_dataset(dataset)
        console.print(
            f"[bold cyan]Full Evaluation Suite[/bold cyan]\n"
            f"Dataset: {ds.name} ({len(ds.test_cases)} cases)\n"
        )

        # 1. 检索对比
        console.print("[bold]1. Retrieval Comparison[/bold]")
        runner = RetrievalRunner()
        retrieval_result = runner.run_comparison(config, ds)
        print_comparison_table(retrieval_result, console)

        if output:
            save_json(retrieval_result, output)
            console.print(f"\n[green]Results saved to {output}[/green]")
