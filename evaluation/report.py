"""评估报告生成。

提供 Rich 表格、JSON 导出和 before/after 对比三种报告格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.table import Table

from evaluation.retrieval.metrics import RetrievalMetrics
from evaluation.retrieval.runner import ComparisonResult


def print_retrieval_table(metrics: RetrievalMetrics, console: Console) -> None:
    """打印单模式检索指标 Rich 表格。

    Args:
        metrics: 检索评估指标
        console: Rich Console 实例
    """
    table = Table(title="Retrieval Evaluation Results")
    table.add_column("Metric", style="bold cyan", min_width=20)
    table.add_column("Value", style="green", justify="right")

    rows = [
        ("File@1", f"{metrics.file_at_1:.4f}"),
        ("File@3", f"{metrics.file_at_3:.4f}"),
        ("File@5", f"{metrics.file_at_5:.4f}"),
        ("File@10", f"{metrics.file_at_10:.4f}"),
        ("Recall@5", f"{metrics.recall_at_5:.4f}"),
        ("Recall@10", f"{metrics.recall_at_10:.4f}"),
        ("MRR", f"{metrics.mrr:.4f}"),
        ("NDCG@5", f"{metrics.ndcg_at_5:.4f}"),
        ("NDCG@10", f"{metrics.ndcg_at_10:.4f}"),
        ("CategoryHit@5", f"{metrics.category_hit_at_5:.4f}"),
        ("SymbolHit@5", f"{metrics.symbol_hit_at_5:.4f}"),
        ("Total Cases", str(metrics.total_cases)),
    ]

    for name, value in rows:
        table.add_row(name, value)

    console.print(table)


def print_comparison_table(
    comparison: ComparisonResult,
    console: Console,
) -> None:
    """打印多模式对比 Rich 表格。

    Args:
        comparison: 多模式对比结果
        console: Rich Console 实例
    """
    modes = list(comparison.configurations.keys())
    if not modes:
        console.print("[yellow]No evaluation results to compare.[/yellow]")
        return

    table = Table(title=f"Retrieval Comparison — {comparison.dataset_name}")
    table.add_column("Metric", style="bold cyan", min_width=20)

    for mode in modes:
        style = "green" if mode == "hybrid_reranked" else "white"
        table.add_column(mode, style=style, justify="right")

    metric_keys = [
        ("File@1", "file_at_1"),
        ("File@3", "file_at_3"),
        ("File@5", "file_at_5"),
        ("File@10", "file_at_10"),
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5"),
        ("NDCG@10", "ndcg_at_10"),
        ("CategoryHit@5", "category_hit_at_5"),
        ("SymbolHit@5", "symbol_hit_at_5"),
    ]

    for label, key in metric_keys:
        values = []
        for mode in modes:
            m = comparison.configurations[mode]
            values.append(f"{getattr(m, key):.4f}")
        table.add_row(label, *values)

    table.add_row("Total Cases", *(str(comparison.total_cases) for _ in modes))
    console.print(table)


def print_diff(
    before: RetrievalMetrics,
    after: RetrievalMetrics,
    console: Console,
    label_before: str = "Before",
    label_after: str = "After",
) -> None:
    """打印 before/after 对比表，delta 着色。

    Args:
        before: 变更前的指标
        after: 变更后的指标
        console: Rich Console 实例
        label_before: 前列标签
        label_after: 后列标签
    """
    table = Table(title="Before / After Comparison")
    table.add_column("Metric", style="bold cyan", min_width=20)
    table.add_column(label_before, justify="right")
    table.add_column(label_after, justify="right")
    table.add_column("Delta", justify="right")

    metric_keys = [
        ("File@1", "file_at_1"),
        ("File@3", "file_at_3"),
        ("File@5", "file_at_5"),
        ("File@10", "file_at_10"),
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5"),
        ("NDCG@10", "ndcg_at_10"),
        ("CategoryHit@5", "category_hit_at_5"),
        ("SymbolHit@5", "symbol_hit_at_5"),
    ]

    for label, key in metric_keys:
        bv = getattr(before, key)
        av = getattr(after, key)
        delta = av - bv
        if delta > 0.001:
            delta_str = f"[green]+{delta:.4f}[/green]"
        elif delta < -0.001:
            delta_str = f"[red]{delta:.4f}[/red]"
        else:
            delta_str = f"{delta:.4f}"
        table.add_row(label, f"{bv:.4f}", f"{av:.4f}", delta_str)

    console.print(table)


def save_json(
    data: Union[RetrievalMetrics, ComparisonResult],
    output_path: str,
) -> None:
    """将评估结果保存为 JSON 文件。

    Args:
        data: 要保存的评估指标或对比结果
        output_path: 输出文件路径
    """
    if isinstance(data, RetrievalMetrics):
        payload = data.to_dict()
    elif isinstance(data, ComparisonResult):
        payload = data.to_dict()
    else:
        payload = data

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
