"""评估报告生成。

提供 Rich 表格、JSON 导出和 before/after 对比三种报告格式。
支持新指标（Precision@K, MAP）和按类别/难度分组统计。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.table import Table

from evaluation.retrieval.metrics import RetrievalMetrics
from evaluation.retrieval.runner import ComparisonResult

# 核心指标行定义：(显示名, 字段名)
_METRIC_ROWS = [
    ("File@1", "file_at_1"),
    ("File@3", "file_at_3"),
    ("File@5", "file_at_5"),
    ("File@10", "file_at_10"),
    ("Precision@5", "precision_at_5"),
    ("Precision@10", "precision_at_10"),
    ("Recall@5", "recall_at_5"),
    ("Recall@10", "recall_at_10"),
    ("MRR", "mrr"),
    ("MAP", "map_score"),
    ("NDCG@5", "ndcg_at_5"),
    ("NDCG@10", "ndcg_at_10"),
    ("CategoryHit@5", "category_hit_at_5"),
    ("SymbolHit@5", "symbol_hit_at_5"),
]


def print_retrieval_table(metrics: RetrievalMetrics, console: Console) -> None:
    """打印单模式检索指标 Rich 表格。"""
    table = Table(title="Retrieval Evaluation Results")
    table.add_column("Metric", style="bold cyan", min_width=20)
    table.add_column("Value", style="green", justify="right")

    for name, key in _METRIC_ROWS:
        value = getattr(metrics, key)
        table.add_row(name, f"{value:.4f}")

    table.add_row("Total Cases", str(metrics.total_cases))

    # 置信区间
    if metrics.confidence_intervals:
        console.print(table)
        ci_table = Table(title="95% Confidence Intervals (Bootstrap)")
        ci_table.add_column("Metric", style="bold cyan", min_width=20)
        ci_table.add_column("Low", style="yellow", justify="right")
        ci_table.add_column("High", style="yellow", justify="right")
        for metric_name, (lo, hi) in metrics.confidence_intervals.items():
            ci_table.add_row(metric_name, f"{lo:.4f}", f"{hi:.4f}")
        console.print(ci_table)
    else:
        console.print(table)

    # 分组统计
    if metrics.by_category:
        print_breakdown_table(metrics.by_category, "Category", console)
    if metrics.by_difficulty:
        print_breakdown_table(metrics.by_difficulty, "Difficulty", console)


def print_comparison_table(
    comparison: ComparisonResult,
    console: Console,
) -> None:
    """打印多模式对比 Rich 表格。"""
    modes = list(comparison.configurations.keys())
    if not modes:
        console.print("[yellow]No evaluation results to compare.[/yellow]")
        return

    table = Table(title=f"Retrieval Comparison — {comparison.dataset_name}")
    table.add_column("Metric", style="bold cyan", min_width=20)

    for mode in modes:
        style = "green" if mode == "hybrid_reranked" else "white"
        table.add_column(mode, style=style, justify="right")

    for label, key in _METRIC_ROWS:
        values = []
        for mode in modes:
            m = comparison.configurations[mode]
            values.append(f"{getattr(m, key):.4f}")
        table.add_row(label, *values)

    table.add_row("Total Cases", *(str(comparison.total_cases) for _ in modes))
    console.print(table)


def print_breakdown_table(
    breakdown: dict[str, RetrievalMetrics],
    group_label: str,
    console: Console,
) -> None:
    """打印按类别/难度/查询类型分组的统计表。"""
    if not breakdown:
        return

    core_metrics = [
        ("File@5", "file_at_5"),
        ("Precision@5", "precision_at_5"),
        ("Recall@5", "recall_at_5"),
        ("MRR", "mrr"),
        ("MAP", "map_score"),
        ("NDCG@5", "ndcg_at_5"),
    ]

    table = Table(title=f"Breakdown by {group_label}")
    table.add_column(group_label, style="bold cyan")
    table.add_column("Cases", justify="right")
    for label, _ in core_metrics:
        table.add_column(label, justify="right")

    for name, m in sorted(breakdown.items()):
        row = [name, str(m.total_cases)]
        for _, key in core_metrics:
            row.append(f"{getattr(m, key):.4f}")
        table.add_row(*row)

    console.print(table)


def print_diff(
    before: RetrievalMetrics,
    after: RetrievalMetrics,
    console: Console,
    label_before: str = "Before",
    label_after: str = "After",
) -> None:
    """打印 before/after 对比表，delta 着色。"""
    table = Table(title="Before / After Comparison")
    table.add_column("Metric", style="bold cyan", min_width=20)
    table.add_column(label_before, justify="right")
    table.add_column(label_after, justify="right")
    table.add_column("Delta", justify="right")

    for label, key in _METRIC_ROWS:
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
    """将评估结果保存为 JSON 文件。"""
    if isinstance(data, RetrievalMetrics):
        payload = data.to_dict()
    elif isinstance(data, ComparisonResult):
        payload = data.to_dict()
    else:
        payload = data

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
