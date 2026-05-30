"""评估数据集 YAML 加载器。

从 YAML 文件加载回归测试数据集，使用 Pydantic V2 进行校验。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.datasets.schema import RegressionDataset


def load_dataset(path: str | Path) -> RegressionDataset:
    """从 YAML 文件加载回归测试数据集。

    Args:
        path: YAML 文件路径

    Returns:
        校验后的 RegressionDataset 实例

    Raises:
        FileNotFoundError: 文件不存在
        pydantic.ValidationError: 数据格式不符合 Schema
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return RegressionDataset(**data)
