"""路径解析辅助工具。

提供数据目录解析和目录自动创建的辅助函数。
"""

from __future__ import annotations

from pathlib import Path


def resolve_data_dir(config_path: str) -> Path:
    """根据配置文件位置解析数据目录路径。

    Args:
        config_path: 配置文件路径

    Returns:
        配置文件所在目录下的 data 子目录
    """
    return Path(config_path).parent / "data"


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则递归创建。

    Args:
        path: 目录路径

    Returns:
        创建或已存在的目录 Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
