"""日志工具模块。

提供统一的日志配置，使用 Rich 库美化控制台输出，
支持 markup 标记语法。
"""

from __future__ import annotations

import logging

import rich.logging


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志，使用 Rich 渲染器美化输出。

    Args:
        level: 日志级别，如 "INFO", "DEBUG", "WARNING"
    """
    handler = rich.logging.RichHandler(show_time=False, show_path=False, markup=True)
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[handler],
    )


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 Logger 实例。"""
    return logging.getLogger(name)
