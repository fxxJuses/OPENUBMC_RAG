"""GitCode 仓库同步模块，负责克隆和拉取 openUBMC 仓库。

提供仓库的发现、克隆、更新和列表功能，
支持增量同步（仅拉取已有仓库的更新）和按需克隆。
"""

from __future__ import annotations

import logging
from pathlib import Path

import git

from ubmc_rag.config.settings import AppConfig, RepoConfig
from ubmc_rag.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


class GitSync:
    """Git 仓库同步器，管理 openUBMC 代码仓库的克隆与更新。

    Attributes:
        config: 应用配置
        clone_dir: 本地克隆目录的路径
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.clone_dir = ensure_dir(config.git.clone_dir)

    def get_repo_url(self, repo: RepoConfig) -> str:
        """根据仓库配置生成完整的 Git 克隆 URL。"""
        return f"{self.config.git.base_url}/{repo.name}.git"

    def clone_or_pull(self, repo: RepoConfig) -> Path:
        """克隆仓库或拉取最新更新。

        如果仓库目录已存在则执行 git pull，否则执行 git clone。

        Args:
            repo: 仓库配置

        Returns:
            仓库的本地路径
        """
        repo_path = self.clone_dir / repo.name
        url = self.get_repo_url(repo)

        if repo_path.exists():
            logger.info("Pulling updates for [bold]%s[/bold]", repo.name)
            try:
                repo_obj = git.Repo(repo_path)
                repo_obj.remotes.origin.pull(self.config.git.branch)
            except Exception as e:
                logger.warning("Failed to pull %s: %s", repo.name, e)
        else:
            logger.info("Cloning [bold]%s[/bold]", repo.name)
            git.Repo.clone_from(url, repo_path, branch=self.config.git.branch)

        return repo_path

    def sync_all(self, repos: list[str] | None = None, clone_missing: bool = False) -> list[Path]:
        """同步所有配置的仓库，或指定的子集。

        Args:
            repos: 需要同步的仓库名称列表，为 None 时同步全部
            clone_missing: 是否克隆本地不存在的仓库

        Returns:
            成功同步的仓库路径列表
        """
        targets = self.config.git.repos
        if repos:
            targets = [r for r in targets if r.name in repos]

        paths = []
        for repo_conf in targets:
            repo_path = self.clone_dir / repo_conf.name
            if not repo_path.exists() and not clone_missing:
                logger.warning(
                    "Repo %s not found locally, skipping (use --clone-missing)",
                    repo_conf.name,
                )
                continue
            try:
                path = self.clone_or_pull(repo_conf)
                paths.append(path)
            except Exception as e:
                logger.error("Failed to sync %s: %s", repo_conf.name, e)

        return paths

    def list_cloned_repos(self) -> list[Path]:
        """列出克隆目录中所有已克隆的仓库。"""
        return [
            p for p in self.clone_dir.iterdir()
            if p.is_dir() and (p / ".git").exists()
        ]
