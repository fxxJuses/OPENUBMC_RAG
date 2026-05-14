"""GitCode repository discovery and cloning."""

from __future__ import annotations

import logging
from pathlib import Path

import git

from ubmc_rag.config.settings import AppConfig, RepoConfig
from ubmc_rag.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


class GitSync:
    def __init__(self, config: AppConfig):
        self.config = config
        self.clone_dir = ensure_dir(config.git.clone_dir)

    def get_repo_url(self, repo: RepoConfig) -> str:
        return f"{self.config.git.base_url}/{repo.name}.git"

    def clone_or_pull(self, repo: RepoConfig) -> Path:
        """Clone a repo if not present, or pull latest changes."""
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
        """Sync all configured repos, or a subset."""
        targets = self.config.git.repos
        if repos:
            targets = [r for r in targets if r.name in repos]

        paths = []
        for repo_conf in targets:
            repo_path = self.clone_dir / repo_conf.name
            if not repo_path.exists() and not clone_missing:
                logger.warning("Repo %s not found locally, skipping (use --clone-missing)", repo_conf.name)
                continue
            try:
                path = self.clone_or_pull(repo_conf)
                paths.append(path)
            except Exception as e:
                logger.error("Failed to sync %s: %s", repo_conf.name, e)

        return paths

    def list_cloned_repos(self) -> list[Path]:
        """List all cloned repos in the clone directory."""
        return [p for p in self.clone_dir.iterdir() if p.is_dir() and (p / ".git").exists()]
