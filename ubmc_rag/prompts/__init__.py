"""Modular prompt library for composable agent system prompts.

Prompts are stored as .md files and composed at runtime.
"""

from pathlib import Path


class PromptLibrary:
    """Load and compose prompt modules from the prompts/ directory."""

    def __init__(self, prompts_dir: Path | None = None):
        self._dir = prompts_dir or Path(__file__).parent
        self._cache: dict[str, str] = {}

    def load(self, relative_path: str) -> str:
        """Load a single prompt file. Results are cached."""
        if relative_path not in self._cache:
            path = self._dir / relative_path
            self._cache[relative_path] = path.read_text(encoding="utf-8").strip()
        return self._cache[relative_path]

    def compose(self, *relative_paths: str, separator: str = "\n\n") -> str:
        """Load and concatenate multiple prompt files."""
        return separator.join(self.load(p) for p in relative_paths)

    def get_system_prompt(self) -> str:
        """Compose the full agent system prompt from chat/ modules."""
        return self.compose(
            "chat/identity.md",
            "chat/strategy.md",
            "chat/tools_guide.md",
            "chat/evidence_rules.md",
            "chat/domain_context.md",
            "chat/output_format.md",
        )
