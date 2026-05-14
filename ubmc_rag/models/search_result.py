"""Search result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ubmc_rag.models.code_chunk import CodeChunk


@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    source: str = "hybrid"  # "bm25", "dense", "hybrid"

    def to_dict(self) -> dict:
        return {
            "content": self.chunk.content,
            "file_path": self.chunk.file_path,
            "repo": self.chunk.repo_name,
            "language": self.chunk.language,
            "chunk_type": self.chunk.chunk_type,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "score": round(self.score, 4),
            "source": self.source,
            "symbols": [
                {"name": s.name, "kind": s.kind, "signature": s.signature}
                for s in self.chunk.symbols
            ],
        }
