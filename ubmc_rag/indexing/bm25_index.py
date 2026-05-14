"""BM25 keyword index with code-aware tokenizer."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from ubmc_rag.models.code_chunk import CodeChunk

logger = logging.getLogger(__name__)

# Code-aware tokenizer: splits on word boundaries, camelCase, underscores, operators
_TOKENIZE_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|\d+|[a-zA-Z]\w*|[^\s\w]")


def code_tokenize(text: str) -> list[str]:
    """Tokenize code text for BM25 indexing."""
    tokens = _TOKENIZE_RE.findall(text)
    return [t.lower() for t in tokens if len(t) > 1]


class BM25Index:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[CodeChunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [code_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(chunks))

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Search and return (chunk_id, score) pairs."""
        if self._bm25 is None:
            return []

        tokenized_query = code_tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self._chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def get_chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    def save(self, path: Path) -> None:
        """Serialize index data to disk."""
        data = {
            "chunk_ids": self._chunk_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path) -> bool:
        """Load index data from disk."""
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index loaded: %d documents", len(self._chunk_ids))
        return True
