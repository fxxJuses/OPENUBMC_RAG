"""Result reranker — applies boost rules and diversity filtering."""

from __future__ import annotations

from ubmc_rag.config.settings import SearchConfig
from ubmc_rag.models.search_result import SearchResult


class Reranker:
    def __init__(self, config: SearchConfig):
        self.config = config

    def rerank(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """Apply reranking rules to search results."""
        if not results:
            return results

        boosted = []
        for r in results:
            score = r.score
            query_lower = query.lower()

            # Exact symbol name match boost
            for sym in r.chunk.symbols:
                if sym.name.lower() in query_lower:
                    score *= self.config.symbol_match_boost
                    break

            # File path match boost
            if any(part in r.chunk.file_path.lower() for part in query_lower.split() if len(part) > 2):
                score *= self.config.filepath_match_boost

            # MDS model match boost
            mds_class = r.chunk.metadata.get("mds_class", "")
            if mds_class and mds_class.lower() in query_lower:
                score *= self.config.mds_model_match_boost

            boosted.append(SearchResult(
                chunk=r.chunk,
                score=score,
                source=r.source,
            ))

        # Sort by boosted score
        boosted.sort(key=lambda x: x.score, reverse=True)

        # Diversity filter: limit results per file
        filtered = []
        file_counts: dict[str, int] = {}
        for r in boosted:
            key = r.chunk.file_path
            count = file_counts.get(key, 0)
            if count >= self.config.diversity_max_per_file:
                r = SearchResult(
                    chunk=r.chunk,
                    score=r.score * 0.7,
                    source=r.source,
                )
            filtered.append(r)
            file_counts[key] = count + 1

        return filtered
