"""搜索结果模型。

封装代码检索的结果，包含匹配的代码分块、相关度分数和来源信息。
"""

from __future__ import annotations

from dataclasses import dataclass

from ubmc_rag.models.code_chunk import CodeChunk


@dataclass
class SearchResult:
    """代码搜索结果，包含匹配分块及其相关度评分。

    Attributes:
        chunk: 匹配到的代码分块
        score: 相关度分数，由 RRF 融合和重排序计算得出
        source: 分数来源，可选值："bm25"（关键词匹配），
                "dense"（向量相似度），"hybrid"（混合检索）
    """

    chunk: CodeChunk
    score: float
    source: str = "hybrid"

    def to_dict(self) -> dict:
        """将搜索结果转换为字典，用于 JSON 序列化和展示。"""
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
