"""Query processor — classifies queries and extracts filters."""

from __future__ import annotations

import re
from typing import Optional

# Patterns to detect code-like queries
_CODE_INDICATORS = re.compile(r'[{}()\[\];=<>!&|+\-*/\\]|::|\.\w+\(|\bfunction\b|\bclass\b|\blocal\b|\breturn\b|\bimport\b|\brequire\b')


class ProcessedQuery:
    def __init__(
        self,
        original: str,
        is_code_query: bool = False,
        keywords: list[str] | None = None,
        filters: dict | None = None,
    ):
        self.original = original
        self.is_code_query = is_code_query
        self.keywords = keywords or []
        self.filters = filters or {}


class QueryProcessor:
    # Common openUBMC language keywords
    LANG_KEYWORDS = {
        "lua": ["lua", "luajit"],
        "c": ["c语言", "c code"],
        "cpp": ["c++", "cpp"],
        "python": ["python", "py"],
        "json": ["json", "mds", "csr", "ipmi", "sr"],
    }

    CHUNK_TYPE_KEYWORDS = {
        "function": ["函数", "function", "方法", "method"],
        "class": ["类", "class", "class定义"],
        "mds_model": ["模型", "model", "mds模型"],
        "mds_ipmi_cmd": ["ipmi命令", "ipmi command", "ipmi"],
        "csr_object": ["csr", "设备对象", "device object"],
        "section": ["文档", "document", "文档说明"],
    }

    def process(self, query: str) -> ProcessedQuery:
        is_code = self._detect_code_query(query)
        filters = self._extract_filters(query)
        keywords = self._extract_keywords(query, filters)
        cleaned = self._clean_query(query, filters)

        return ProcessedQuery(
            original=cleaned,
            is_code_query=is_code,
            keywords=keywords,
            filters=filters,
        )

    def _detect_code_query(self, query: str) -> bool:
        """Detect if query is code-like (contains operators, brackets, etc.)."""
        return bool(_CODE_INDICATORS.search(query))

    def _extract_filters(self, query: str) -> dict:
        filters = {}
        query_lower = query.lower()

        for lang, keywords in self.LANG_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    filters["language"] = lang
                    break

        for chunk_type, keywords in self.CHUNK_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    filters["chunk_type"] = chunk_type
                    break

        return filters

    def _extract_keywords(self, query: str, filters: dict) -> list[str]:
        """Extract significant terms from query."""
        # Remove common stop words
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
            "会", "着", "没有", "看", "好", "自己", "这",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "how", "what", "where", "when", "which", "who", "why",
            "do", "does", "did", "can", "could", "should", "would",
        }
        words = re.findall(r"[a-zA-Z_]\w*|[一-鿿]+", query)
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    def _clean_query(self, query: str, filters: dict) -> str:
        """Remove filter-indicating words from query for embedding."""
        cleaned = query
        for keywords in self.LANG_KEYWORDS.values():
            for kw in keywords:
                cleaned = cleaned.replace(kw, "")
        for keywords in self.CHUNK_TYPE_KEYWORDS.values():
            for kw in keywords:
                cleaned = cleaned.replace(kw, "")
        return cleaned.strip() or query
