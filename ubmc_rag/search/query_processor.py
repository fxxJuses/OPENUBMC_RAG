"""查询处理器 —— 分类查询意图并提取过滤条件。

分析用户输入的查询文本，判断是否为代码类查询，
提取语言和分块类型过滤条件，并进行关键词提取和查询清洗。
"""

from __future__ import annotations

import re

# 代码特征正则：检测运算符、括号、关键字等代码模式
_CODE_INDICATORS = re.compile(
    r'[{}()\[\];=<>!&|+\-*/\\]|::|\.\w+\(|'
    r'\bfunction\b|\bclass\b|\blocal\b|\breturn\b|\bimport\b|\brequire\b'
)


class ProcessedQuery:
    """处理后的查询对象，包含分析结果和过滤条件。

    Attributes:
        original: 清洗后的查询文本
        is_code_query: 是否为代码类查询
        keywords: 提取的关键词列表
        filters: 过滤条件字典（如 language, chunk_type）
    """

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
    """查询处理器，分析查询意图并提取过滤条件。

    支持：
    - 代码查询检测：通过语法特征判断查询是否为代码片段
    - 语言过滤提取：从查询中识别语言关键词（如 "lua", "c++"）
    - 分块类型提取：识别 "函数"、"模型" 等类型关键词
    - 查询清洗：移除过滤关键词以提升嵌入质量
    """

    # openUBMC 语言关键词映射
    LANG_KEYWORDS = {
        "lua": ["lua", "luajit"],
        "c": ["c语言", "c code"],
        "cpp": ["c++", "cpp"],
        "python": ["python", "py"],
        "json": ["json", "mds", "csr", "ipmi", "sr"],
    }

    # 分块类型关键词映射
    CHUNK_TYPE_KEYWORDS = {
        "function": ["函数", "function", "方法", "method"],
        "class": ["类", "class", "class定义"],
        "mds_model": ["模型", "model", "mds模型"],
        "mds_ipmi_cmd": ["ipmi命令", "ipmi command", "ipmi"],
        "csr_object": ["csr", "设备对象", "device object"],
        "section": ["文档", "document", "文档说明"],
    }

    def process(self, query: str) -> ProcessedQuery:
        """处理原始查询，返回包含分析结果的 ProcessedQuery 对象。

        Args:
            query: 用户输入的原始查询文本

        Returns:
            包含意图分析、过滤条件和关键词的处理后查询
        """
        is_code = self._detect_code_query(query)
        filters = self._extract_filters(query)
        keywords = self._extract_keywords(query)
        cleaned = self._clean_query(query)

        return ProcessedQuery(
            original=cleaned,
            is_code_query=is_code,
            keywords=keywords,
            filters=filters,
        )

    def _detect_code_query(self, query: str) -> bool:
        """检测查询是否包含代码特征（运算符、括号、编程关键字等）。"""
        return bool(_CODE_INDICATORS.search(query))

    def _extract_filters(self, query: str) -> dict:
        """从查询中提取语言和分块类型过滤条件。"""
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

    def _extract_keywords(self, query: str) -> list[str]:
        """从查询中提取有意义的词元，过滤停用词。

        支持中文字符和英文标识符的提取。
        """
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

    def _clean_query(self, query: str) -> str:
        """移除过滤关键词以提升向量嵌入的语义质量。

        如果清洗后为空，则返回原始查询。
        """
        cleaned = query
        for keywords in self.LANG_KEYWORDS.values():
            for kw in keywords:
                cleaned = cleaned.replace(kw, "")
        for keywords in self.CHUNK_TYPE_KEYWORDS.values():
            for kw in keywords:
                cleaned = cleaned.replace(kw, "")
        return cleaned.strip() or query
