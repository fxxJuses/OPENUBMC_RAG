"""
查询处理器 —— 分类查询意图、中英术语扩展、提取过滤条件。

分析用户输入的查询文本，判断是否为代码类查询，
提取语言和分块类型过滤条件，进行中英双语技术术语扩展，
清洗查询以提升嵌入质量。
"""

from __future__ import annotations

import difflib
import re

# 代码特征正则：检测运算符、括号、关键字等代码模式
_CODE_INDICATORS = re.compile(
    r'[{}()\[\];=<>!&|+\-*/\\\\]|::|\.\w+\(|'
    r'\bfunction\b|\bclass\b|\blocal\b|\breturn\b|\bimport\b|\brequire\b'
)

# ─────────────────────────────────────────────────────────────────
# openUBMC 领域中英技术术语双向映射
# ─────────────────────────────────────────────────────────────────

# 中文 → 英文核心术语映射（按 openUBMC 组件和 BMC 领域）
_ZH_EN_TERMS: dict[str, str] = {
    # 组件名
    "传感器": "sensor",
    "电源": "power",
    "风扇": "fan",
    "温度": "temperature",
    "电压": "voltage",
    "内存": "memory",
    "硬盘": "disk",
    "板卡": "board",
    "总线": "bus",
    # 功能概念
    "阈值": "threshold",
    "日志": "sel log",
    "事件": "event",
    "热插拔": "hotplug",
    "上电": "power on",
    "下电": "power off",
    "接口": "interface",
    "数据": "data",
    "模型": "model",
    "配置": "config",
    "服务": "service",
    "管理": "management mgmt",
    "监控": "monitor monitoring",
    "升级": "upgrade",
    "扫描": "scan detect",
    "启动": "startup init",
    "初始化": "initialize init",
    "依赖": "dependency dep",
    "关系": "relation",
    "完整": "full complete",
    "路径": "path",
    "链路": "link",
    "联动": "interaction",
    "传递": "transfer",
    "流转": "flow",
    "触发": "trigger",
    "控制": "control ctrl",
    "设备": "device dev",
    "组件": "component",
    "定义": "definition def",
    "命令": "command cmd",
    "数据流": "dataflow",
    "对象": "object obj",
    "应用": "application app",
    "入口": "entry",
    "导出": "export dump",
    "信息": "info",
    "读取": "read",
    "写入": "write",
    "收发": "send receive",
    "告警": "alert",
    "实现": "implementation impl",
    "电容": "capacitor",
    "电路": "circuit",
    # FRU 相关
    "可更换": "fru replaceable",
    "固件": "firmware",
    "版本": "version",
    # IPMI 相关
    "传感": "sensing",
    "状态": "status",
    "协议": "protocol",
}

# 英文缩写 → 全称/别名映射（用于同义词扩展）
_EN_SYNONYMS: dict[str, str] = {
    "mgmt": "management",
    "mgr": "manager",
    "ctrl": "control",
    "dev": "device",
    "cfg": "config",
    "svc": "service",
    "obj": "object",
    "impl": "implementation",
    "dep": "dependency",
    "cmd": "command",
    "app": "application",
}


class ProcessedQuery:
    """处理后的查询对象，包含分析结果和扩展词。

    Attributes:
        original: 原始查询文本（清洗后，用于 Dense 嵌入）
        expanded: 扩展后的查询文本（用于 BM25 关键词检索）
        is_code_query: 是否为代码类查询
        keywords: 提取的关键词列表
        filters: 过滤条件字典（如 language, chunk_type）
        expansion_terms: 扩展出的额外术语列表
    """

    def __init__(
        self,
        original: str,
        expanded: str | None = None,
        is_code_query: bool = False,
        keywords: list[str] | None = None,
        filters: dict | None = None,
        expansion_terms: list[str] | None = None,
    ):
        self.original = original
        self.expanded = expanded or original
        self.is_code_query = is_code_query
        self.keywords = keywords or []
        self.filters = filters or {}
        self.expansion_terms = expansion_terms or []


class QueryProcessor:
    """查询处理器，分析查询意图、扩展术语并提取过滤条件。

    支持：
    - 代码查询检测：通过语法特征判断查询是否为代码片段
    - 语言过滤提取：从查询中识别语言关键词（如 "lua", "c++"）
    - 分块类型提取：识别 "函数"、"模型" 等类型关键词
    - 中英技术术语双向扩展：将中文术语映射为英文关键词用于 BM25
    - 查询清洗：移除纯过滤关键词以提升嵌入质量
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

    # 查询清洗时要移除的纯过滤关键词（有歧义的短词不移除）
    _CLEAN_STOPWORDS = {"json", "mds", "csr", "sr"}

    # 从代码库符号名提取的高频领域术语（用于拼写纠错）
    _DEFAULT_DOMAIN_TERMS: list[str] = [
        "sensor", "sensors", "threshold", "config", "monitoring", "management",
        "alert", "event", "logging", "function", "init", "initialize", "startup",
        "power", "control", "ipmi", "ipmb", "protocol", "command", "fru",
        "frudata", "fructrl", "devmon", "device", "vpd", "bios", "pcie", "mdb",
        "bus", "i2c", "libipmi", "sel", "reading", "service", "model",
        "application", "handler", "callback", "register", "subscribe", "publish",
        "interface", "dependency", "dependencies", "component", "object",
        "instance", "discrete", "metric", "voltage", "temperature", "firmware",
        "upgrade", "capacitor", "circuit", "payload", "restore", "timeout",
        "flag", "chassis", "state", "machine", "executor", "format", "policy",
        "wrapper", "channel", "driver", "adapter", "message", "filter", "manager",
        "root", "discovery",
    ]

    def __init__(self) -> None:
        self._domain_terms: list[str] = list(self._DEFAULT_DOMAIN_TERMS)

    def set_domain_terms(self, terms: list[str]) -> None:
        """设置领域术语词典（替换默认词典）。

        Args:
            terms: 领域术语列表，用于拼写纠错匹配。
        """
        self._domain_terms = list(terms)

    def process(self, query: str) -> ProcessedQuery:
        """处理原始查询，返回包含分析和扩展的 ProcessedQuery 对象。

        Args:
            query: 用户输入的原始查询文本

        Returns:
            包含意图分析、过滤条件、扩展术语和清洗后的查询
        """
        is_code = self._detect_code_query(query)
        filters = self._extract_filters(query)
        keywords = self._extract_keywords(query)
        expanded_terms = self._expand_query(query)
        cleaned = self._clean_query(query)
        expanded = self._build_expanded_query(query, expanded_terms)

        return ProcessedQuery(
            original=cleaned,
            expanded=expanded,
            is_code_query=is_code,
            keywords=keywords,
            filters=filters,
            expansion_terms=expanded_terms,
        )

    def _detect_code_query(self, query: str) -> bool:
        """检测查询是否包含代码特征（运算符、括号、编程关键字等）。"""
        return bool(_CODE_INDICATORS.search(query))

    def _has_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符。"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _expand_query(self, query: str) -> list[str]:
        """执行中英技术术语双向扩展。

        策略：
        1. 中文查询 → 将中文技术术语替换为对应英文关键词
        2. 英文缩写 → 扩展为全称
        3. 不添加冗余（已有词不重复添加）

        Returns:
            扩展出的额外术语列表
        """
        expanded = set()

        # 步骤 1: 中文 → 英文扩展
        if self._has_chinese(query):
            for zh_term, en_terms in _ZH_EN_TERMS.items():
                if zh_term in query:
                    for en_word in en_terms.split():
                        if en_word.lower() not in query.lower():
                            expanded.add(en_word.lower())

        # 步骤 2: 从查询中提取英文标识符，检查是否需要扩展缩写
        english_tokens = re.findall(r'[a-zA-Z_]\w*', query)
        for token in english_tokens:
            token_lower = token.lower()
            if token_lower in _EN_SYNONYMS:
                synonym = _EN_SYNONYMS[token_lower]
                if synonym.lower() not in query.lower():
                    expanded.add(synonym.lower())

        # 步骤 3: 反向——如果查询中有英文术语，检查是否应添加中文
        # （用于 BM25 索引中带中文注释的代码段）
        for en_term_phrase, zh_term in [
            ("sensor", "传感器"), ("power", "电源"), ("threshold", "阈值"),
            ("sel", "日志"), ("event", "事件"), ("hotplug", "热插拔"),
            ("frudata", "FRU数据"), ("fructrl", "FRU控制"),
            ("ipmi", "IPMI"), ("pcie", "PCIe"), ("bios", "BIOS"),
        ]:
            if en_term_phrase.lower() in query.lower():
                if zh_term not in query:
                    expanded.add(zh_term)

        # 步骤 4: 拼写纠错——对不在词典中的英文 token 用 difflib 纠错
        for token in english_tokens:
            token_lower = token.lower()
            # 跳过已在词典中的 token（拼写正确）或过短的 token
            if token_lower in self._domain_terms or len(token_lower) < 3:
                continue
            matches = difflib.get_close_matches(
                token_lower, self._domain_terms, n=1, cutoff=0.8
            )
            if matches:
                correction = matches[0]
                # 纠错结果不应已在查询中出现
                if correction not in query.lower() and correction != token_lower:
                    expanded.add(correction)

        return list(expanded)

    def _build_expanded_query(self, original: str, expansion_terms: list[str]) -> str:
        """构建扩展后的查询字符串（用于 BM25）。"""
        if not expansion_terms:
            return original
        # 原始查询 + 扩展术语，用空格连接
        return original + " " + " ".join(expansion_terms)

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
        words = re.findall(r"[a-zA-Z_]\w*|[\u4e00-\u9fff]+", query)
        return [w for w in words if w.lower() not in stop_words and len(w) > 1]

    def _clean_query(self, query: str) -> str:
        """移除纯过滤关键词以提升向量嵌入的语义质量。

        只移除有歧义的短词（如 json、mds），保留有语义价值的词
        （如 ipmi、function、model），这些词对 embedding 理解查询意图很重要。
        """
        cleaned = query
        for kw in self._CLEAN_STOPWORDS:
            cleaned = cleaned.replace(kw, "")
        return cleaned.strip() or query
