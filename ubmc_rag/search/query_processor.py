"""
查询处理器 —— 分类查询意图、中英术语扩展、提取过滤条件。

分析用户输入的查询文本，判断是否为代码类查询，
提取语言和分块类型过滤条件，进行中英双语技术术语扩展，
清洗查询以提升嵌入质量。

H2 迭代四增强:
1. 结构性查询扩展：检测已知组件名时附加 service.json / model.json 路径关键词
2. 查询类型路由：根据 query_type 调整 BM25/Dense 权重
3. 简单拼写纠错：对常见术语做 Levenshtein 距离 ≤1 的纠错
"""

from __future__ import annotations

import re

# 代码特征正则：检测运算符、括号、关键字等代码模式
_CODE_INDICATORS = re.compile(
    r'[{}()\[\];=<>!&|+\-*/\\\\]|::|\.\w+\(|'
    r'\bfunction\b|\bclass\b|\blocal\b|\breturn\b|\bimport\b|\brequire\b'
)

# ─────────────────────────────────────────────────────────────────
# H2: 已知 openUBMC 组件名（用于结构性查询扩展）
# ─────────────────────────────────────────────────────────────────
_KNOWN_COMPONENTS: set[str] = {
    "sensor", "sensor_mgmt", "devmon", "vpd", "frudata", "fructrl",
    "bus_tools", "libipmi", "power_mgmt", "bios", "pcie_device",
    "mdb_interface", "infrastructure",
}

# H2: 结构性扩展关键词 —— 组件名检测到后附加到查询中
_STRUCTURAL_KEYWORDS: list[str] = ["service.json", "model.json"]

# ─────────────────────────────────────────────────────────────────
# H2: 常见术语列表（用于拼写纠错，Levenshtein ≤1）
# ─────────────────────────────────────────────────────────────────
_COMMON_TERMS: list[str] = [
    "sensor", "power", "bios", "fru", "ipmi", "mds", "csr",
    "pcie", "vpd", "sel", "threshold", "interface", "service",
    "model", "config", "command", "monitor", "device", "event",
    "management", "manager", "control", "status", "protocol",
    "firmware", "version",
]


def _levenshtein(s1: str, s2: str) -> int:
    """计算两个字符串之间的 Levenshtein 编辑距离。"""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            sub = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, sub))
        prev_row = curr_row
    return prev_row[-1]


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
        query_type: 查询类型（exact_match / semantic_match / mixed）
        keywords: 提取的关键词列表
        filters: 过滤条件字典（如 language, chunk_type）
        expansion_terms: 扩展出的额外术语列表
    """

    def __init__(
        self,
        original: str,
        expanded: str | None = None,
        is_code_query: bool = False,
        query_type: str | None = None,
        keywords: list[str] | None = None,
        filters: dict | None = None,
        expansion_terms: list[str] | None = None,
    ):
        self.original = original
        self.expanded = expanded or original
        self.is_code_query = is_code_query
        self.query_type = query_type or "mixed"
        self.keywords = keywords or []
        self.filters = filters or {}
        self.expansion_terms = expansion_terms or []

    @property
    def bm25_weight_adjust(self) -> float:
        """基于 query_type 的 BM25 权重调整量。"""
        if self.query_type == "exact_match":
            return +0.1
        if self.query_type == "semantic_match":
            return -0.1
        return 0.0

    @property
    def dense_weight_adjust(self) -> float:
        """基于 query_type 的 Dense 权重调整量。"""
        if self.query_type == "exact_match":
            return -0.1
        if self.query_type == "semantic_match":
            return +0.1
        return 0.0


class QueryProcessor:
    """查询处理器，分析查询意图、扩展术语并提取过滤条件。

    支持：
    - 代码查询检测：通过语法特征判断查询是否为代码片段
    - 语言过滤提取：从查询中识别语言关键词（如 "lua", "c++"）
    - 分块类型提取：识别 "函数"、"模型" 等类型关键词
    - 中英技术术语双向扩展：将中文术语映射为英文关键词用于 BM25
    - 查询清洗：移除纯过滤关键词以提升嵌入质量
    - H2-结构性扩展：检测组件名后附加 service.json/model.json
    - H2-查询类型路由：根据 query_type 调整检索权重
    - H2-拼写纠错：对常见术语做 Levenshtein ≤1 纠错
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

    def process(self, query: str, query_type: str | None = None) -> ProcessedQuery:
        """处理原始查询，返回包含分析和扩展的 ProcessedQuery 对象。

        Args:
            query: 用户输入的原始查询文本
            query_type: 查询类型，可选值：exact_match / semantic_match / mixed。
                       如果未提供，自动检测。

        Returns:
            包含意图分析、过滤条件、扩展术语和清洗后的查询
        """
        # H2: 拼写纠错（在分类和扩展之前执行）
        corrected = self._correct_spelling(query)

        is_code = self._detect_code_query(corrected)
        filters = self._extract_filters(corrected)
        keywords = self._extract_keywords(corrected)

        # H2: 查询类型检测或使用传入的值
        if query_type is None:
            query_type = self._detect_query_type(corrected)

        expanded_terms = self._expand_query(corrected, query_type=query_type)
        cleaned = self._clean_query(corrected)
        expanded = self._build_expanded_query(corrected, expanded_terms)

        return ProcessedQuery(
            original=cleaned,
            expanded=expanded,
            is_code_query=is_code,
            query_type=query_type,
            keywords=keywords,
            filters=filters,
            expansion_terms=expanded_terms,
        )

    # ─────────────────────────────────────────────────────────────
    # H2: 查询类型自动检测
    # ─────────────────────────────────────────────────────────────

    def _detect_query_type(self, query: str) -> str:
        """自动检测查询类型。

        规则：
        - exact_match: 纯代码符号/标识符（无自然语言特征）
        - semantic_match: 包含自然语言（中文或英文句子）
        - mixed: 同时包含代码元素和自然语言

        Returns:
            检测到的查询类型字符串
        """
        has_chinese = self._has_chinese(query)
        has_code = self._detect_code_query(query)

        # 提取英文单词 token（不含下划线连接的多词标识符）
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        has_natural_english = len(english_words) > 2 and not has_code

        # 纯代码符号：有代码特征 且 无中文 且 英文词数 ≤2
        if has_code and not has_chinese and len(english_words) <= 2:
            return "exact_match"

        # 纯标识符匹配：全是 snake_case 标识符，无空格分割的自然语言
        identifier_only = re.fullmatch(r'[a-zA-Z_]\w*(\s+[a-zA-Z_]\w*)*', query.strip())
        if identifier_only and not has_chinese and not has_natural_english:
            return "exact_match"

        # 纯中文或纯语义
        if has_chinese and not has_code:
            return "semantic_match"

        # 纯英文语义（多词自然语言，无代码特征）
        if has_natural_english and not has_code and not has_chinese:
            return "semantic_match"

        # 同时有代码和自然语言
        if has_code and (has_chinese or has_natural_english):
            return "mixed"

        # 同时有中文和英文标识符
        if has_chinese and english_words:
            return "mixed"

        # 默认
        return "mixed"

    # ─────────────────────────────────────────────────────────────
    # H2: 简单拼写纠错
    # ─────────────────────────────────────────────────────────────

    def _correct_spelling(self, query: str) -> str:
        """对常见术语做 Levenshtein 距离 ≤1 的纠错。

        仅对查询中的英文标识符 token 进行纠错，不对中文部分操作。
        不引入外部拼写库，仅使用内置的 _COMMON_TERMS 列表。

        Args:
            query: 原始查询文本

        Returns:
            纠错后的查询文本
        """
        # 提取所有英文 token
        tokens = re.findall(r'[a-zA-Z_]\w*', query)
        corrected_query = query

        for token in tokens:
            token_lower = token.lower()  # Keep original case for replacement
            # 跳过已在已知术语列表中的词
            if token_lower in _COMMON_TERMS:
                continue
            # 跳过太短的词（≤2 字符太容易误纠）
            if len(token_lower) <= 2:
                continue

            # 找到 Levenshtein 距离 ≤1 的匹配
            best_match: str | None = None
            for term in _COMMON_TERMS:
                if _levenshtein(token_lower, term) <= 1:
                    best_match = term
                    break  # 第一个匹配即采纳

            if best_match and best_match != token_lower:
                # 保持原词的大小写特征
                if token.isupper():
                    replacement = best_match.upper()
                elif token[0].isupper():
                    replacement = best_match.capitalize()
                else:
                    replacement = best_match

                # 用 word boundary 来替换（避免部分匹配）
                corrected_query = re.sub(
                    r'\b' + re.escape(token) + r'\b',
                    replacement,
                    corrected_query,
                    count=1,
                )

        return corrected_query

    # ─────────────────────────────────────────────────────────────
    # 原有方法
    # ─────────────────────────────────────────────────────────────

    def _detect_code_query(self, query: str) -> bool:
        """检测查询是否包含代码特征（运算符、括号、编程关键字等）。"""
        return bool(_CODE_INDICATORS.search(query))

    def _has_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符。"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _expand_query(self, query: str, query_type: str = "mixed") -> list[str]:
        """执行中英技术术语双向扩展 + 结构性查询扩展。

        策略：
        1. 中文查询 → 将中文技术术语替换为对应英文关键词
        2. 英文缩写 → 扩展为全称
        3. 不添加冗余（已有词不重复添加）
        4. H2: 检测已知组件名 → 附加 service.json / model.json 路径关键词
           （仅对 semantic_match 和 mixed 类型生效；exact_match 不附加）

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

        # 步骤 4 (H2): 结构性查询扩展
        # 仅对 semantic_match 和 mixed 类型生效，exact_match 不附加
        # 以避免污染精确匹配查询的关键词空间
        if query_type in ("semantic_match", "mixed"):
            structural_terms = self._get_structural_expansions(query)
            for term in structural_terms:
                expanded.add(term)

        return list(expanded)

    # ─────────────────────────────────────────────────────────────
    # H2: 结构性查询扩展
    # ─────────────────────────────────────────────────────────────

    def _get_structural_expansions(self, query: str) -> list[str]:
        """检测已知组件名，返回结构性扩展关键词。

        当查询中提到已知的 openUBMC 组件名时，
        自动附加 service.json 和 model.json 路径关键词，
        帮助 BM25 匹配到组件的 MDS 配置文件和模型定义。

        智能条件：仅当查询中包含服务/模型/配置/依赖/关系/接口等
        结构性关键词时才附加，避免污染纯功能查询。

        Args:
            query: 用户查询文本（可能包含中文和英文）

        Returns:
            结构性扩展关键词列表
        """
        # 结构性触发关键词 —— 查询中提到这些才附加 service.json / model.json
        _structural_triggers = [
            "service", "服务", "model", "模型", "config", "配置",
            "dependency", "依赖", "relation", "关系", "interface", "接口",
            "definition", "定义", "mds", "MDS", "ipmi", "IPMI",
        ]

        query_lower = query.lower()

        # 先检查是否有组件名
        has_component = any(c.lower() in query_lower for c in _KNOWN_COMPONENTS)
        if not has_component:
            return []

        # 检查是否有结构性触发词
        has_trigger = any(t.lower() in query_lower for t in _structural_triggers)
        if not has_trigger:
            return []

        # 两个条件都满足，才附加结构性关键词
        expansions: list[str] = []
        for kw in _STRUCTURAL_KEYWORDS:
            if kw.lower() not in query_lower:
                expansions.append(kw)

        return expansions

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
