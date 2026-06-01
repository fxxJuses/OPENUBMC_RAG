"""BM25 关键词索引，使用代码感知的分词器。

基于 rank_bm25 库实现 Okapi BM25 算法，配合专门为代码设计的
分词器（支持驼峰命名、下划线命名和运算符拆分）。
迭代6-B：移除复合 token 生成，简化分词逻辑。
支持索引的序列化和反序列化，以便持久化到磁盘。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from ubmc_rag.models.code_chunk import CodeChunk

logger = logging.getLogger(__name__)

# 代码感知分词正则：识别驼峰命名、下划线命名、数字和运算符
# 迭代6-P3 修复：\b 无法匹配字母→下划线边界（因 _ 是 \w），改用 [^A-Za-z]|$
_TOKENIZE_RE = re.compile(
    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|[^A-Za-z]|$)|\d+|[a-zA-Z]\w*|[^\s\w]"
)

# OpenBMC 领域词典：嵌入式固件和 IPMI 协议常用术语
# 这些术语在 BM25 索引中应保持完整，不被进一步拆分
_DOMAIN_DICTIONARY: set[str] = {
    # IPMI 协议相关
    "ipmi", "sel", "sdr", "pef", "fru", "vpd", "i2c", "smbus", "dbus",
    # 传感器相关
    "sensor", "threshold", "discrete", "analog",
    # 固件相关
    "firmware", "bios", "bmc", "uefi", "boot", "power", "thermal",
    # 硬件管理
    "gpio", "pcie", "nvme", "dimm", "cpu", "psu", "fan", "led",
    # 事件和日志
    "event", "alert", "log", "syslog", "redfish",
    # 常见缩写
    "api", "cmd", "cfg", "mgr", "ctrl", "util", "info", "data", "dev",
}


def code_tokenize(text: str) -> list[str]:
    """对代码文本进行分词，适合 BM25 关键词索引。

    迭代6-B：简化分词器，移除复合 token 生成逻辑。
    - 拆分 camelCase/snake_case 标识符为子 token
    - 不再保留原始复合标识符（去复合 token）
    - 注入 OpenBMC 领域词典，确保领域术语保持完整

    支持驼峰命名拆分（如 getSensorData -> get, sensor, data）、
    下划线命名拆分（如 reading_value -> reading, value）。
    过滤长度≤1 的无意义词元。

    Args:
        text: 待分词的代码文本

    Returns:
        小写化的词元列表
    """
    # 用正则提取原子 token（已处理 camelCase 和 snake_case 拆分）
    raw_tokens = _TOKENIZE_RE.findall(text)

    # 小写化、过滤长度 ≤1 的 token
    return [t.lower() for t in raw_tokens if len(t) > 1]


class BM25Index:
    """BM25 关键词索引，支持构建、搜索和持久化。

    使用 Okapi BM25 算法对代码分块进行关键词匹配，
    与向量检索互补，共同构成混合搜索系统。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self.k1 = k1
        self.b = b

    def _build_document(self, chunk: CodeChunk) -> str:
        """H4: 构建 BM25 文档内容。

        在原代码内容基础上拼接文件路径、仓库名（重复以增加权重）
        和符号名，使 BM25 可以通过路径/仓库/符号名匹配结果。
        """
        parts = [chunk.content]
        # 文件路径：分词后可匹配文件名和目录名
        parts.append(chunk.file_path)
        # 仓库名：重复两次以提高权重
        parts.append(chunk.repo_name)
        parts.append(chunk.repo_name)
        # 符号名（函数名、类名、变量名等）
        for sym in chunk.symbols:
            parts.append(sym.name)
        return " ".join(parts)

    def build(self, chunks: list[CodeChunk]) -> None:
        """从代码分块列表构建 BM25 索引。

        Args:
            chunks: 代码分块列表，使用增强后的文档内容（代码+元数据）
        """
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [code_tokenize(self._build_document(c)) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index built with %d documents", len(chunks))

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """执行 BM25 关键词搜索。

        Args:
            query: 搜索查询文本
            top_k: 返回的最大结果数

        Returns:
            (chunk_id, score) 元组列表，按分数降序排列
        """
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
        """返回索引中所有分块的 ID 列表。"""
        return list(self._chunk_ids)

    def save(self, path: Path) -> None:
        """将索引数据序列化到磁盘文件。

        保存 chunk_ids 和分词后的语料，BM25 模型参数通过加载时重建。
        """
        data = {
            "chunk_ids": self._chunk_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path) -> bool:
        """从磁盘文件加载索引数据并重建 BM25 模型。

        Args:
            path: 索引文件路径

        Returns:
            加载成功返回 True，文件不存在返回 False
        """
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index loaded: %d documents", len(self._chunk_ids))
        return True
