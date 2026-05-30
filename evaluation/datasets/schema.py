"""评估数据集的 Pydantic V2 数据模型。

定义回归测试用例的结构，包含查询、期望文件、期望符号等信息，
用于检索质量评估和 Agent 回答质量评估。

字段命名与 CodeChunk / SearchResult 模型对齐，
方便在指标计算中直接匹配。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExpectedFile(BaseModel):
    """期望在搜索结果中出现的文件。

    Attributes:
        repo_name: 仓库名，匹配 CodeChunk.repo_name（如 "sensor", "libipmi"）
        file_path: 文件路径，匹配 CodeChunk.file_path（如 "src/lualib/sensor_service.lua"）
        relevance: 相关度等级，1=相关, 2=高度相关, 3=核心。用于 NDCG 计算。
    """

    repo_name: str
    file_path: str
    relevance: int = 1


class TestCase(BaseModel):
    """单条回归测试用例。

    Attributes:
        id: 唯一标识符，如 "TC-001"
        query: 搜索查询文本
        category: 查询类别，可选 "single_function" / "single_component" / "cross_component"
        query_type: 查询类型，可选 "exact_match" / "semantic_match" / "chinese" / "mixed"
        expected_files: 期望命中的文件列表（Ground Truth）
        expected_symbols: 期望出现的符号名列表（可选）
        expected_repos: 期望命中的仓库名列表（用于 CategoryHit 指标）
        difficulty: 难度等级 "easy" / "normal" / "hard"
        description: 人类可读的用例描述
    """

    id: str
    query: str
    category: str
    query_type: str = "semantic_match"
    expected_files: list[ExpectedFile]
    expected_symbols: list[str] = Field(default_factory=list)
    expected_repos: list[str] = Field(default_factory=list)
    difficulty: str = "normal"
    description: str = ""


class RegressionDataset(BaseModel):
    """回归测试数据集。

    Attributes:
        name: 数据集名称，如 "regression_v1"
        version: 版本号，如 "1.0"
        description: 数据集描述
        test_cases: 测试用例列表
    """

    name: str
    version: str = "1.0"
    description: str = ""
    test_cases: list[TestCase]
