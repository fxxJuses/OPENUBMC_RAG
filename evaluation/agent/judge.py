"""LLM-as-Judge 评分器。

使用与回答模型不同的 LLM 对 Agent 回答进行独立评分，
实现模型隔离以避免自我评估偏差。
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from evaluation.agent.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT, WEIGHTS

logger = logging.getLogger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class JudgeResult:
    """Judge 评分结果。

    Attributes:
        solution_quality: 解决方案质量 (0-10)
        localization: 定位准确性 (0-10)
        completeness: 完整性 (0-10)
        evidence_reliability: 证据可靠性 (0-10)
        weighted_score: 加权总分 (0-10)
        passed: 是否通过 (≥ 6.0)
        reasoning: 评价理由
    """

    def __init__(
        self,
        solution_quality: int,
        localization: int,
        completeness: int,
        evidence_reliability: int,
        reasoning: str = "",
    ):
        self.solution_quality = solution_quality
        self.localization = localization
        self.completeness = completeness
        self.evidence_reliability = evidence_reliability
        self.reasoning = reasoning

        scores = {
            "solution_quality": solution_quality,
            "localization": localization,
            "completeness": completeness,
            "evidence_reliability": evidence_reliability,
        }
        self.weighted_score = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        self.passed = self.weighted_score >= 6.0

    def to_dict(self) -> dict:
        return {
            "solution_quality": self.solution_quality,
            "localization": self.localization,
            "completeness": self.completeness,
            "evidence_reliability": self.evidence_reliability,
            "weighted_score": round(self.weighted_score, 2),
            "passed": self.passed,
            "reasoning": self.reasoning,
        }


def create_judge_llm(
    model: str = "glm-4-flash",
    api_key: str | None = None,
) -> ChatOpenAI:
    """创建 Judge LLM 实例（使用不同于回答模型的 LLM）。

    Args:
        model: Judge 模型名称
        api_key: DashScope API 密钥

    Returns:
        配置好的 ChatOpenAI 实例
    """
    import os

    return ChatOpenAI(
        model=model,
        api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=_DASHSCOPE_BASE_URL,
        temperature=0.0,  # 评分需要确定性输出
    )


def judge_answer(
    judge_llm: ChatOpenAI,
    query: str,
    answer: str,
    expected_files: list[dict[str, str]],
    retrieved_context: str = "",
) -> JudgeResult:
    """使用 LLM-as-Judge 对 Agent 回答进行评分。

    Args:
        judge_llm: Judge LLM 实例
        query: 用户查询
        answer: Agent 的回答
        expected_files: 期望相关文件列表 [{"repo_name": ..., "file_path": ...}]
        retrieved_context: 检索到的代码上下文（用于验证证据）

    Returns:
        JudgeResult 评分结果
    """
    expected_str = "\n".join(f"- {f['repo_name']}/{f['file_path']}" for f in expected_files)
    if not expected_str:
        expected_str = "(无明确期望文件)"

    context_preview = retrieved_context[:3000] if retrieved_context else "(无检索上下文)"

    prompt = JUDGE_USER_PROMPT.format(
        query=query,
        expected_files=expected_str,
        answer=answer,
        retrieved_context=context_preview,
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = judge_llm.invoke(messages)
    return _parse_judge_response(response.content)


def _parse_judge_response(content: str) -> JudgeResult:
    """解析 Judge LLM 的 JSON 响应。

    容错处理：尝试从响应中提取 JSON 块。
    """
    # 尝试直接解析
    text = content.strip()

    # 去掉可能的 markdown 代码块标记
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse judge response: %s", content[:200])
                return _default_judge_result()
        else:
            logger.warning("No JSON found in judge response: %s", content[:200])
            return _default_judge_result()

    return JudgeResult(
        solution_quality=int(data.get("solution_quality", 0)),
        localization=int(data.get("localization", 0)),
        completeness=int(data.get("completeness", 0)),
        evidence_reliability=int(data.get("evidence_reliability", 0)),
        reasoning=data.get("reasoning", ""),
    )


def _default_judge_result() -> JudgeResult:
    """返回默认的低分 JudgeResult（解析失败时的降级策略）。"""
    return JudgeResult(
        solution_quality=0,
        localization=0,
        completeness=0,
        evidence_reliability=0,
        reasoning="Judge response parsing failed",
    )
