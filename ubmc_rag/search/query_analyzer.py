"""LLM-based query analyzer — classifies intent and generates sub-queries for RAG retrieval."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_ANALYZE_PROMPT = """\
分析用户的代码检索问题，生成适合代码搜索引擎的子查询，输出结构化 JSON。

## 意图分类规则
- relationship: 询问两个或多个组件/模块之间的关系、交互、联合机制
- code_lookup: 查找某个函数、类、变量的定义或用法
- architecture: 询问整体架构、设计模式、组件职责
- debug: 排查问题、分析错误、理解行为异常
- general: 其他一般性问题

## 子查询生成规则（重要：子查询用于代码检索，不是回答问题）

### 关键原则
- 子查询应该是**简短的关键词组合**，不是完整句子或问题描述
- 包含可能的函数名、变量名、模块名（用 snake_case 命名推测）
- 同时包含中文语义词和英文代码词，提升 BM25 + 向量双路命中率
- 每个子查询 3-8 个词，不要写成长句

### 按意图类型生成
- code_lookup / general: 直接使用原始问题，1 个子查询
- relationship: 生成 3-5 个子查询，示例：
  用户问"sensor和power的关系" →
  ["sensor power_mgmt require import",
   "get_sensor_reading power threshold",
   "ipmi sensor power supply reading",
   "power_mgmt sensor_const sensor_management",
   "sensor power_monitor"]
- architecture: 生成 2-3 个关键词子查询
- debug: 生成 2-3 个关键词子查询

## openUBMC 已知微组件
sensor, sensor_mgmt, devmon, vpd, frudata, fructrl, bus_tools, libipmi, power_mgmt

## 输出格式（严格 JSON，不要添加 markdown 代码块标记）
{{
  "intent": "relationship|code_lookup|architecture|debug|general",
  "components": ["组件名列表"],
  "sub_queries": ["关键词子查询1", "关键词子查询2"],
  "reasoning": "分析理由"
}}

用户问题：{question}
"""


@dataclass
class AnalyzedQuery:
    original: str
    intent: str = "general"
    components: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    reasoning: str = ""


class QueryAnalyzer:
    """Use LLM to analyze query intent and generate optimal sub-queries."""

    def __init__(self, llm: ChatOpenAI, console=None):
        self.llm = llm
        self.console = console

    def analyze(self, question: str) -> AnalyzedQuery:
        from rich.panel import Panel

        prompt = _ANALYZE_PROMPT.format(question=question)
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_response(question, response.content)

            if self.console:
                sub_text = "\n".join(f"  {i}. {q}" for i, q in enumerate(result.sub_queries, 1))
                self.console.print(Panel(
                    f"[bold]Prompt sent to LLM:[/bold]\n{prompt[:200]}...\n\n"
                    f"[bold]LLM raw response:[/bold]\n{response.content}\n\n"
                    f"[bold]Parsed intent:[/bold] {result.intent}\n"
                    f"[bold]Components:[/bold] {', '.join(result.components) or '-'}\n"
                    f"[bold]Sub-queries:[/bold]\n{sub_text}\n"
                    f"[bold]Reasoning:[/bold] {result.reasoning}",
                    title="[yellow]Step 2: Query Analysis[/yellow]",
                    border_style="yellow",
                ))

            return result
        except Exception:
            logger.exception("Query analysis failed, falling back to original query")
            return AnalyzedQuery(original=question, sub_queries=[question])

    def _parse_response(self, question: str, raw: str) -> AnalyzedQuery:
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)
        sub_queries = data.get("sub_queries", [question])
        if not sub_queries:
            sub_queries = [question]

        return AnalyzedQuery(
            original=question,
            intent=data.get("intent", "general"),
            components=data.get("components", []),
            sub_queries=sub_queries,
            reasoning=data.get("reasoning", ""),
        )
