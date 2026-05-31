"""Agent 回答质量评估器。

运行 Agent 处理回归测试集中的查询，收集回答和工具调用轨迹，
然后使用 LLM-as-Judge 进行四层加权评分。

模型隔离设计：回答使用 Qwen，评分使用 GLM，避免自我评估偏差。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evaluation.agent.judge import JudgeResult, create_judge_llm, judge_answer
from evaluation.datasets.schema import RegressionDataset, TestCase
from ubmc_rag.config.settings import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentEvalResult:
    """单条 Agent 评估结果。

    Attributes:
        test_case_id: 测试用例 ID
        query: 查询文本
        answer: Agent 最终回答
        tool_calls: 工具调用列表 [{name, args}]
        judge_result: Judge 评分结果
        error: 执行错误信息（如有）
    """

    test_case_id: str
    query: str
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    judge_result: JudgeResult | None = None
    error: str = ""


@dataclass
class AgentEvalMetrics:
    """Agent 评估汇总指标。

    Attributes:
        avg_solution_quality: 平均解决方案质量 (0-10)
        avg_localization: 平均定位准确性 (0-10)
        avg_completeness: 平均完整性 (0-10)
        avg_evidence_reliability: 平均证据可靠性 (0-10)
        avg_overall_score: 平均加权总分 (0-10)
        pass_rate: 通过率 (≥ 6.0 的比例)
        avg_tool_calls: 平均工具调用次数
        hallucination_rate: 幻觉率 (1 - evidence_reliability/10)
        total_cases: 总用例数
        per_case: 每条用例的详细结果
    """

    avg_solution_quality: float = 0.0
    avg_localization: float = 0.0
    avg_completeness: float = 0.0
    avg_evidence_reliability: float = 0.0
    avg_overall_score: float = 0.0
    pass_rate: float = 0.0
    avg_tool_calls: float = 0.0
    hallucination_rate: float = 0.0
    total_cases: int = 0
    per_case: list[AgentEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为 JSON 可序列化字典。"""
        return {
            "avg_solution_quality": round(self.avg_solution_quality, 2),
            "avg_localization": round(self.avg_localization, 2),
            "avg_completeness": round(self.avg_completeness, 2),
            "avg_evidence_reliability": round(self.avg_evidence_reliability, 2),
            "avg_overall_score": round(self.avg_overall_score, 2),
            "pass_rate": round(self.pass_rate, 4),
            "avg_tool_calls": round(self.avg_tool_calls, 1),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "total_cases": self.total_cases,
        }


class AgentEvaluator:
    """Agent 回答质量评估器。

    使用与生产环境相同的 Agent 架构处理查询，
    然后用不同的 LLM 进行评分。

    索引在初始化时加载一次，所有用例共享同一引擎实例。

    Attributes:
        config: 应用配置
        judge_llm: 评分用 LLM（GLM）
        answer_model: 回答用模型名称（Qwen）
        index_mgr: 索引管理器（复用）
        engine: 混合搜索引擎（复用）
    """

    def __init__(
        self,
        config: AppConfig,
        judge_model: str = "glm-4-flash",
        answer_model: str = "qwen-plus",
    ):
        self.config = config
        self.answer_model = answer_model
        self.judge_llm = create_judge_llm(model=judge_model)

        # 加载索引一次，所有用例共享
        from ubmc_rag.indexing.index_manager import IndexManager
        from ubmc_rag.search.hybrid_search import HybridSearchEngine

        self.index_mgr = IndexManager(config)
        loaded = self.index_mgr.load_index()
        if not loaded:
            raise RuntimeError("No index found. Run 'ubmc-rag index' first.")

        chunks = self.index_mgr.get_all_chunks()
        self.engine = HybridSearchEngine(
            embedder=self.index_mgr.embedder,
            vector_store=self.index_mgr.vector_store,
            bm25=self.index_mgr.bm25,
            config=config,
        )
        self.engine.set_chunk_index(chunks)
        logger.info("AgentEvaluator initialized with %d chunks", len(chunks))

    def evaluate_single(
        self,
        test_case: TestCase,
        max_turns: int = 5,
    ) -> AgentEvalResult:
        """对单条测试用例运行 Agent 并评分。

        Args:
            test_case: 回归测试用例
            max_turns: Agent 最大轮次

        Returns:
            评估结果
        """
        result = AgentEvalResult(
            test_case_id=test_case.id,
            query=test_case.query,
        )

        try:
            # 构建 Agent（每次新建以确保无状态）
            agent, engine = self._create_agent()

            # 运行 Agent
            messages = [HumanMessage(content=test_case.query)]
            agent_result = agent.invoke({"messages": messages})

            # 提取回答和工具调用
            result.answer = self._extract_answer(agent_result["messages"])
            result.tool_calls = self._extract_tool_calls(agent_result["messages"])

            # 收集检索上下文用于 Judge 评估证据可靠性
            retrieved_context = self._collect_retrieved_context(agent_result["messages"])

            # Judge 评分
            expected_files = [
                {"repo_name": f.repo_name, "file_path": f.file_path}
                for f in test_case.expected_files
            ]
            result.judge_result = judge_answer(
                self.judge_llm,
                query=test_case.query,
                answer=result.answer,
                expected_files=expected_files,
                retrieved_context=retrieved_context,
            )

        except Exception as e:
            logger.error("Error evaluating case %s: %s", test_case.id, e)
            result.error = str(e)

        return result

    def evaluate_batch(
        self,
        dataset: RegressionDataset,
        max_turns: int = 5,
    ) -> AgentEvalMetrics:
        """对整个数据集运行 Agent 评估。

        Args:
            dataset: 回归测试数据集
            max_turns: Agent 最大轮次

        Returns:
            汇总的评估指标
        """
        results: list[AgentEvalResult] = []

        for i, tc in enumerate(dataset.test_cases):
            logger.info(
                "Evaluating case %d/%d: %s (%s)",
                i + 1,
                len(dataset.test_cases),
                tc.id,
                tc.query[:50],
            )
            result = self.evaluate_single(tc, max_turns=max_turns)
            results.append(result)

            # 进度输出
            if result.judge_result:
                logger.info(
                    "  Score: %.2f (pass=%s)",
                    result.judge_result.weighted_score,
                    result.judge_result.passed,
                )

        return self._aggregate_metrics(results)

    def _create_agent(self):
        """创建 Agent 实例（复用已加载的索引引擎）。"""
        from langchain.agents import create_agent

        from ubmc_rag.chat.chain import _AGENT_SYSTEM_PROMPT, create_llm
        from ubmc_rag.chat.tools import create_tools

        tools = create_tools(self.engine, self.index_mgr)
        llm = create_llm(model=self.answer_model)
        agent = create_agent(model=llm, tools=tools, system_prompt=_AGENT_SYSTEM_PROMPT)

        return agent, self.engine

    def _extract_answer(self, messages: list) -> str:
        """从 Agent 输出中提取最终回答。"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content or ""
        return ""

    def _extract_tool_calls(self, messages: list) -> list[dict]:
        """提取所有工具调用记录。"""
        calls = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append(
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        }
                    )
        return calls

    def _collect_retrieved_context(self, messages: list) -> str:
        """收集所有工具返回的检索上下文（供 Judge 评估证据可靠性）。"""
        contexts = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content or ""
                if len(content) > 500:
                    content = content[:500] + "..."
                contexts.append(f"[{msg.name}]: {content}")
        return "\n---\n".join(contexts)

    def _aggregate_metrics(self, results: list[AgentEvalResult]) -> AgentEvalMetrics:
        """汇总评估结果为 AgentEvalMetrics。"""
        valid = [r for r in results if r.judge_result is not None and not r.error]
        n = len(valid)

        if n == 0:
            return AgentEvalMetrics(
                total_cases=len(results),
                per_case=results,
            )

        avg_sq = sum(r.judge_result.solution_quality for r in valid) / n
        avg_loc = sum(r.judge_result.localization for r in valid) / n
        avg_comp = sum(r.judge_result.completeness for r in valid) / n
        avg_ev = sum(r.judge_result.evidence_reliability for r in valid) / n
        avg_overall = sum(r.judge_result.weighted_score for r in valid) / n
        pass_rate = sum(1 for r in valid if r.judge_result.passed) / n
        avg_tools = sum(len(r.tool_calls) for r in valid) / n

        return AgentEvalMetrics(
            avg_solution_quality=avg_sq,
            avg_localization=avg_loc,
            avg_completeness=avg_comp,
            avg_evidence_reliability=avg_ev,
            avg_overall_score=avg_overall,
            pass_rate=pass_rate,
            avg_tool_calls=avg_tools,
            hallucination_rate=1.0 - avg_ev / 10.0,
            total_cases=len(results),
            per_case=results,
        )
