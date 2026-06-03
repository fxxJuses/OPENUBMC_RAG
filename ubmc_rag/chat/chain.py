"""ReAct Agent 核心模块，使用 DashScope Qwen LLM 驱动 openUBMC 代码检索对话。

实现基于工具调用的 ReAct 推理模式：
1. LLM 分析用户问题，决定是否需要调用检索工具
2. 选择合适的工具和查询词进行代码检索
3. 基于检索结果生成带来源引用的回答
4. 维护对话历史并支持历史裁剪
"""

from __future__ import annotations

import logging
import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from ubmc_rag.prompts import PromptLibrary

logger = logging.getLogger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"

def create_llm(api_key: str | None = None, model: str = _DEFAULT_MODEL) -> ChatOpenAI:
    """创建 DashScope Qwen LLM 实例。

    Args:
        api_key: DashScope API 密钥，为空时从环境变量读取
        model: 模型名称，默认 "qwen-plus"

    Returns:
        配置好的 ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=model,
        api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=_DASHSCOPE_BASE_URL,
        temperature=0.3,
    )


def _extract_final_answer(messages: list[BaseMessage]) -> str:
    """从 Agent 输出消息中提取最终 AI 回答。

    从消息列表末尾向前查找第一条不含 tool_calls 的 AIMessage。
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content or ""
    return ""


def _trim_history(messages: list[BaseMessage], max_messages: int = 40) -> list[BaseMessage]:
    """裁剪对话历史，截断过长的 ToolMessage 内容以控制上下文长度。

    保留最近 max_messages 条消息，对超过 2000 字符的工具返回内容进行截断。
    """
    if len(messages) <= max_messages:
        return messages

    trimmed = messages[-max_messages:]
    result = []
    for msg in trimmed:
        if isinstance(msg, ToolMessage) and len(msg.content) > 2000:
            result.append(ToolMessage(
                content=msg.content[:2000] + "\n...[truncated]",
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        else:
            result.append(msg)
    return result


def _render_debug_trace(console, new_messages: list[BaseMessage]) -> None:
    """渲染 Agent 工具调用的调试追踪信息。

    展示每一步的工具选择、调用参数、返回结果和最终回答。
    """
    from rich.panel import Panel

    for msg in new_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                console.print(Panel(
                    f"[bold]Calling:[/bold] {tc['name']}\n"
                    f"[bold]Args:[/bold] {tc['args']}",
                    title="[cyan]Agent: Tool Selection[/cyan]",
                    border_style="cyan",
                ))
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:500] + ("..." if len(msg.content) > 500 else "")
            console.print(Panel(
                f"[bold]Tool:[/bold] {msg.name}\n[bold]Output:[/bold]\n{preview}",
                title=f"[yellow]Tool Result: {msg.name}[/yellow]",
                border_style="yellow",
                subtitle=f"[dim]content_length={len(msg.content)}[/dim]",
            ))
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            preview = msg.content[:300] + (
                "..." if len(msg.content or "") > 300 else ""
            )
            console.print(Panel(
                f"[bold]Final answer:[/bold]\n{preview}",
                title="[green]Agent: Final Response[/green]",
                border_style="green",
            ))


def run_chat(
    config,
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    debug: bool = False,
) -> None:
    """运行交互式 CLI 对话循环。

    初始化 Agent、加载索引、启动交互式问答循环。
    每轮对话中 Agent 会根据问题自主决定是否调用工具。

    Args:
        config: 应用配置
        api_key: DashScope API 密钥
        model: LLM 模型名称
        debug: 是否启用调试模式（显示工具调用追踪）
    """
    from rich.console import Console

    from ubmc_rag.chat.retriever import create_retriever
    from ubmc_rag.chat.tools import create_tools
    from ubmc_rag.indexing.index_manager import IndexManager

    console = Console()

    # 初始化检索器和 LLM
    console.print("[bold cyan]Loading index...[/bold cyan]")
    retriever = create_retriever(config)
    llm = create_llm(api_key=api_key, model=model)

    index_mgr = IndexManager(config)
    index_mgr.load_index()

    # 尝试加载文档索引
    docs_loaded = index_mgr.load_docs_index()
    if docs_loaded:
        n = len(index_mgr.get_all_docs_chunks())
        console.print(f"[green]Loaded docs index: {n} doc chunks[/green]")

    # 创建 Agent
    tools = create_tools(retriever.engine, index_mgr)
    prompt_lib = PromptLibrary()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt_lib.get_system_prompt(),
    )

    if debug:
        console.print(
            "[bold yellow]Debug mode ON — "
            "Agent tool calls will be traced[/bold yellow]"
        )

    console.print("[bold green]openUBMC Code Assistant ready![/bold green]")
    console.print("Type your question (or 'quit' to exit):\n")

    messages: list[BaseMessage] = []

    while True:
        try:
            question = console.input("[bold blue]> [/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break

        messages.append(HumanMessage(content=question))

        result = agent.invoke({"messages": messages})
        new_messages = result["messages"]

        if debug:
            _render_debug_trace(console, new_messages[len(messages):])

        final_answer = _extract_final_answer(new_messages)
        console.print(f"\n{final_answer}\n")

        # 裁剪对话历史
        messages = _trim_history(new_messages, max_messages=40)
