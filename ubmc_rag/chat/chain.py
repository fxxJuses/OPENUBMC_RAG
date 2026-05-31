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

logger = logging.getLogger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"

_AGENT_SYSTEM_PROMPT = """\
你是 openUBMC 代码助手，专门帮助开发者理解 openUBMC 微组件架构的代码。

## 工作策略
1. 先分析用户问题，判断是否需要检索代码
   - 需要检索：涉及具体代码、函数、组件、架构细节
   - 不需要检索：基于已检索结果的追问（如画图、进一步解释）、纯概念讨论
2. 根据问题选择合适的工具和查询词，可以多次调用不同工具后再回答
3. 如果之前的对话历史中已有相关检索结果，可以直接基于上下文回答

## 基本规则
1. 引用代码时标注文件路径和行号，格式：`repo_name/file_path:start_line`
2. 用中文回答，代码保持原文
3. 解释代码时结合 openUBMC 的微组件架构背景（MDS 模型、MDB 接口、组件间 RPC 通信等）

## 证据约束（严格遵守）
4. 每个事实性论断必须标注来源，格式：论断内容 [Source N]
5. 只根据检索到的代码回答，不要使用你的先验知识进行推测
6. 如果检索结果不足以回答问题，明确说"根据检索到的代码，无法确定"并建议用户用更具体的关键词搜索
7. 不要编写假设性或示例性代码。如果要说明某个机制，只引用源码中实际存在的代码
8. 不要对组件之间的关系做推理，除非源码中有明确的调用、require、import 等直接证据
"""


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

    # 创建 Agent
    tools = create_tools(retriever.engine, index_mgr)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=_AGENT_SYSTEM_PROMPT,
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
