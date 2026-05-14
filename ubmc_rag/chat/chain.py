"""RAG Chain with DashScope Qwen LLM + openUBMC code retriever."""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"

_SYSTEM_PROMPT = """\
你是 openUBMC 代码助手，专门帮助开发者理解 openUBMC 微组件架构的代码。

基于检索到的代码片段回答用户问题。遵循以下规则：

1. 引用代码时标注文件路径和行号，格式：`repo_name/file_path:start_line`
2. 如果检索结果不足以回答，明确告知用户，并建议用更具体的关键词搜索
3. 用中文回答，代码保持原文
4. 解释代码时结合 openUBMC 的微组件架构背景（MDS 模型、MDB 接口、组件间 RPC 通信等）
"""

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "你的任务是将用户的追问改写为独立的代码搜索查询。"
        "结合对话历史，生成一个具体的、可用于代码检索的查询。"
        "只输出改写后的查询，不要解释。"
        "如果用户的问题已经是独立的（没有指代前文），直接原样返回。"
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def create_llm(api_key: str | None = None, model: str = _DEFAULT_MODEL) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=_DASHSCOPE_BASE_URL,
        temperature=0.3,
    )


def _rewrite_query(llm: ChatOpenAI, question: str, history: list) -> str:
    """Rewrite a follow-up question into a standalone search query."""
    if not history:
        return question
    chain = _REWRITE_PROMPT | llm
    response = chain.invoke({"question": question, "history": history})
    return response.content.strip()


def create_rag_chain(llm: ChatOpenAI):
    """Create a simple RAG chain: prompt -> generate."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", """\
基于以下检索到的代码片段回答问题。

检索结果：
{context}

用户问题：{question}
"""),
    ])

    chain = prompt | llm
    return chain


def run_chat(config, api_key: str | None = None, model: str = _DEFAULT_MODEL) -> None:
    """Run interactive CLI chat loop."""
    from rich.console import Console
    from rich.panel import Panel

    from ubmc_rag.chat.retriever import create_retriever

    console = Console()

    # Initialize
    console.print("[bold cyan]Loading index...[/bold cyan]")
    retriever = create_retriever(config)
    llm = create_llm(api_key=api_key, model=model)
    chain = create_rag_chain(llm)

    console.print("[bold green]openUBMC Code Assistant ready![/bold green]")
    console.print("Type your question (or 'quit' to exit):\n")

    history: list = []

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

        # Rewrite follow-up questions into standalone search queries
        search_query = _rewrite_query(llm, question, history)
        if search_query != question:
            console.print(f"[dim]Search query: {search_query}[/dim]")

        # Retrieve
        docs = retriever.invoke(search_query)

        if not docs:
            console.print("[yellow]No relevant code found.[/yellow]\n")
            continue

        # Build context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            context_parts.append(
                f"[{i}] {m['repo']}/{m['file_path']}:{m['start_line']}-{m['end_line']} "
                f"(score={m['score']})\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Show sources
        console.print(Panel(
            "\n".join(
                f"  [{i}] {doc.metadata['repo']}/{doc.metadata['file_path']}:"
                f"{doc.metadata['start_line']}-{doc.metadata['end_line']}"
                for i, doc in enumerate(docs, 1)
            ),
            title="Sources",
            border_style="dim",
        ))

        # Generate
        response = chain.invoke({
            "context": context,
            "question": question,
            "history": history,
        })

        answer = response.content
        console.print(f"\n{answer}\n")

        # Update history (keep last 10 turns)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        if len(history) > 20:
            history = history[-20:]
