# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openUBMC Code RAG — a self-hosted code retrieval-augmented generation system for the openUBMC BMC management platform. Indexes 11+ micro-component repositories (Lua, C/C++, JSON configs) and provides hybrid semantic + keyword search via CLI, MCP Server, and interactive Chat.

## Development Commands

```bash
# Install dependencies
uv sync

# Install with dev tools (pytest, ruff)
uv sync --extra dev

# Run CLI
ubmc-rag version
ubmc-rag index --clone-missing    # clone repos + build index
ubmc-rag search "query"           # hybrid search
ubmc-rag chat --debug             # interactive RAG agent with debug trace
ubmc-rag serve                    # MCP server (stdio)

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_parsers/test_lua_parser.py

# Lint
uv run ruff check .
uv run ruff format .
```

Required env: `DASHSCOPE_API_KEY` in `.env` (for embedding + LLM APIs).

## Architecture

**Entry point**: `ubmc_rag.cli.main:app` (Typer). Also runnable via `python -m ubmc_rag`.

**Three user-facing paths** share the same search engine:
- **CLI** (`cli/`) — Typer commands: index, search, components, chat, serve
- **MCP Server** (`mcp_server/server.py`) — FastMCP with 5 tools + 3 resources
- **Chat** (`chat/`) — LangChain ReAct Agent (DashScope Qwen LLM) with 5 RAG tools

**Core data flow**:
1. **Ingestion** (`ingestion/`): GitSync → FileFilter → AST Parsers (Tree-sitter) → CodeChunk[]
2. **Indexing** (`indexing/`): Embedder (DashScope text-embedding-v4, 1024-dim) → ChromaDB + BM25 dual write
3. **Search** (`search/`): QueryProcessor → Dense + BM25 parallel → RRF fusion → Reranker (symbol/path boost + diversity)

**Config** (`config/settings.py`): Pydantic V2 models loaded from `config/default_config.yaml`. `AppConfig.from_yaml()` is the main entry.

**Models** (`models/`): `CodeChunk` (core unit with symbols + metadata), `SearchResult`, `ComponentInfo` — all plain dataclasses.

## Key Conventions

- Python >= 3.10, line length 100 (ruff config in pyproject.toml)
- Ruff rules: E, F, I, N, W
- Parsers extend `BaseParser` (Tree-sitter based), dispatched by `Chunker` based on file extension
- All external API calls go through DashScope (OpenAI-compatible base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `data/` directory holds cloned repos and indexes (gitignored, not in version control)

## Design Docs

Detailed architecture and design decisions live in `docs/design/`:
- `architecture.md` — full system architecture
- `react-agent.md` — Chat ReAct Agent design
