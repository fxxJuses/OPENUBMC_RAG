# openUBMC Code RAG

基于 RAG（检索增强生成）的 openUBMC 代码问答系统。通过混合检索引擎（向量语义 + BM25 关键词）从 openUBMC 微组件代码库中检索相关代码片段，由 LLM 生成精准回答。

## 核心特性

- **AST 感知分块**：基于 Tree-sitter 解析 Lua / C / C++ 源码，按函数、类、结构体等语义边界切分
- **混合检索**：ChromaDB 向量搜索 + BM25 关键词搜索，通过 RRF（Reciprocal Rank Fusion）融合排序
- **多轮对话**：追问自动改写为独立搜索 query，结合上下文检索
- **MCP Server**：通过 FastMCP 暴露 5 个检索工具，可接入 Claude Desktop / VS Code
- **CLI 工具**：完整的命令行界面，支持索引构建、搜索、交互式问答

## 系统架构

```
GitCode 仓库 → AST 解析分块 → DashScope Embedding → 混合索引
                                                           ↓
                 CLI / MCP Client ← LLM (Qwen) ← 混合检索引擎 (RRF)
```

## 快速开始

### 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone <repo-url>
cd openUBMC_RAG
uv sync
```

### 配置 API Key

在项目根目录创建 `.env` 文件：

```
DASHSCOPE_API_KEY=sk-your-key-here
```

获取 API Key：[DashScope 控制台](https://dashscope.console.aliyun.com/)

### 构建索引

```bash
# 克隆仓库并构建索引
ubmc-rag index --clone-missing

# 完全重建
ubmc-rag index --full-rebuild
```

### 搜索代码

```bash
# 自然语言搜索
ubmc-rag search "sensor 组件如何获取温度数据"

# 按语言过滤
ubmc-rag search "ThresholdSensor" -l lua

# JSON 格式输出
ubmc-rag search "ipmi_get_sensor_reading" -f json
```

### 交互式问答

```bash
ubmc-rag chat                  # 默认 qwen-plus 模型
ubmc-rag chat -m qwen-max      # 使用更强模型
```

### 启动 MCP Server

```bash
ubmc-rag serve                 # stdio 模式（Claude Desktop）
ubmc-rag serve -t sse -p 8080  # SSE HTTP 模式
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `ubmc-rag index` | 构建/更新搜索索引 |
| `ubmc-rag search QUERY` | 搜索代码 |
| `ubmc-rag chat` | 交互式 RAG 问答 |
| `ubmc-rag components` | 列出已索引的组件 |
| `ubmc-rag serve` | 启动 MCP Server |
| `ubmc-rag version` | 查看版本 |

## 项目结构

```
openUBMC_RAG/
├── config/default_config.yaml        # 全局配置
├── ubmc_rag/
│   ├── cli/                          # CLI 命令（Typer）
│   ├── config/                       # Pydantic V2 配置管理
│   ├── ingestion/                    # 数据摄取
│   │   ├── git_sync.py               # GitCode 仓库克隆
│   │   ├── file_filter.py            # 文件过滤
│   │   ├── parsers/                  # AST 解析器（Lua/C++/JSON）
│   │   └── chunker.py                # 分块协调器
│   ├── indexing/                     # 索引管理
│   │   ├── embedder.py               # DashScope 嵌入 API
│   │   ├── vector_store.py           # ChromaDB 向量存储
│   │   ├── bm25_index.py             # BM25 关键词索引
│   │   └── index_manager.py          # 索引编排
│   ├── search/                       # 检索引擎
│   │   ├── hybrid_search.py          # RRF 混合检索
│   │   ├── query_processor.py        # 查询理解
│   │   └── reranker.py               # 结果重排
│   ├── chat/                         # LLM 问答
│   │   ├── chain.py                  # LangChain RAG Chain
│   │   └── retriever.py              # 自定义 Retriever
│   └── mcp_server/server.py          # MCP Server
└── tests/                            # 单元测试
```

## 技术栈

| 组件 | 技术 |
|------|------|
| CLI 框架 | Typer + Rich |
| 配置管理 | Pydantic V2 + YAML |
| AST 解析 | Tree-sitter (Lua/C/C++) |
| 向量数据库 | ChromaDB (HNSW cosine) |
| 关键词检索 | BM25Okapi |
| 嵌入模型 | DashScope text-embedding-v4 |
| LLM | DashScope Qwen (qwen-plus/qwen-max) |
| RAG 框架 | LangChain |
| MCP Server | FastMCP |

## 配置说明

编辑 `config/default_config.yaml`：

```yaml
git:
  base_url: "https://gitcode.com/openUBMC"
  repos:
    - name: "sensor"
    - name: "devmon"
    # ...

indexing:
  embedding_provider: "dashscope"
  embedding_dim: 1024

search:
  bm25_weight: 0.4      # BM25 权重
  dense_weight: 0.6     # 向量搜索权重
  rrf_k: 60             # RRF 常数
```

## License

MIT
