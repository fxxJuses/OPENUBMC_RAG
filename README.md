# openUBMC Code RAG

基于 RAG（检索增强生成）的 openUBMC 代码问答系统。通过混合检索引擎（向量语义 + BM25 关键词）从 openUBMC 微组件代码库中检索相关代码片段，由 LLM 生成精准回答。

## 核心特性

- **AST 感知分块**：基于 Tree-sitter 解析 Lua / C / C++ / Python / JSON 源码，按函数、类、结构体等语义边界切分
- **混合检索**：ChromaDB 向量搜索 + BM25 关键词搜索，通过 RRF（Reciprocal Rank Fusion）融合排序
- **Reranker 重排**：符号/路径加权 + 多样性控制，提升结果精度
- **ReAct Agent**：LLM 自主决策检索策略，支持多轮对话和追问改写
- **MCP Server**：通过 FastMCP 暴露 5 个检索工具，可接入 Claude Desktop / VS Code
- **CLI 工具**：完整的命令行界面，支持索引构建、搜索、交互式问答、质量评估
- **检索评估**：50 条回归测试用例，13 项检索指标，四模式 A/B 对比，Bootstrap 置信区间

## 系统架构

```
GitCode 仓库 → AST 解析分块 → DashScope Embedding → 混合索引 (ChromaDB + BM25)
                                                              ↓
                CLI / MCP Client ← ReAct Agent (Qwen) ← 混合检索引擎 (RRF + Reranker)
```

## 检索效果

50 条回归测试用例，hybrid_reranked 模式实测：

| 核心指标 | 值 | 说明 |
|---------|-----|------|
| File@5 | 0.44 | Top-5 结果包含期望文件的比例 |
| File@10 | 0.50 | Top-10 结果包含期望文件的比例 |
| MRR | 0.35 | 首个相关结果排名倒数均值 |
| MAP | 0.26 | 平均精度均值 |
| NDCG@5 | 0.53 | 排序质量（考虑相关性等级） |
| CategoryHit@5 | 0.78 | Top-5 命中正确组件的比例 |
| SymbolHit@5 | 0.82 | Top-5 命中期望符号的比例 |

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
ubmc-rag chat --debug          # 开启调试追踪
ubmc-rag chat -m qwen-max      # 使用更强模型
```

### 启动 MCP Server

```bash
ubmc-rag serve                 # stdio 模式（Claude Desktop）
ubmc-rag serve -t sse -p 8080  # SSE HTTP 模式
```

### 质量评估

```bash
ubmc-rag eval retrieval                         # 单模式评测
ubmc-rag eval retrieval --mode all              # 四模式 A/B 对比
ubmc-rag eval retrieval --mode all -o result.json  # 导出 JSON
ubmc-rag eval agent                             # Agent 回答质量评估
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `ubmc-rag index` | 构建/更新搜索索引 |
| `ubmc-rag search QUERY` | 搜索代码 |
| `ubmc-rag chat` | 交互式 RAG 问答（ReAct Agent） |
| `ubmc-rag components` | 列出已索引的组件 |
| `ubmc-rag serve` | 启动 MCP Server |
| `ubmc-rag eval` | 运行检索/Agent 质量评估 |
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
│   │   ├── parsers/                  # AST 解析器（Lua/C++/Python/JSON/Markdown）
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
│   │   ├── agent.py                  # ReAct Agent
│   │   ├── retriever.py              # 自定义 Retriever
│   │   └── tools.py                  # RAG 工具集
│   └── mcp_server/server.py          # MCP Server
├── evaluation/                       # 评估框架
│   ├── datasets/                     # 回归测试数据集（50 条）
│   ├── retrieval/                    # 检索质量评估
│   ├── agent/                        # Agent 回答质量评估
│   └── report.py                     # 报告生成
└── tests/                            # 单元测试
```

## 技术栈

| 组件 | 技术 |
|------|------|
| CLI 框架 | Typer + Rich |
| 配置管理 | Pydantic V2 + YAML |
| AST 解析 | Tree-sitter (Lua/C/C++/Python) |
| 向量数据库 | ChromaDB (HNSW cosine) |
| 关键词检索 | BM25Okapi |
| 嵌入模型 | DashScope text-embedding-v4 (1024-dim) |
| LLM | DashScope Qwen (qwen-plus/qwen-max) |
| Agent 框架 | LangChain ReAct |
| MCP Server | FastMCP |
| 评估指标 | File@K, Precision@K, Recall@K, MRR, MAP, NDCG@K |

## 配置说明

编辑 `config/default_config.yaml`：

```yaml
git:
  base_url: "https://gitcode.com/openUBMC"
  repos:
    - name: "sensor"
    - name: "devmon"
    # ... 13 个微组件仓库

indexing:
  embedding_provider: "dashscope"
  embedding_dim: 1024

search:
  bm25_weight: 0.4      # BM25 权重
  dense_weight: 0.6     # 向量搜索权重
  rrf_k: 60             # RRF 常数
```

## 文档

- [系统架构设计](docs/design/architecture.md)
- [ReAct Agent 设计](docs/design/react-agent.md)
- [评估框架设计](docs/design/evaluation.md)
- [评估优化变更记录](docs/design/evaluation-v2-changelog.md)

## License

MIT
