# openUBMC Code RAG 系统架构设计

## 1. 背景与目标

openUBMC 是华为开源的 BMC（Baseboard Management Controller）管理软件，采用微组件架构，代码托管在 GitCode。项目涉及 11+ 核心组件，主要语言为 Lua（业务逻辑）、C/C++（驱动）、Python（构建工具），配合大量 JSON 配置文件（MDS 模型、CSR 设备描述、IPMI 命令定义）。

**目标**：借鉴 SourceGraph 的核心思想（精确关键词索引、符号提取、排名算法），构建一个免费的、自托管的代码 RAG 系统，通过 MCP Server + CLI + 交互式 Chat 实现快速代码检索与问答。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接入层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  CLI 工具     │  │ 交互式 Chat   │  │  MCP Server (FastMCP) │  │
│  │  (Typer)     │  │ (ReAct Agent) │  │  search_code          │  │
│  │              │  │              │  │  find_definitions      │  │
│  │  version     │  │  LangChain   │  │  find_references       │  │
│  │  index       │  │  DashScope   │  │  list_components       │  │
│  │  search      │  │  Qwen LLM   │  │  get_component_deps    │  │
│  │  components  │  │  5 RAG Tools │  │  + 3 Resources         │  │
│  │  serve       │  │  多轮对话     │  │                        │  │
│  │  chat        │  │  引用溯源     │  │                        │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │
│         │                 │                      │               │
└─────────┼─────────────────┼──────────────────────┼───────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       混合搜索引擎                               │
│  ┌────────────────┐  ┌─────────────┐  ┌───────────────────┐    │
│  │ QueryProcessor  │  │ HybridSearch│  │   Reranker        │    │
│  │ 查询理解        │  │ RRF 融合    │  │ 符号/路径/多样性   │    │
│  │ 过滤提取        │  │             │  │ 重排              │    │
│  └────────┬───────┘  └──────┬──────┘  └───────────────────┘    │
│           │                  │                                   │
└───────────┼──────────────────┼──────────────────────────────────┘
            │                  │
            ▼                  ▼
┌───────────────────┐  ┌───────────────────┐
│  BM25 稀疏索引     │  │  Dense 向量索引    │
│  (rank_bm25)      │  │  (ChromaDB/HNSW)  │
│                   │  │                   │
│  代码感知分词器     │  │  DashScope         │
│  camelCase 拆分   │  │  text-embedding-v4 │
│  snake_case 拆分  │  │  1024 维            │
│  精确符号匹配      │  │  语义相似度        │
└─────────┬─────────┘  └────────┬──────────┘
          │                      │
          └──────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      索引管理层                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  IndexManager                                         │       │
│  │  • 编排 Embedder → ChromaDB + BM25 双写              │       │
│  │  • 增量更新（文件 MD5 校验）                           │       │
│  │  • 索引持久化与加载                                    │       │
│  └──────────────────────────────────────────────────────┘       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据摄取管线                                 │
│  ┌──────────┐  ┌────────────┐  ┌────────────────────────────┐  │
│  │ GitSync   │  │ FileFilter │  │  AST Parsers (Tree-sitter) │  │
│  │           │  │            │  │                            │  │
│  │ GitCode   │→ │ 语言过滤    │→ │  Lua Parser               │  │
│  │ 仓库拉取   │  │ .gitignore │  │  C/C++ Parser             │  │
│  │           │  │ 排除规则    │  │  Python Parser            │  │
│  └──────────┘  └────────────┘  │  JSON Parser (Schema感知)   │  │
│                                 │  Markdown Parser           │  │
│                                 └────────────┬───────────────┘  │
│                                              │                  │
│                                 ┌────────────▼───────────────┐  │
│                                 │  Chunker (分块协调器)       │  │
│                                 │  • 多语言解析器选择          │  │
│                                 │  • AST 感知分块             │  │
│                                 │  • 符号提取                 │  │
│                                 └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 核心模块说明

### 3.1 数据摄取管线（Ingestion Pipeline）

**GitSync** — 从 GitCode 拉取 openUBMC 组织下的多个仓库：
- 支持 `git clone` + `git pull` 增量同步
- 配置驱动：YAML 中声明要索引的仓库名列表

**FileFilter** — 文件过滤：
- 基于扩展名映射到语言类型（`.lua` → Lua, `.c/.h` → C 等）
- 排除规则：`build/`, `gen/`, `test/`, `third_party/` 等
- 尊重 `.gitignore` 规则

**AST Parsers** — 多语言 AST 感知解析器：

| 解析器 | 技术 | 分块策略 | 特殊处理 |
|--------|------|----------|----------|
| LuaParser | Tree-sitter-lua | 函数/方法级 + class() 模式识别 | `function_declaration` 节点、method_index_expression |
| CCppParser | Tree-sitter-c/cpp | 函数/结构体/类型定义级 | typedef 包含 struct 去重、LUAMOD_API 绑定识别 |
| PythonParser | Tree-sitter-python | 函数/类级，大类按方法拆分 | >200 行类自动拆分 |
| JsonParser | Schema 感知 | 按 JSON Schema 语义拆分 | service.json → 依赖/接口元数据<br>model.json → 按 MDS 类拆分<br>ipmi.json → 按命令拆分<br>.sr → 拓扑/对象拆分 |
| MarkdownParser | 正则 | 按 ATX 标题切分 | 保留表格内容 |

**Chunker** — 分块协调器：
- 根据文件扩展名自动选择解析器
- 编排解析流程，聚合所有 CodeChunk

### 3.2 索引层（Indexing）

**双索引架构**：

```
CodeChunk → Embedder → ChromaDB (HNSW 余弦相似度)
          → BM25Index (rank_bm25 代码感知分词)
```

**Embedder**：
- 模型：DashScope text-embedding-v4（OpenAI 兼容接口）
- 1024 维向量
- 批量编码（batch_size=10），单文本最长 24000 字符
- 速率限制：最小 0.1s 间隔
- 失败重试 1 次，二次失败填充零向量

**VectorStore (ChromaDB)**：
- 持久化存储到 `data/index/` 目录
- HNSW + cosine 相似度
- 存储：content + embedding + metadata（语言、仓库、文件路径、符号名、chunk 类型）
- 支持元数据过滤查询

**BM25Index**：
- 代码感知分词器：camelCase/snake_case/运算符边界切分
- 参数：k1=1.5, b=0.75
- 内存索引，JSON 序列化持久化
- 精确关键词匹配：符号名、标识符、函数名

**IndexManager**：
- 编排 Embedder → ChromaDB + BM25 双写
- 增量更新：文件 MD5 校验
- 索引统计和加载

### 3.3 混合搜索引擎（Hybrid Search）

**查询处理 (QueryProcessor)**：
- 自然语言/代码查询自动分类
- 过滤条件提取（语言、chunk 类型）
- 关键词提取（中英文停用词过滤）

**RRF 融合 (Reciprocal Rank Fusion)**：

```
RRF_score(d) = w_bm25 × 1/(k + rank_bm25(d)) + w_dense × 1/(k + rank_dense(d))
```

- 默认权重：BM25=0.4, Dense=0.6（语义查询偏好）
- 代码片段查询时反转：BM25=0.6, Dense=0.4（精确匹配偏好）
- k=60（标准 RRF 常量，防止头部结果主导）

**重排 (Reranker)**：
- 精确符号名匹配 → ×1.5
- 文件路径子串匹配 → ×1.3
- MDS 模型类名精确匹配 → ×2.0
- 同文件多样性限制：最多 3 条结果，超出降权 ×0.7

### 3.4 用户接入层

**CLI 工具** (Typer)：

| 命令 | 功能 |
|------|------|
| `ubmc-rag version` | 显示版本号 |
| `ubmc-rag index [--repos] [--full-rebuild] [--clone-missing]` | 构建或更新索引 |
| `ubmc-rag search QUERY [-l lang] [-r repo] [-t type] [-k N] [--code] [--format table\|json\|plain]` | 混合搜索 |
| `ubmc-rag components [-v] [--format table\|json]` | 列出组件统计 |
| `ubmc-rag serve [--transport stdio\|sse] [--host] [--port]` | 启动 MCP 服务器 |
| `ubmc-rag chat [--model qwen-plus] [--api-key] [--debug]` | 启动交互式 RAG 对话 |

**MCP Server** (FastMCP)：

| 工具 | 功能 | 关键参数 |
|------|------|----------|
| `search_code` | 混合语义+关键词搜索 | query, language, repo, chunk_type, top_k |
| `find_definitions` | 查找符号定义 | symbol_name, language |
| `find_references` | 查找符号引用 | symbol_name |
| `list_components` | 列出组件 | - |
| `get_component_deps` | 获取组件依赖 | component_name |

MCP 资源：`ubmc://component/{name}/info`、`ubmc://mds/{name}/models`、`ubmc://mds/{name}/ipmi`

传输：stdio（Claude Desktop/VS Code）+ SSE（HTTP）

**Chat 模块** (ReAct Agent + LangChain)：

基于 LangChain 的 ReAct Agent 架构，LLM 自主决策何时调用检索工具，支持多轮对话。

| 组件 | 功能 |
|------|------|
| `chain.py` | Agent 主循环，DashScope Qwen LLM 接入，对话历史管理（max 40 轮），调试追踪 |
| `tools.py` | 5 个 LangChain @tool 工具，映射搜索引擎能力，结果标注 `[Source N]` 引用 |
| `retriever.py` | `UBMCRetriever` (LangChain BaseRetriever)，支持单查询和多查询合并检索 |

- System Prompt 强制引用溯源，回答必须附带 `[Source N]` 标记
- 历史裁剪：超出 40 条时截断，ToolMessage 内容截断至 2000 字符
- 调试模式：Rich 面板展示完整 Agent 推理轨迹

## 4. 数据模型

```
CodeChunk
├── chunk_id: str (UUID)
├── content: str (源代码文本)
├── file_path: str
├── repo_name: str
├── language: str (lua/c/cpp/python/json/markdown)
├── component_name: str
├── start_line / end_line: int
├── chunk_type: str (function/method/class/mds_model/mds_ipmi_cmd/mds_service/
│                  mds_type_def/csr_object/csr_topology/section/file/block/
│                  config_block/typedef)
├── symbols: list[Symbol]
│   ├── name: str
│   ├── kind: str (function/class/method/variable/interface/ipmi_command/
│   │              dependency/section)
│   ├── line_start / line_end: int
│   ├── language: str
│   └── signature: str | None
├── metadata: dict (mds_class, dependencies, netfn, cmd, ...)
└── embedding: list[float] | None (索引后清除)

SearchResult
├── chunk: CodeChunk
├── score: float
└── source: str ("bm25"/"dense"/"hybrid")

ComponentInfo
├── name, repo_name, language, description
├── file_count, function_count, class_count
├── dependencies, required_interfaces, provided_interfaces
├── ipmi_commands, mds_classes
└── to_dict() → JSON 序列化
```

## 5. 数据流

```
1. 索引阶段:
   GitCode repos ──clone──▶ 本地文件
   本地文件 ──FileFilter──▶ 可处理文件列表
   可处理文件 ──Parser──▶ CodeChunk[] (含符号提取)
   CodeChunk[] ──Embedder──▶ 含 embedding 的 CodeChunk[]
   CodeChunk[] ──IndexManager──▶ ChromaDB + BM25 双索引

2. 搜索阶段:
   用户查询 ──QueryProcessor──▶ ProcessedQuery (分类+过滤+关键词)
   ProcessedQuery ──┬──Dense Search──▶ 向量相似度排序
                   └──BM25 Search──▶ 关键词匹配排序
   双路结果 ──RRF 融合──▶ 统一排序
   统一排序 ──Reranker──▶ 最终结果 (boost + diversity)
   最终结果 ──CLI/MCP──▶ 用户

3. 对话阶段:
   用户提问 ──ReAct Agent──▶ LLM 自主决策
   Agent ──Tool Call──▶ search_code / find_definitions / find_references / ...
   检索结果 ──LLM 推理──▶ 引用溯源回答 [Source N]
   多轮对话 ──历史管理──▶ 上下文窗口内持续对话
```

## 6. 配置管理

基于 Pydantic V2 的分层配置系统，YAML 文件驱动。

| 配置类 | 管控范围 | 关键参数 |
|--------|----------|----------|
| `GitConfig` | 仓库同步 | base_url, clone_dir, branch, auth_token, repos |
| `IngestionConfig` | 数据摄取 | languages（扩展名/启用状态/模式）, exclude_paths, chunk 大小限制 |
| `IndexingConfig` | 索引构建 | persist_dir, embedding_provider, embedding_dim(1024), BM25 参数 |
| `SearchConfig` | 搜索调优 | RRF k/权重, top_k, boost 因子, diversity 限制 |
| `MCPConfig` | 服务部署 | transport, host, port |

默认配置文件 `config/default_config.yaml`，通过 `AppConfig.from_yaml()` 加载，自动读取 `.env` 环境变量。

## 7. 辅助模块

- **logging.py**：基于 Rich Handler 的日志系统，`setup_logging(level)` + `get_logger(name)`
- **paths.py**：数据目录解析 `resolve_data_dir()`，目录创建 `ensure_dir()`

## 8. 技术选型依据

| 选择 | 替代方案 | 选择理由 |
|------|----------|----------|
| ChromaDB | Qdrant, Milvus, FAISS | Python 原生，轻量，快速原型，适合中小规模 |
| DashScope text-embedding-v4 | Jina Code V2, CodeBERT | 国内 API 无需翻墙，1024 维精度高，OpenAI 兼容接口 |
| Tree-sitter | 正则、ctags | AST 精确分块，召回率 +4.3%，多语言统一接口 |
| RRF 融合 | 线性组合、ConvexCE | Rank-based 不受分数尺度影响，无需归一化 |
| FastMCP | 原始 MCP SDK | 装饰器 API 减少样板代码，PyPI 包含在 mcp 中 |
| rank_bm25 | Whoosh, Elasticsearch | 纯 Python，轻量，可自定义分词器 |
| LangChain ReAct Agent | 原生 Prompt, LangGraph | 成熟的 Agent 框架，工具调用标准化，对话历史管理 |
| DashScope Qwen (qwen-plus) | OpenAI GPT, Claude API | 国内访问稳定，中文能力强，OpenAI 兼容协议 |
| Pydantic V2 | dataclasses, attrs | 配置验证，YAML 反序列化，类型安全 |
