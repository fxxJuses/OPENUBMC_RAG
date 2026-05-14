# openUBMC Code RAG 系统架构设计

## 1. 背景与目标

openUBMC 是华为开源的 BMC（Baseboard Management Controller）管理软件，采用微组件架构，代码托管在 GitCode。项目涉及 11+ 核心组件，主要语言为 Lua（业务逻辑）、C/C++（驱动）、Python（构建工具），配合大量 JSON 配置文件（MDS 模型、CSR 设备描述、IPMI 命令定义）。

**目标**：借鉴 SourceGraph 的核心思想（精确关键词索引、符号提取、排名算法），构建一个免费的、自托管的代码 RAG 系统，通过 MCP Server + CLI 工具实现快速代码检索。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接入层                                │
│  ┌──────────────┐              ┌──────────────────────────┐     │
│  │  CLI 工具     │              │  MCP Server (FastMCP)    │     │
│  │  (Typer)     │              │  search_code             │     │
│  │              │              │  find_definitions         │     │
│  │  index       │              │  find_references          │     │
│  │  search      │              │  list_components          │     │
│  │  components  │              │  get_component_deps       │     │
│  │  serve       │              │  + 3 Resources            │     │
│  └──────┬───────┘              └──────────┬───────────────┘     │
│         │                                  │                    │
└─────────┼──────────────────────────────────┼────────────────────┘
          │                                  │
          ▼                                  ▼
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
│  代码感知分词器     │  │  jina-embeddings- │
│  camelCase 拆分   │  │  v2-base-code     │
│  snake_case 拆分  │  │  768 维 / 8K ctx  │
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
- 模型：jinaai/jina-embeddings-v2-base-code
- 768 维向量，8192 token 上下文
- 支持 30 种语言（含 Lua）
- 批量编码，normalize_embeddings=True

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
| `ubmc-rag index [--repos] [--full-rebuild] [--clone-missing]` | 构建或更新索引 |
| `ubmc-rag search QUERY [-l lang] [-r repo] [-k N]` | 混合搜索 |
| `ubmc-rag components [-v]` | 列出组件统计 |
| `ubmc-rag serve [--transport stdio\|sse]` | 启动 MCP 服务器 |

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
├── chunk_type: str (function/method/class/mds_model/mds_ipmi_cmd/csr_object/section/...)
├── symbols: list[Symbol]
│   ├── name: str
│   ├── kind: str (function/class/method/variable/interface/ipmi_command)
│   ├── line_start / line_end: int
│   └── signature: str | None
└── metadata: dict (mds_class, dependencies, netfn, cmd, ...)
```

## 5. 数据流

```
1. 索引阶段:
   GitCode repos ──clone──▶ 本地文件
   本地文件 ──FileFilter──▶ 可处理文件列表
   可处理文件 ──Parser──▶ CodeChunk[] (含符号提取)
   CodeChunk[] ──Embedder──▶ 含 embedding 的 CodeChunk[]
   CodeChunk[] ──IndexManager──▶ ChromaDB + BM25 双索引

2. 查询阶段:
   用户查询 ──QueryProcessor──▶ ProcessedQuery (分类+过滤+关键词)
   ProcessedQuery ──┬──Dense Search──▶ 向量相似度排序
                   └──BM25 Search──▶ 关键词匹配排序
   双路结果 ──RRF 融合──▶ 统一排序
   统一排序 ──Reranker──▶ 最终结果 (boost + diversity)
   最终结果 ──CLI/MCP──▶ 用户
```

## 6. 技术选型依据

| 选择 | 替代方案 | 选择理由 |
|------|----------|----------|
| ChromaDB | Qdrant, Milvus, FAISS | Python 原生，轻量，快速原型，适合中小规模 |
| Jina Code V2 | CodeBERT, UniXcoder | 8K 上下文（vs 512），支持 Lua，Apache 2.0 |
| Tree-sitter | 正则、ctags | AST 精确分块，召回率 +4.3%，多语言统一接口 |
| RRF 融合 | 线性组合、ConvexCE | Rank-based 不受分数尺度影响，无需归一化 |
| FastMCP | 原始 MCP SDK | 装饰器 API 减少样板代码，PyPI 包含在 mcp 中 |
| rank_bm25 | Whoosh, Elasticsearch | 纯 Python，轻量，可自定义分词器 |
