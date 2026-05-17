# openUBMC Code RAG — 面试准备文档

本文档梳理项目涉及的核心技术点、设计决策、可能被追问的问题及参考回答。按主题分类，便于针对性准备。

---

## 一、项目概述（30 秒电梯演讲）

> 我做了一个针对 openUBMC（华为开源 BMC 管理软件）的代码 RAG 系统。openUBMC 采用微组件架构，代码托管在 GitCode 上，主要用 Lua 和 C/C++ 编写。系统通过 Tree-sitter 做 AST 感知的代码分块，用 ChromaDB 做向量索引、BM25 做关键词索引，两者通过 RRF 融合排序。Chat 模块采用 ReAct Agent 架构——LLM 自主决定是否调用检索工具（search_code、find_definitions 等 5 个 Tool），简单追问直接回答，代码查询按需检索，复杂问题可多次调用不同工具。同时通过 MCP Server 对外暴露检索能力，可以接入 Claude Desktop 或 VS Code。整个系统纯 Python 实现，不依赖 GPU。

---

## 二、RAG 基础概念

### Q: 什么是 RAG？为什么不用 fine-tuning？

**RAG（Retrieval-Augmented Generation）** 是一种将检索和生成结合的架构：先从外部知识库检索相关文档，再将检索结果作为上下文喂给 LLM 生成回答。

与 fine-tuning 的对比：

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时，更新索引即可 | 需要重新训练 |
| 成本 | 较低（只需建索引） | 高（需要训练资源） |
| 幻觉控制 | 强（基于检索结果回答） | 弱（依赖模型记忆） |
| 可解释性 | 强（可展示检索来源） | 弱（黑盒） |
| 适用场景 | 知识频繁更新、需要溯源 | 风格模仿、领域术语学习 |

本项目选择 RAG 的原因：代码库频繁更新，需要精确引用代码位置，且需要控制幻觉（不能编造不存在的 API）。

### Q: RAG 的典型流程？

```
用户问题 → Query 改写/理解 → 检索（向量 + 关键词） → 重排 → 构建 Prompt → LLM 生成
```

本项目的具体实现：
1. **Query 处理**：`QueryProcessor` 判断是否为代码查询，提取语言/类型过滤条件
2. **多轮改写**：`_rewrite_query()` 用 LLM 把追问改写为独立搜索 query
3. **意图分析**：`QueryAnalyzer` 用 LLM 分析意图（relationship/code_lookup/architecture/debug），生成子查询
4. **混合检索**：ChromaDB 向量搜索 + BM25 关键词搜索，RRF 融合；关系查询走多查询聚合
5. **重排**：符号匹配加权、文件路径匹配、MDS 模型匹配、结果多样性
6. **生成**：检索结果 + 对话历史 + 系统 Prompt（含 citation enforcement）→ Qwen LLM

---

## 三、代码分块（Chunking）

### Q: 为什么不能简单按行数切分？AST 感知分块的好处？

**按行数切分的问题：**
- 可能在函数/类中间截断，破坏语义完整性
- 一个函数被切成两半，检索到的一半缺乏上下文
- BM25 索引的 token 粒度不对，影响关键词检索效果

**AST 感知分块的优势：**
- 以函数、类、结构体为最小语义单元
- 检索到的片段本身就有完整语义，LLM 能直接理解
- 能提取符号信息（函数名、类名），用于后续精确匹配和重排

### Q: Tree-sitter 是什么？为什么选它？

Tree-sitter 是一个增量式解析器生成器，特点是：
- **快**：C 实现，能在毫秒级解析文件
- **增量解析**：文件修改后只重新解析变化部分
- **容错**：即使代码有语法错误也能生成部分 AST
- **多语言**：社区已为 40+ 语言提供了 grammar

替代方案对比：

| 方案 | 优势 | 劣势 |
|------|------|------|
| **Tree-sitter** | 快、容错、多语言 | 需要了解各语言的 node type |
| 正则匹配 | 简单 | 不精确，嵌套结构处理差 |
| LSP (pyright/clangd) | 精确、类型信息 | 启动慢、依赖重、不支持 Lua |
| ast 模块（Python 内置） | 零依赖 | 只支持 Python |

### Q: Lua 代码分块有什么特殊处理？

openUBMC 的 Lua 代码大量使用 `class()` 模式（类似 Lua 面向对象框架），这不是标准 Lua 语法。处理方式：

1. 识别 `variable_declaration` 中包含 `class(` 调用的节点
2. 从中提取类名和整个类定义体
3. 类体内的方法通过 `method_index_expression`（`function obj:method()` 语法）识别
4. 小文件（<=20 行）直接作为一个完整 chunk，避免过度拆分

---

## 四、向量搜索与嵌入

### Q: 嵌入模型选型过程？

经历了三次迭代：
1. **jina-embeddings-v2-base-code**（本地）：768 维，支持 30 种语言含 Lua，但不支持中文
2. **Qwen3-Embedding-0.6B**（本地 MPS）：中文友好，但在 Mac MPS 上 OOM（attention 矩阵溢出）
3. **DashScope text-embedding-v4**（在线 API）：1024 维，中英代码都友好，无本地资源限制

最终选择 DashScope API 的原因：
- 无需本地 GPU/MPS，跨平台一致
- 1024 维向量提供更丰富的语义表示
- OpenAI 兼容接口，切换成本低
- 批量调用支持（batch size <= 10）

### Q: 为什么用 ChromaDB？与其他向量数据库的对比？

| 数据库 | 优势 | 劣势 |
|--------|------|------|
| **ChromaDB** | 轻量、嵌入式、Python 原生、零运维 | 不适合超大规模（亿级） |
| FAISS | 极快、Facebook 出品 | 内存常驻、无持久化、无元数据过滤 |
| Milvus | 分布式、高性能 | 架构重、依赖多 |
| Pinecone | 全托管 | 付费、数据出域 |

本项目选择 ChromaDB 的原因：
- 纯 Python，pip install 即用
- 内置 HNSW 索引 + 元数据过滤（按语言、组件名过滤）
- 数据持久化到本地目录，无需额外服务
- 3921 条 chunk 的规模完全在 ChromaDB 舒适区

### Q: 向量搜索的原理？HNSW 是什么？

**向量搜索**本质是"找最近邻"：将文本编码为高维向量，在向量空间中找到距离最近的 k 个点。本项目用**余弦相似度**度量距离。

**HNSW（Hierarchical Navigable Small World）** 是一种近似最近邻（ANN）算法：
- 构建多层图结构，上层稀疏（长距离跳转），下层稠密（精确搜索）
- 搜索时从顶层开始逐层向下，类似跳表（skip list）
- 时间复杂度 O(log n)，比暴力搜索 O(n) 快很多
- 精度-速度权衡：通过 `ef_search` 参数控制，值越大越精确但越慢

---

## 五、BM25 与混合检索

### Q: BM25 的原理？与 TF-IDF 的区别？

**BM25** 是 TF-IDF 的改进版，核心公式：

```
BM25(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D| / avgdl))
```

与 TF-IDF 的关键区别：
- **词频饱和**：通过 k1 参数控制，词频增长到一定程度后收益递减（TF-IDF 线性增长）
- **文档长度归一化**：通过 b 参数控制长文档的惩罚程度
- 参数：k1=1.5（词频饱和点），b=0.75（长度归一化强度）

### Q: 为什么需要混合检索？只用向量搜索不行吗？

**向量搜索擅长**：语义相似（"获取温度" ↔ "读取传感器数值"），跨语言
**BM25 擅长**：精确关键词匹配（函数名 `ipmi_get_sensor_reading`），代码标识符

代码检索场景中，用户经常搜索精确的函数名、变量名，BM25 在这类查询上明显优于向量搜索。反之，自然语言描述的查询向量搜索更优。混合取长补短。

### Q: RRF（Reciprocal Rank Fusion）的原理？

```
RRF_score(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))
```

- k=60 是论文推荐的经验常数，作用是平滑排名（避免 rank=1 和 rank=2 差距过大）
- w_bm25=0.4, w_dense=0.6 是默认权重
- 代码查询时反转：BM25 权重提升到 0.6，因为精确匹配更重要

**为什么选 RRF 而不是分数加权融合？**
- RRF 基于排名而非原始分数，天然解决了两个检索系统分数尺度不同的问题（BM25 分数范围和余弦相似度范围完全不同）
- 无需归一化，实现简单

### Q: 代码感知分词器做了什么？

标准分词器（按空格/标点切分）对代码效果差。例如 `ipmi_get_sensor_reading` 应该被切成 `ipmi`, `get`, `sensor`, `reading` 四个 token。

实现方式：
```python
_TOKENIZE_RE = re.compile(
    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|\d+|[a-zA-Z]\w*|[^\s\w]"
)
```

处理 camelCase（`GetSensorReading` → `Get`, `Sensor`, `Reading`）、snake_case（`get_sensor` → `get`, `sensor`）和连续大写（`IPMI` → `IPMI`）。

---

## 六、重排（Reranking）

### Q: 重排规则的设计思路？

检索结果经过 RRF 融合后，还需要基于领域知识进一步调整：

| 规则 | 加权 | 原因 |
|------|------|------|
| 精确符号名匹配 | ×1.5 | 用户搜函数名时，定义位置应排最前 |
| 文件路径匹配 | ×1.3 | 搜索 "sensor" 时，sensor 目录下的文件更相关 |
| MDS 模型类名匹配 | ×2.0 | openUBMC 的 MDS 模型是核心抽象，精确匹配价值最高 |
| 同文件多样性限制 | 最多 3 条 | 避免某文件独占所有结果 |

**为什么不做 LLM rerank？** LLM rerank（如 Cohere Rerank）效果好但延迟高、有成本。规则 rerank 在代码检索场景已经够用——代码的精确匹配信号（函数名、文件路径）比语义匹配更可靠。

---

## 七、ReAct Agent 架构

### Q: 为什么从固定管线重构为 ReAct Agent？

**原架构**是固定 4 步串行管线：Rewrite → Analyze → Retrieve → Generate。三个核心痛点：

1. **追问触发无意义检索**：用户追问"画流程图"仍走完整 RAG 检索，Query Analyzer 还误判意图为 architecture
2. **检索策略不可动态调整**：单次 Analyze → Retrieve，LLM 无法根据中间结果补充检索
3. **检索决策与生成分离**：LLM 生成回答时无法主动补充检索

**ReAct Agent 解决方案**：LLM 自主决定是否调用工具、调用哪个工具、调用几次。追问时 LLM 看到对话历史有足够上下文就不调用工具，直接回答。

### Q: ReAct 和 Plan-and-Solve 有什么区别？为什么选 ReAct？

| 维度 | ReAct | Plan-and-Solve |
|------|-------|----------------|
| 决策方式 | LLM 逐步 reactive 决策 | 先生成完整计划再逐步执行 |
| 实现复杂度 | 低（单 Agent + Tools） | 高（多节点 StateGraph） |
| 延迟 | 简单查询 1 次 LLM 调用 | 每次查询至少 3 次（规划+执行+综合） |
| 复杂查询 | LLM 可能遗忘多步计划 | 显式计划确保执行完整 |

**选择 ReAct 的理由**：
1. 代码助手的核心场景是 1-2 次检索 + 生成，不需要复杂分步编排
2. 核心痛点（追问不触发检索）是 ReAct 的天然行为
3. 未来可渐进升级为 Plan-and-Solve

### Q: `create_agent` 和 `create_react_agent` 的区别？

`create_react_agent` 是 `langgraph.prebuilt` 中的旧 API，已被弃用。`create_agent` 是 `langchain.agents` 中的新 API（LangChain 1.2+），参数更清晰：

```python
# 旧（已弃用）
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model=llm, tools=tools, prompt=sys_prompt)

# 新（推荐）
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)
```

两者底层都返回 `CompiledStateGraph`，但新 API 支持 middleware、response_format 等扩展能力。

### Q: LangChain Tool 是怎么注册的？@tool 装饰器的原理？

本项目使用**工厂函数 + 闭包**模式注册 Tool：

```python
def create_tools(engine, index_mgr) -> list[BaseTool]:
    @tool
    def search_code(query: str, ...) -> str:
        """Tool description..."""
        return engine.search(...)  # 闭包捕获 engine
    return [search_code, ...]
```

`@tool` 装饰器将函数转为 `StructuredTool` 实例，自动从函数签名和 docstring 生成 Tool schema。`create_agent` 会将所有 Tool 的 name、description、input schema 注入给 LLM，LLM 通过 function calling 选择调用哪个工具。

**为什么用闭包而不是传参**：LangChain Tool 的函数签名必须是可序列化的（LLM 需要知道参数 schema），`engine`/`index_mgr` 不应出现在签名中。闭包让这些对象在函数内部可用，但对外不可见。

### Q: Agent 的 System Prompt 是怎么设计的？为什么工具说明不在 Prompt 里？

Prompt 分三部分：角色定义 + 工作策略 + 证据约束。

**工具说明不在 System Prompt 中的原因**：`create_agent` 自动将所有 Tool 的 `description` 和 `input schema` 注入 LLM 的 function calling 上下文。如果在 System Prompt 中重复工具列表，会导致信息冗余，且新增/修改 Tool 时需要同步更新两处。

### Q: 对话历史是怎么管理的？和固定管线有什么区别？

| 维度 | 固定管线 | ReAct Agent |
|------|---------|-------------|
| 历史类型 | HumanMessage + AIMessage | HumanMessage + AIMessage + ToolMessage |
| 追问处理 | 显式 LLM Rewrite 调用 | 对话历史自然理解 |
| 历史大小控制 | 保留最近 20 条 | 保留最近 40 条 + ToolMessage 截断 |

**ToolMessage 截断**：工具返回的代码片段可能很长（数千字符），历史超过 40 条时将 ToolMessage.content 截断到 2000 字符。因为 LLM 在当前轮已经处理了完整内容，历史中的截断版本足够维持上下文连贯性。

### Q: Agent 如何判断"不需要检索"的场景？

依赖 LLM 的推理能力 + System Prompt 中的工作策略引导：

```
1. 先分析用户问题，判断是否需要检索代码
   - 需要检索：涉及具体代码、函数、组件、架构细节
   - 不需要检索：基于已检索结果的追问（如画图、进一步解释）、纯概念讨论
```

LLM 看到对话历史中已有完整上下文时，会判断不需要调用工具。这比规则判断更灵活——"帮我更详细地解释第二步"和"还有其他加载方式吗"都是追问，但后者可能需要额外检索。

### Q: Debug 模式是怎么实现的？

Agent 调用 `invoke()` 后返回完整的 messages 链。遍历新增消息，按类型渲染 Rich Panel：

- `AIMessage` 含 `tool_calls` → 展示工具选择和参数（Agent Decision）
- `ToolMessage` → 展示工具返回摘要（Tool Result）
- `AIMessage` 不含 `tool_calls` → 展示最终回答摘要（Final Response）

关键代码：
```python
result = agent.invoke({"messages": messages})
new_messages = result["messages"]
if debug:
    _render_debug_trace(console, new_messages[len(messages):])
```

---

## 八、MCP Server

### Q: MCP 是什么？

**MCP（Model Context Protocol）** 是 Anthropic 提出的开放协议，让 LLM 应用通过标准化接口访问外部工具和数据源。类似于 USB-C 协议——统一了 LLM 与工具之间的连接方式。

本项目的 MCP Server 提供：
- 5 个工具：`search_code`, `find_definitions`, `find_references`, `list_components`, `get_component_deps`
- 3 个资源：`ubmc://component/{name}/info`, `ubmc://mds/{name}/models`, `ubmc://mds/{name}/ipmi`
- 支持 stdio（Claude Desktop）和 SSE（HTTP）两种传输方式

### Q: MCP 的 stdio 和 SSE 有什么区别？

- **stdio**：通过标准输入/输出通信，适合本地桌面应用（Claude Desktop 启动子进程）
- **SSE（Server-Sent Events）**：HTTP 长连接，适合远程服务或 Web 应用

---

## 九、性能与工程实践

### Q: 索引构建的内存优化？

3921 条 chunk 的嵌入计算是内存密集型。优化策略：
1. **分批处理**：每 64 条 chunk 为一批，计算嵌入后立即写入 ChromaDB
2. **及时释放**：写入后将 `chunk.embedding = None`，释放嵌入向量占用的内存
3. **显式 GC**：每批之间调用 `gc.collect()` 强制回收
4. **先 BM25 后向量**：BM25 索引构建不消耗 GPU/嵌入资源，先完成避免内存压力叠加

### Q: DashScope API 的限制和应对？

- **批量限制**：单次最多 10 条输入 → `_API_BATCH_SIZE = 10`
- **输入长度限制**：单条最大 8192 token → `_MAX_CHARS = 24000` 截断
- **重试策略**：失败后等待 2 秒重试一次，仍然失败则用零向量兜底
- **限速**：每批之间 sleep 0.1 秒（`_MIN_INTERVAL`）

### Q: 增量索引更新的设计？

通过 MD5 checksum 机制：
- 索引构建时保存每个 chunk 的 `{repo}:{file_path}` → MD5 映射
- 下次构建时对比 checksum，只重新处理变化的文件
- `full_rebuild=True` 时清空 ChromaDB 全量重建

---

## 十、系统设计类问题

### Q: 如果代码库规模增长 100 倍（40 万 chunk），系统怎么扩展？

分层回答：

1. **嵌入计算**：改用异步并发（aiohttp + asyncio），DashScope API 支持并发调用
2. **向量存储**：ChromaDB → Milvus/Qdrant（分布式向量数据库）
3. **BM25**：rank-bm25 是内存索引，40 万文档需要换成 Elasticsearch/OpenSearch
4. **检索**：增加 HNSW 的 ef_search 参数，或者引入 ANN 索引分片
5. **LLM**：增加 context 压缩（对检索结果做摘要），避免超出 token 限制

### Q: 检索质量怎么评估？

评估维度：
1. **召回率**：人工标注一批 query 的相关文档，计算检索结果的 Recall@K
2. **MRR（Mean Reciprocal Rank）**：第一个相关结果的排名倒数的均值
3. **端到端**：人工评估 LLM 回答的准确性和有用性（1-5 分）

本项目缺少的：自动化评估 pipeline。可用的方案：
- RAGAS 框架（自动生成测试集并评估）
- 人工标注 golden set

### Q: 如何减少 LLM 幻觉？

这是一个实际踩过的坑。用户问"sensor 和电源管理是否有联合关系"时，LLM 编造了两者的交互关系和假设性代码。根因有三：检索没有捞到跨组件证据、没有意图识别、Prompt 缺少证据约束。

**改进方案（三层防御）：**

1. **Prompt 加固（Citation Enforcement）**：要求每个论断必须标注 `[Source N]`，禁止使用先验知识推理，禁止编造示例代码，证据不足时必须说"无法确定"
2. **LLM 查询分析器**：在检索前用 LLM 分析用户意图（relationship / code_lookup / architecture / debug / general），对于关系类查询自动生成 3-5 个交叉引用子查询
3. **多查询检索聚合**：多个子查询分别检索后按 chunk_id 去重合并，扩大召回率

**效果对比**：改动前 sensor+power 关系查询幻觉率 100%（3/3 论断无支撑）；改动后 LLM 被迫基于检索证据回答，关系类问题要么给出有支撑的论断，要么正确告知"证据不足"。

**ReAct Agent 模式下的幻觉验证**：重构为 Agent 后，对 bios-pcie_device 关系查询的回答逐条核对源码，所有函数名、调用关系、参数名均准确，未发现幻觉。证据约束规则在 Agent 模式下仍然有效。

> 详细方案见 `docs/design/anti-hallucination.md` 和 `docs/design/react-agent.md`

---

## 十一、开放性问题

### Q: 这个项目你觉得还有什么不足？如果给你更多时间你会改进什么？

可以提到的方向（展示你的思考深度）：

1. **多语言 AST 解析**：目前 JSON/Python/Markdown 解析器已实现但未启用，可以开启以覆盖配置文件和文档
2. **Agent 模式**：已完成 ReAct Agent 架构重构，LLM 可自主调用 5 个工具（search_code、find_definitions、find_references、list_components、get_component_deps），按需检索
3. **评估体系**：缺少自动化的检索质量评估 pipeline
4. **增量更新**：目前 checksum 机制已实现但未完整对接 git pull 的增量流程
5. **Context 压缩**：长代码片段可以先用 LLM 做摘要，减少 token 消耗
6. **多模态**：支持图表、架构图的索引和检索

### Q: 你在这个过程中遇到了什么技术挑战？

可以讲的几个真实故事：

1. **Tree-sitter Lua 节点类型**：文档写的是 `function_definition`，实际是 `function_declaration`，通过 AST 探索才发现
2. **本地模型 OOM**：Qwen3-Embedding 在 Mac MPS 上 OOM（35 GB attention 矩阵），最终切换到在线 API
3. **DashScope API 批量限制**：遇到 400 错误才知道 batch size 上限是 10
4. **LangChain Prompt 模板坑**：`HumanMessage(content="{context}")` 不插值，导致 LLM 收到字面量
5. **追问检索失效**：多轮对话中追问没有上下文，检索返回无关结果，通过 LLM 改写解决
6. **RAG 幻觉治理**：LLM 在"关系查询"场景下会基于先验知识推理出编造的组件交互关系和假设性代码。根因是检索没有覆盖跨组件证据 + Prompt 缺少 citation enforcement。通过三层防御（Prompt 加固 + LLM 查询分析器 + 多查询检索聚合）解决
7. **固定管线到 Agent 重构**：原 4 步串行管线对每个查询都做完整 Rewrite→Analyze→Retrieve→Generate，追问"画流程图"仍触发无意义检索。重构为 ReAct Agent 后，LLM 自主决定是否调工具——追问直接回答，代码查询按需检索。关键决策是在 ReAct 和 Plan-and-Solve 之间选择了 ReAct，因为代码助手场景不需要复杂分步编排

---

## 十二、快速记忆卡片

| 概念 | 一句话 |
|------|--------|
| RAG | 检索增强生成，先检索再生成，减少幻觉 |
| RRF | 基于排名的融合，解决不同检索系统分数尺度不同的问题 |
| BM25 | TF-IDF 的改进版，加了词频饱和和文档长度归一化 |
| HNSW | 多层图结构的近似最近邻搜索，O(log n) |
| Tree-sitter | 增量式容错解析器，支持多语言 AST |
| ChromaDB | 轻量嵌入式向量数据库，Python 原生 |
| MCP | Model Context Protocol，LLM 工具调用的标准化协议 |
| DashScope | 阿里云 AI 平台，提供嵌入和 LLM 的 OpenAI 兼容接口 |
| ReAct | 推理+行动循环，LLM 自主决定调用工具还是直接回答 |
| `create_agent` | LangChain 新 API，创建 ReAct Agent，底层是 LangGraph CompiledStateGraph |
| ToolMessage | Agent 工具调用的返回消息，包含在对话历史中传递给后续 LLM 调用 |
