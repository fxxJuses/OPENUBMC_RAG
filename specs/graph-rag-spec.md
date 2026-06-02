# Graph RAG 实现规格

## 背景

当前 RAG 系统的检索管线是 chunk 级别的 BM25 + Dense → RRF fusion → 规则 Reranker。
对跨组件查询（cross_component）Recall@5 仅 29.4%，是所有类别中最弱的。
根本原因：管线没有任何代码结构关系的感知能力——`service.json` 声明的依赖、`require()` 调用、
`#include` 引用等信息已经解析但只存为 metadata，检索时从未遍历这些连接。

Graph RAG 通过构建代码知识图并利用图遍历扩展检索上下文，直接解决这个问题。

## 学术验证

DKB 论文 (arXiv 2601.08773) 对比了三种代码图检索方式：

| 方法 | 多跳准确率 | 构建时间 | 成本倍数 |
|------|-----------|---------|---------|
| 无图（baseline） | 6/15 (40%) | N/A | 1x |
| **AST-derived graph** | **15/15 (100%)** | **2-14s** | **~2x** |
| LLM-extracted graph | 13/15 (87%) | 200-884s | ~20-45x |

结论：纯 AST 提取（不用 LLM）是最优方案——最快、最准、最便宜。

---

## 图数据模型

### 节点类型

| 节点类型 | ID 规约 | 属性 | 来源 |
|---------|---------|------|------|
| `component` | `{repo}` | name, languages, file_count | 每个仓库一个 |
| `file` | `{repo}:{rel_path}` | path, language, repo_name | 每个源文件一个 |
| `entity` | `{repo}:{rel_path}:{name}` | name, kind, language, chunk_id, signature | 从 AST 提取的函数/类/方法 |
| `interface` | `iface:{name}` | name, provider | service.json 提供的接口 |

### 边类型

| 边类型 | 源 → 目标 | 含义 | 提取方式 |
|-------|----------|------|---------|
| `DEPENDS_ON` | component → component | 组件依赖 | service.json dependencies |
| `CONTAINS` | file → entity | 文件包含实体 | AST 父子关系 |
| `BELONGS_TO` | file → component | 文件属于组件 | repo_name 映射 |
| `DEFINES` | component → entity | 组件定义实体 | 聚合 CONTAINS |
| `IMPORTS` | entity → component | require()/#include | Lua require(), C #include |
| `CALLS` | entity → entity | 函数调用 | call_expression 匹配 |
| `REQUIRES_IFACE` | component → interface | 需要接口 | service.json required |
| `PROVIDES_IFACE` | component → interface | 提供接口 | service.json provided |
| `USES_MDS_MODEL` | entity → entity | 使用 MDS 模型 | model.json 引用 |

### 不建模的内容

- 单个变量赋值（粒度过细）
- 文件内关系（chunk 级检索已覆盖）
- 运行时调用链（需动态分析）

---

## 技术选型

### 存储：NetworkX DiGraph

- 11 个仓库、~9K chunks → 估计 3K-4K 实体节点、5K-8K 条边
- NetworkX 处理此规模毫无压力（内存 < 50MB）
- 序列化为 JSON 存于 `data/index/knowledge_graph.json`
- 无外部依赖（不需要 Docker、Neo4j）
- 未来可扩展：设计 `GraphStore` 抽象层，需要时可替换为 Neo4j/Memgraph

### 构建方式：两阶段 AST 提取（不用 LLM）

**Pass 1 — 发现所有节点**：遍历所有 CodeChunk，注册所有 entity/file/component 节点
**Pass 2 — 解析所有边**：在完整节点表基础上，解析 require()/#include/调用目标

---

## 检索集成

### 三路 RRF 融合

```
当前: score(d) = dense_w/(k+rank_d) + bm25_w/(k+rank_b)
新增: score(d) = dense_w/(k+rank_d) + bm25_w/(k+rank_b) + graph_w/(k+rank_g)
```

Graph 作为第三路检索加入 RRF，保持现有管线不变。

### 图遍历策略：双向扩展

从检索命中的 seed 节点出发：
1. **Successors（下游）**：该实体依赖什么？沿 CALLS/IMPORTS/DEPENDS_ON 边扩展
2. **Predecessors（上游）**：什么依赖该实体？反向沿相同边扩展
3. **Interface-Consumer Expansion**：当遇到 IMPLEMENTS/PROVIDES_IFACE 边时，找到所有实现同一接口的其他组件（对等方）

### Reranker 图邻接加分

在 Reranker 中新增 `_apply_graph_bonus()`：如果候选 chunk 在图中与 query seed 节点相邻，给予额外加分。

---

## 实现阶段

### Phase 1: 图数据层（不影响检索）

新建 `ubmc_rag/graph/` 包，构建图并持久化，但暂不接入检索管线。

**新建文件**：
- `ubmc_rag/graph/__init__.py`
- `ubmc_rag/graph/schema.py` — 节点/边类型定义
- `ubmc_rag/graph/builder.py` — 两阶段图构建（从 CodeChunk 列表）
- `ubmc_rag/graph/store.py` — NetworkX 存储（load/save/query）

**修改文件**：
- `ubmc_rag/config/settings.py` — 新增 `GraphConfig`
- `ubmc_rag/indexing/index_manager.py` — build_index 中构建图、load_index 中加载图
- `pyproject.toml` — 新增 `networkx>=3.0` 依赖
- `config/default_config.yaml` — 新增 graph 配置段

**验证**：运行 `ubmc-rag index`，检查 `data/index/knowledge_graph.json` 是否生成，
验证节点数、边数是否符合预期。

### Phase 2: 图检索路径

将图遍历接入检索管线作为第三路。

**新建文件**：
- `ubmc_rag/graph/expander.py` — 双向扩展 + 接口消费者扩展

**修改文件**：
- `ubmc_rag/search/hybrid_search.py` — 接受 KnowledgeGraph，新增 graph retrieval path
- `ubmc_rag/search/reranker.py` — RRF 扩展为三路融合
- `ubmc_rag/cli/search_cmd.py` — 传入 graph
- `ubmc_rag/mcp_server/server.py` — 传入 graph
- `ubmc_rag/chat/retriever.py` — 传入 graph

**验证**：`ubmc-rag eval retrieval --mode all` 对比 hybrid vs hybrid_graph。

### Phase 3: Reranker 图加分

在 Reranker 中新增图邻接加分。

**修改文件**：
- `ubmc_rag/search/reranker.py` — 新增 `_apply_graph_bonus()`

### Phase 4: 新工具和资源

- MCP Server 新增 `explore_component_graph` 工具
- Chat Agent 新增 `explore_component_relationships` 工具

---

## GraphConfig 配置

```yaml
graph:
  enabled: true
  persist_path: ""              # 默认 data/index/knowledge_graph.json
  max_hops: 2                   # 最大遍历深度
  graph_weight: 0.20            # 图检索在 RRF 中的权重
  edge_type_weights:
    DEPENDS_ON: 1.0
    IMPORTS: 0.8
    CALLS: 0.7
    DEFINES: 0.5
    REQUIRES_IFACE: 0.9
    PROVIDES_IFACE: 0.9
  graph_adjacency_bonus: 0.005  # reranker 图邻接加分
```

---

## 风险控制

1. **完全叠加式**：graph.enabled = false 时管线与当前完全一致
2. **保守权重**：graph_weight 初始 0.20（vs BM25 0.50, Dense 0.50），避免引入回归
3. **仅用加分不用乘分**：graph_adjacency_bonus 为加法，遵循 H3 迭代教训
4. **评估护轨**：每个 Phase 完成后跑全量评估，File@5 下降 >2pp 则回退
5. **无 LLM 依赖**：图构建纯 AST，2-14 秒完成，无额外 API 成本

---

## 预期收益

| 查询类别 | 当前 Recall@5 | 预期 Recall@5 | 提升来源 |
|---------|--------------|--------------|---------|
| cross_component | 29.4% | ~40-45% | DEPENDS_ON/IMPORTS 边直接发现关联组件 |
| semantic_match | 32.4% | ~38-42% | 图扩展补充 BM25/Dense 漏掉的语义相关代码 |
| 整体 | 40.0% | ~45-50% | 三路融合 + 图邻接加分 |

---

## Tree-sitter 查询模板

### Lua — 关系提取

```python
# require() 调用 → IMPORTS 边
LUA_REQUIRE_QUERY = """
(function_call
    name: (identifier) @_fn
    (#eq? @_fn "require")
    arguments: (arguments
        (string_content) @module_path))
"""

# 函数调用 → CALLS 边
LUA_CALL_QUERY = """
(function_call
    name: (identifier) @callee)
"""
```

### C/C++ — 关系提取

```python
# #include → IMPORTS 边
C_INCLUDE_QUERY = """
(preproc_include
    path: (string_literal) @include_path)
"""

# 函数调用 → CALLS 边
C_CALL_QUERY = """
(call_expression
    function: (identifier) @callee)
"""

# 方法调用 → CALLS 边
C_METHOD_CALL_QUERY = """
(call_expression
    function: (field_expression
        field: (field_identifier) @callee))
"""

# 继承 → INHERITS 边
C_INHERIT_QUERY = """
(class_specifier
    base: (base_class_clause
        (type_identifier) @base_name))
"""
```

### JSON — 关系提取

直接从已有的 `chunk.metadata` 提取：
- `mds_service` chunks → `metadata.dependencies` → DEPENDS_ON 边
- `mds_service` chunks → `metadata.required_interfaces` → REQUIRES_IFACE 边
- `mds_model` chunks → `metadata.mds_class` → USES_MDS_MODEL 边
