# 评估框架优化变更记录（2026-05-31 v3）

## 概述

本次优化将 MDS JSON 配置分块从代码分块中分离，建立独立的双 Collection 架构（`openubmc_code` + `openubmc_mds`），解决了 JSON 配置块在检索结果中淹没源代码的核心问题。检索核心指标 File@5 从 0.44 提升至 0.48（+9%），MRR 从 0.35 提升至 0.38（+10%）。

---

## 一、问题背景

### MDS JSON 淹没问题

在 v2 优化后，384 个 MDS JSON 配置分块（`mds/ipmi.json`、`mds/model.json`、`mds/service.json`）与 7511 个代码分块共用一个 ChromaDB collection 和一个 BM25 索引。MDS JSON 的语义覆盖面极广——几乎与所有 IPMI/传感器/组件相关查询都有高相似度——导致：

- 查询 `ipmi_get_sensor_reading` 的 top-10 全部是各仓库的 `mds/ipmi.json`，期望的 C 头文件排在后面
- 50 个评估用例中约 10 个直接因此失败
- File@5 停滞在 0.44，无法进一步提升

### 分析结论

27/50 失败用例的根因分布：

| 根因 | 影响用例数 | 失败率 |
|------|-----------|--------|
| MDS JSON 淹没 | ~10 | 最大单一因素 |
| sensor vs sensor_mgmt 混淆 | ~5 | 次大因素 |
| mdb_interface 未索引 | 2 | 100% 失败 |
| vpd_service.lua 未索引 | 1 | 100% 失败 |
| 近似文件名竞争 | ~5 | 中等 |
| 语义查询能力不足 | ~6 | 长期问题 |

---

## 二、架构变更：双 Collection 分离

### 核心思路

将所有 MDS JSON 配置分块路由到独立的 ChromaDB collection 和 BM25 索引，代码检索默认只搜索代码集合。MDS 结果通过加权融合机制按需引入，避免淹没源代码。

### 数据流变更

```
旧：Chunker → all chunks → IndexManager (1 collection + 1 BM25)
新：Chunker → code chunks → IndexManager.vector_store (openubmc_code) + bm25
                  mds chunks → IndexManager.vector_store_mds (openubmc_mds) + bm25_mds
```

### 搜索流程变更

```
旧：Dense(全量) + BM25(全量) → RRF → Rerank
新：Dense(code) + BM25(code) → RRF → code_fused
     Dense(mds) + BM25(mds) → RRF → mds_fused × 0.5 weight
     合并 code_fused + mds_fused → Rerank
```

---

## 三、详细变更

### 1. Config 配置新增

**`config/settings.py`**:
- `IndexingConfig.chroma_collection_mds: str = "openubmc_mds"` — MDS 集合名称
- `SearchConfig.mds_result_weight: float = 0.5` — MDS 结果加权系数

**`config/default_config.yaml`**:
```yaml
indexing:
  chroma_collection_mds: "openubmc_mds"
search:
  mds_result_weight: 0.5
```

### 2. Model — MDS chunk 类型判断

**`ubmc_rag/models/code_chunk.py`**:
- 新增 `MDS_CHUNK_TYPES` 常量：`{"mds_service", "mds_model", "mds_ipmi_cmd", "mds_type_def", "csr_topology", "csr_object", "config_block"}`
- 新增 `CodeChunk.is_mds` 属性：判断 chunk_type 是否属于 MDS 配置

### 3. VectorStore — 支持指定 collection name

**`ubmc_rag/indexing/vector_store.py`**:
- 构造函数新增 `collection_name: str | None` 参数，可覆盖 `config.chroma_collection`
- `reset()` 使用实例级 `_collection_name` 而非 config 字段

### 4. IndexManager — 双集合管理

**`ubmc_rag/indexing/index_manager.py`**（重写）:

构造函数：
- `self.vector_store` — 代码集合（`openubmc_code`）
- `self.vector_store_mds` — MDS 集合（`openubmc_mds`）
- `self.bm25` + `self.bm25_mds` — 两组 BM25 索引

`build_index()`:
- 按 `is_mds` 拆分 chunks 为 code_chunks（7511）和 mds_chunks（384）
- 分别构建 BM25 → `bm25_index.json` / `bm25_index_mds.json`
- 分批 embed 时路由到对应 VectorStore

`load_index()`:
- 加载两个 BM25 索引文件
- 从两个 ChromaDB collection 恢复 chunks 到内存

`get_stats()`:
- 返回 `code_chunks`、`mds_chunks`、`chroma_code_count`、`chroma_mds_count` 等分项统计

### 5. HybridSearchEngine — 双路检索 + MDS 加权

**`ubmc_rag/search/hybrid_search.py`**:

构造函数：
- 新增 `vector_store_mds: Optional[VectorStore]` 和 `bm25_mds: Optional[BM25Index]`

`search()`:
- 新增 `include_mds: bool = True` 参数
- **默认行为**：搜索代码集合 + MDS 集合（加权 0.5）
- **`include_mds=False`**：仅搜索代码集合
- 新增 `_search_mds()` 方法：对 MDS 集合独立执行 dense + BM25 → RRF
- MDS 结果分数乘以 `mds_result_weight` 后与代码结果合并

### 6. 调用方更新

**`ubmc_rag/cli/search_cmd.py`**:
- 初始化时传入 `vector_store_mds` 和 `bm25_mds`
- 新增 `--include-mds/--no-include-mds` CLI 选项（默认 True）

**`ubmc_rag/mcp_server/server.py`**:
- 初始化时传入 MDS 组件
- `search_code` 工具新增 `include_mds` 参数

**`ubmc_rag/chat/tools.py`**:
- `search_code` 工具新增 `include_mds` 参数

**`ubmc_rag/chat/retriever.py`**:
- 初始化时传入 MDS 组件

**`ubmc_rag/chat/chain.py`**:
- 补充注入 MDS 组件到 retriever engine

**`evaluation/retrieval/evaluator.py`**:
- 初始化时传入 MDS 组件
- `evaluate()` 新增 `include_mds` 参数（默认 True）

**`ubmc_rag/cli/index_cmd.py`**:
- 更新 stats 输出格式，显示 code/mds 分项统计

### 7. 测试基线更新

**`tests/test_evaluation/test_retrieval_eval.py`**:
- 更新基线阈值以匹配新指标

---

## 四、检索效果对比

### 核心指标（hybrid_reranked, 50/50 cases）

| 指标 | v2（单集合） | v3（双集合加权） | 变化 |
|------|-------------|-----------------|------|
| File@1 | 0.2800 | **0.3200** | +14% |
| File@5 | 0.4400 | **0.4800** | +9% |
| File@10 | 0.5000 | **0.5400** | +8% |
| Precision@5 | 0.0920 | **0.1000** | +9% |
| Recall@5 | 0.3300 | **0.3600** | +9% |
| MRR | 0.3507 | **0.3845** | +10% |
| MAP | 0.2569 | **0.2803** | +9% |
| NDCG@5 | 0.5297 | **0.5598** | +6% |
| NDCG@10 | 0.7040 | **0.7169** | +2% |
| CategoryHit@5 | 0.7800 | **0.8400** | +8% |
| SymbolHit@5 | 0.8200 | 0.8200 | — |

### 三种模式 A/B 对比

| 模式 | File@5 | MRR | MAP | NDCG@5 |
|------|--------|-----|-----|--------|
| Code-only（不搜 MDS） | 0.40 | 0.3045 | 0.2203 | 0.4582 |
| **双集合加权 0.5x（默认）** | **0.48** | **0.3845** | **0.2803** | **0.5598** |
| 旧单集合（v2 baseline） | 0.44 | 0.3507 | 0.2569 | 0.5297 |

### 分组统计

| 类别 | File@5 | MRR |
|------|--------|-----|
| single_function | 0.42 | 0.30 |
| single_component | 0.50 | 0.43 |
| cross_component | 0.53 | 0.44 |

| 难度 | File@5 | MRR |
|------|--------|-----|
| easy | 0.63 | 0.43 |
| normal | 0.40 | 0.34 |
| hard | 0.50 | 0.41 |

---

## 五、变更文件清单

| 文件 | 改动类型 |
|------|----------|
| `config/default_config.yaml` | 新增 chroma_collection_mds、mds_result_weight |
| `ubmc_rag/config/settings.py` | IndexingConfig/SearchConfig 新增字段 |
| `ubmc_rag/models/code_chunk.py` | 新增 MDS_CHUNK_TYPES 常量和 is_mds 属性 |
| `ubmc_rag/indexing/vector_store.py` | 构造函数支持 collection_name 覆盖 |
| `ubmc_rag/indexing/index_manager.py` | 重写：双集合管理、双 BM25、分批路由 |
| `ubmc_rag/search/hybrid_search.py` | 新增 MDS 双路检索和加权融合 |
| `ubmc_rag/cli/index_cmd.py` | 更新 stats 输出格式 |
| `ubmc_rag/cli/search_cmd.py` | 传入 MDS 组件 + --include-mds 选项 |
| `ubmc_rag/mcp_server/server.py` | 传入 MDS 组件 + include_mds 参数 |
| `ubmc_rag/chat/tools.py` | include_mds 参数 |
| `ubmc_rag/chat/retriever.py` | 传入 MDS 组件 |
| `ubmc_rag/chat/chain.py` | 补充注入 MDS 组件 |
| `evaluation/retrieval/evaluator.py` | 传入 MDS 组件 + include_mds 参数 |
| `tests/test_evaluation/test_retrieval_eval.py` | 更新基线阈值 |

---

## 六、索引数据分布

| 集合 | ChromaDB | BM25 | 说明 |
|------|----------|------|------|
| `openubmc_code` | 7,511 | 7,511 | Lua/C/C++ 源代码 |
| `openubmc_mds` | 384 | 384 | MDS JSON 配置 |
| **总计** | **7,895** | **7,895** | |

---

## 七、后续优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| P0 | 修复 mdb_interface 索引（0 chunks） | 2 个用例 +4 pp |
| P0 | 修复 vpd_service.lua 未索引 | 1 个用例 +2 pp |
| P1 | Reranker 增加 repo 级别区分（sensor/sensor_mgmt） | 4~5 个用例 +8~10 pp |
| P1 | 语义查询增强（中文翻译/关键词扩展） | 5~6 个用例 +10 pp |
| P1 | 调优 mds_result_weight（当前 0.5，可尝试 0.3~0.7） | 精细调优 +2~3 pp |
| P2 | BM25 元数据过滤（repo_name 缩小范围） | 精确查询 +5 pp |
| P2 | 多查询检索（语义查询拆分子查询） | cross_component +10 pp |
