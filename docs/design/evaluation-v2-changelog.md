# 评估框架优化变更记录（2026-05-31）

## 概述

本次优化围绕评估体系展开，修复了 6 个关键 Bug，新增 Precision@K/MAP 等标准指标和 Bootstrap 置信区间，扩展数据集至 50 条用例。检索核心指标 File@5 从初始的 0.28 提升至 0.44（+57%）。

---

## 一、Bug 修复

### 1. recall_at_k 重复计数（metrics.py）

**问题**：`recall_at_k` 统计的是 top-K 中匹配期望文件的结果数量，而非命中的唯一期望文件数量。同一文件的多个 chunk 会被重复计数，导致 recall 可超过 1.0。

**修复**：改为追踪已命中的期望文件集合，去重后再计算比例。

**影响**：所有涉及 recall 的指标回归正确。

### 2. IndexManager.load_index() 未恢复 chunks（index_manager.py）

**问题**：`load_index()` 只加载 BM25 索引文件，不从 ChromaDB 恢复 `_chunks_index`。导致搜索引擎的 `_chunk_cache` 为空，BM25 返回的所有 chunk_id 查找失败，BM25 检索路径返回 0 条结果。

**修复**：新增 `_load_chunks_from_chroma()` 方法，在 `load_index()` 时从 ChromaDB 读取所有文档和元数据，重建 `_chunks_index` 内存索引。

**影响**：BM25 检索从全部返回 0 条恢复到正常工作。这是本次最大的单一修复——BM25 修复前 File@5=0.28，修复后 hybrid_reranked 模式 File@5 跃升至 0.40+。

### 3. ChromaDB 多条件过滤报错（hybrid_search.py）

**问题**：当 `QueryProcessor` 同时提取出 `language` 和 `chunk_type` 两个过滤条件时，ChromaDB 的 `where` 参数包含两个键（如 `{'language': 'json', 'chunk_type': 'mds_ipmi_cmd'}`），触发 `ValueError: Expected where to have exactly one operator`。

**修复**：在 `hybrid_search.py` 的 `search()` 方法中，当 `where` 包含多个键时，用 ChromaDB 的 `$and` 操作符组合条件。

**影响**：恢复了 9 条 JSON 配置文件相关的测试用例（原被跳过）。

### 4. QueryProcessor 过度清洗查询词（query_processor.py）

**问题**：`_clean_query()` 会移除所有 `LANG_KEYWORDS` 中的词，而 `LANG_KEYWORDS["json"]` 包含 "ipmi"。导致查询 "libipmi protocol interface" 被清洗为 "lib protocol interface"，严重影响 BM25 和 Dense 的匹配。

**修复**：将 `_CLEAN_STOPWORDS` 缩减为仅有歧义的短词 `{"json", "mds", "csr", "sr"}`，不再移除 "ipmi"、"function"、"model" 等有区分度的词。

**影响**：涉及 IPMI 相关查询的用例（约占 20%）检索质量显著提升。

### 5. JSON 文件未索引（default_config.yaml）

**问题**：配置文件中 `json.enabled: false`，导致 MDS 模型定义、IPMI 命令描述、服务配置等 384 个 JSON chunk 未被索引。约 19% 的期望文件在索引中不存在。

**修复**：将 `json.enabled` 改为 `true`，并执行全量索引重建（`full_rebuild=True`）。

**影响**：新增 384 个 JSON chunk，覆盖 MDS models、IPMI commands、service definitions。涉及 JSON 文件的查询 File@5 提升约 +8 pp。

### 6. file_path 包含仓库前缀（chunker.py）

**问题**：所有 chunk 的 `file_path` 都带有 `data/repos/{repo}/` 前缀（如 `data/repos/sensor/src/lualib/sensor_service.lua`），导致评估时的路径匹配依赖脆弱的后缀匹配，Reranker 的路径相似度计算也受影响。

**修复**：在 `chunker.parse_repo()` 中，解析完成后统一去除 `data/repos/{repo}/` 前缀，只保留相对仓库的路径。

**影响**：file_path 归一化后，评估匹配更准确，Reranker 的路径 boost 生效。

---

## 二、新增功能

### 1. Precision@K 和 MAP 指标（metrics.py）

新增两个标准信息检索指标：

- **Precision@K**：top-K 中相关结果的比例（按文件去重）
- **MAP (Mean Average Precision)**：所有查询的 Average Precision 均值

`CaseResult` 新增 `precision_at_k: dict[int, float]` 和 `average_precision: float` 字段。
`RetrievalMetrics` 新增 `precision_at_5`, `precision_at_10`, `map_score` 汇总字段。

### 2. 按类别/难度/查询类型分组统计（metrics.py）

`RetrievalMetrics` 新增三个分组字典：

- `by_category: dict[str, RetrievalMetrics]` — 按 single_function / single_component / cross_component
- `by_difficulty: dict[str, RetrievalMetrics]` — 按 easy / normal / hard
- `by_query_type: dict[str, RetrievalMetrics]` — 按 exact_match / semantic_match / fuzzy_match

`compute_metrics()` 内部提取 `_compute_flat_metrics()` 避免递归，然后对每个分组字段调用 `_compute_breakdown()` 单独计算。

### 3. Bootstrap 95% 置信区间（metrics.py）

新增 `compute_confidence_intervals()` 函数：

- 1000 次 bootstrap 重采样（`random.Random(seed=42)` 确保可复现）
- 对 File@5, MRR, MAP, NDCG@5 等核心指标计算 95% CI
- 仅在样本量 >= 5 时计算

### 4. 报告增强（report.py）

- 所有指标行统一在 `_METRIC_ROWS` 列表中定义，避免遗漏
- 新增 `print_breakdown_table()` 输出分组统计
- 置信区间单独输出为 CI 表

### 5. pytest 配置（pyproject.toml）

添加 `[tool.pytest.ini_options] testpaths = ["tests"]`，防止 pytest 误收集 `data/repos/` 下的测试文件。

---

## 三、数据集扩展

`regression_v1.yaml` 从 30 条扩展到 50 条：

| 新增范围 | 编号 | 内容 |
|----------|------|------|
| 边界用例 | TC-031~TC-038 | 超短查询、纯中文、拼写错误 |
| 跨组件深查 | TC-039~TC-044 | VPD→fructrl、mdb→sensor 等调用链 |
| 模糊查询 | TC-045~TC-050 | 拼写变体、泛化术语 |

现在覆盖全部 13 个仓库，难度分布更均衡。

---

## 四、当前检索效果

### 核心指标（hybrid_reranked, 50/50 cases）

| 指标 | 值 |
|------|-----|
| File@1 | 0.2800 |
| File@5 | 0.4400 |
| File@10 | 0.5000 |
| Precision@5 | 0.0920 |
| Precision@10 | 0.0540 |
| Recall@5 | 0.3300 |
| Recall@10 | 0.3800 |
| MRR | 0.3507 |
| MAP | 0.2569 |
| NDCG@5 | 0.5297 |
| NDCG@10 | 0.7040 |
| CategoryHit@5 | 0.7800 |
| SymbolHit@5 | 0.8200 |

### 四模式 A/B 对比

| 模式 | File@5 | MRR | MAP | NDCG@5 |
|------|--------|-----|-----|--------|
| bm25_only | 0.34 | 0.2551 | 0.1703 | 0.3531 |
| dense_only | 0.36 | 0.2445 | 0.1880 | 0.3768 |
| hybrid | 0.40 | 0.2655 | 0.1993 | 0.4323 |
| hybrid_reranked | **0.44** | **0.3507** | **0.2569** | **0.5297** |

### 分组统计摘要

| 类别 | File@5 | MRR |
|------|--------|-----|
| single_function | 0.30 | 0.27 |
| single_component | 0.55 | 0.44 |
| cross_component | 0.47 | 0.34 |

---

## 五、变更文件清单

| 文件 | 改动类型 |
|------|----------|
| `pyproject.toml` | 添加 pytest testpaths |
| `evaluation/retrieval/metrics.py` | 修复 recall bug；新增 Precision@K/MAP/分组统计/Bootstrap CI |
| `evaluation/retrieval/evaluator.py` | 修复 `_reconstruct_from_dense` 补全 symbols |
| `evaluation/agent/evaluator.py` | 复用 IndexManager（load 一次） |
| `evaluation/datasets/regression_v1.yaml` | 30→50 条用例 |
| `evaluation/report.py` | 新增指标行、分组表、CI 表 |
| `tests/test_evaluation/test_retrieval_eval.py` | 新增基线断言和 15 个指标单元测试 |
| `ubmc_rag/indexing/index_manager.py` | `load_index()` 恢复 chunks + `_load_chunks_from_chroma()` |
| `ubmc_rag/search/hybrid_search.py` | ChromaDB `$and` 多条件过滤 + Symbol 反序列化 |
| `ubmc_rag/search/query_processor.py` | 缩减清洗停用词列表 |
| `ubmc_rag/ingestion/chunker.py` | file_path 归一化去前缀 |
| `config/default_config.yaml` | 启用 JSON 解析 |

---

## 六、后续优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| P0 | Reranker 调优（当前 hybrid 0.40 → hybrid_reranked 0.44，提升有限） | File@5 +5~10 pp |
| P1 | BM25 元数据过滤（用 repo_name 缩小搜索范围） | 精确查询 +10 pp |
| P1 | 多查询检索（语义查询拆分为多个子查询） | cross_component +15 pp |
| P2 | 代码符号分词优化（下划线/驼峰拆分） | single_function +10 pp |
| P2 | CI 集成（eval 指标低于基线则构建失败） | 防止检索质量退化 |
