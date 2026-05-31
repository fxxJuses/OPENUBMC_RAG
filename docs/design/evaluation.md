# 评估框架设计与现状

> **更新记录**：2026-05-31 完成评估框架 V2 优化，详见 [evaluation-v2-changelog.md](evaluation-v2-changelog.md)。核心变更：修复 6 个 Bug（BM25 返回 0.0、ChromaDB 过滤报错等）、新增 Precision@K/MAP/Bootstrap CI 指标、数据集扩展至 50 条、File@5 从 0.28 提升至 0.44。

## 概述

评估框架对标日志分析 Agent 平台的 Evals 体系，为 openUBMC Code RAG 系统提供检索质量和 Agent 回答质量的自动化评测能力。

## 目录结构

```
evaluation/
├── datasets/
│   ├── schema.py              # Pydantic V2 数据集模型（ExpectedFile, TestCase, RegressionDataset）
│   ├── loader.py              # YAML 加载 + 校验
│   └── regression_v1.yaml     # 30 条回归测试用例
├── retrieval/
│   ├── metrics.py             # 6 项检索指标（File@K, Recall@K, MRR, NDCG, CategoryHit, SymbolHit）
│   ├── evaluator.py           # 检索评估器（支持 bm25_only/dense_only/hybrid/hybrid_reranked 4 种模式）
│   └── runner.py              # 多模式 A/B 对比 Runner
├── agent/
│   ├── prompts.py             # 四层评分 Prompt + 权重（方案质量 0.30 → 定位 0.25 → 完整性 0.25 → 证据 0.20）
│   ├── judge.py               # LLM-as-Judge（模型隔离：Qwen 回答 + GLM 评分）
│   └── evaluator.py           # Agent 评估器（运行 Agent + Judge）
├── report.py                  # Rich 表格 + JSON 导出 + before/after 对比
└── eval_cmd.py                # CLI：ubmc-rag eval {retrieval|agent|all}
```

## CLI 命令

```bash
ubmc-rag eval retrieval --mode hybrid_reranked          # 单模式评测
ubmc-rag eval retrieval --mode all -o results.json      # 四模式 A/B 对比
ubmc-rag eval retrieval --mode bm25_only -v             # 仅 BM25
ubmc-rag eval agent --judge-model glm-4-flash           # Agent 回答评估
ubmc-rag eval all                                        # 完整评估套件
```

## 数据集设计

### regression_v1.yaml（30 条用例）

| 类别 | 数量 | 查询示例 | 特点 |
|------|------|----------|------|
| single_function | 10 | `get_sensor_data`、`传感器阈值设置` | 精确符号名、语义查询、中英混杂 |
| single_component | 10 | `sensor组件的依赖关系`、`pcie_device loading` | 组件依赖、加载流程、整体逻辑 |
| cross_component | 10 | `sensor和power_mgmt关系`、`IPMI 完整路径` | 组件交互、调用链路 |

每条用例标注 `expected_files`（repo_name + file_path + relevance 等级）、`expected_symbols`、`category`、`difficulty`。

### 匹配策略

索引中的 `file_path` 含 `data/repos/` 前缀（如 `data/repos/sensor/src/lualib/sensor_service.lua`），数据集使用相对路径（`src/lualib/sensor_service.lua`）。评估时：
1. 先精确匹配 `repo_name:file_path`
2. 不命中则做后缀匹配（`result_path.endswith(expected_path)`）

## 评估指标

### 检索指标（retrieval/metrics.py）

| 指标 | 公式 | 说明 |
|------|------|------|
| File@K | top-K 含任一 expected file → 1 | 文件级命中率 |
| Recall@K | top-K 命中数 / 总 expected 数 | 召回率 |
| MRR | 1 / first_relevant_rank | 首个相关结果排名倒数 |
| NDCG@K | DCG@K / IDCG@K，使用 relevance 等级 | 排序质量 |
| CategoryHit@K | top-K 含任一 expected repo → 1 | 仓库级命中 |
| SymbolHit@K | top-K 含任一 expected symbol → 1 | 符号级命中 |

### Agent 指标（agent/prompts.py）

四层加权评分，阈值 6.0/10 判 pass：

| 维度 | 权重 | 评估内容 |
|------|------|----------|
| solution_quality | 0.30 | 回答是否准确解决查询，代码引用是否正确 |
| localization | 0.25 | 文件路径/行号是否准确 |
| completeness (5W1H) | 0.25 | What/Where/Why/How/Who/When 覆盖 |
| evidence_reliability | 0.20 | 每个论断是否有来源标注，是否无编造 |

---

## 当前实测结果（2026-05-30）

### 检索指标总览

| 指标 | 简历目标 | 当前实测 | 差距 |
|------|---------|---------|------|
| File@5 | 84% | **59%** | **-25 pp** |
| CategoryHit@5 | 85% | **91%** | **+6 pp** ✅ |
| SymbolHit@5 | — | 73% | — |
| MRR | — | 0.55 | — |

> 22/30 用例成功评估，8 条因 ChromaDB where 多键过滤 bug 被跳过。

### 按查询类别拆解

| 类别 | 评估数 | File@5 | File@10 | MRR | CategoryHit@5 | SymbolHit@5 |
|------|--------|--------|---------|-----|---------------|-------------|
| single_function | 6/10 | 50% | 50% | 0.50 | 100% | 17% |
| single_component | 9/10 | **78%** | 78% | **0.78** | 100% | 89% |
| cross_component | 7/10 | 43% | 43% | 0.29 | 71% | 100% |

### 关键发现

1. **CategoryHit 反超简历目标**：91% > 85%，搜索引擎在「定位到正确组件」上表现优秀
2. **single_component 最好**：File@5 = 78%，接近简历目标的 84%
3. **single_function 最差**：精确符号名查询（如 `get_sensor_data`）File@5 只有 50%，BM25 对代码符号的分词质量不足
4. **cross_component 有挑战**：跨组件语义查询（如 `sensor和power_mgmt关系`）File@5 = 43%，embedding 对技术领域语义理解有限

---

## 待解决问题

### 🔴 P0：修复阻断性问题

#### 1. ChromaDB where 多键过滤 bug

**现象**：当 `QueryProcessor` 同时提取出 `language` 和 `chunk_type` 过滤条件时，ChromaDB 的 `where` 参数包含两个键（如 `{'language': 'json', 'chunk_type': 'mds_ipmi_cmd'}`），触发 `ValueError: Expected where to have exactly one operator`。

**影响**：8/30（27%）回归用例直接崩溃跳过，且这些多为 JSON 配置文件查询（ipmi.json、model.json），本应是比较容易拿分的场景。

**修复方案**：在 `hybrid_search.py` 的 `search()` 方法中，当 `where` 包含多个键时，用 ChromaDB 的 `$and` 操作符组合条件：

```python
# 当前（有问题）：
where = {"language": "json", "chunk_type": "mds_ipmi_cmd"}

# 修复为：
where = {"$and": [
    {"language": "json"},
    {"chunk_type": "mds_ipmi_cmd"},
]}
```

**文件**：`ubmc_rag/search/hybrid_search.py:96-106`

**预期收益**：恢复 8 条用例，File@5 预估提升 +10~15 pp。

#### 2. Recall/NDCG 重复计数 bug

**现象**：`Recall@K` 和 `NDCG@K` 出现 > 1.0 的异常值（如 Recall@5 = 1.43、NDCG@5 = 1.00）。

**原因**：后缀匹配时，多个搜索结果可能匹配同一个 expected file（如 `sensor/sensor_service.lua` 和 `sensor_mgmt/sensor_service.cpp` 都以 `sensor_service.lua` 后缀匹配不到，但某些路径确实会重复匹配同一 expected key），导致同一 expected file 被重复计数。

**修复方案**：在 `recall_at_k` 和 `ndcg_at_k` 中，维护已匹配的 expected key 集合，避免重复计数：

```python
def recall_at_k(results, expected, k):
    relevant = _get_relevant_set(expected)
    matched = set()
    for r in results[:k]:
        rkey = _result_key(r)
        for ek in relevant:
            if ek not in matched and _is_match(rkey, {ek}):
                matched.add(ek)
    return len(matched) / len(relevant)
```

**文件**：`evaluation/retrieval/metrics.py`

### 🟡 P1：检索质量提升

#### 3. 代码符号分词优化

**现状**：`get_sensor_data` 这种精确函数名查询，BM25 需要将驼峰/下划线命名拆分为 token（`get`, `sensor`, `data`），否则整串作为关键词难以匹配。

**方案**：在 `bm25_index.py` 的分词器中增加：
- 下划线拆分（`get_sensor_data` → `get sensor data`）
- 驼峰拆分（`getSensorData` → `get sensor data`）
- 保留原始 token 用于精确匹配

**预期收益**：single_function 类别 File@5 预估 +15~20 pp。

#### 4. 中英技术术语双向扩展

**简历数据**：查询扩展对中英混杂 query 效果 **+9 pp**。

**现状**：`query_processor.py` 已有基本的查询处理，但缺少中英术语映射表。

**方案**：在 `query_processor.py` 或新建 `query_expander.py` 中添加：
- 固定术语映射：`热插拔 ↔ hotplug`、`传感器 ↔ sensor`、`阈值 ↔ threshold`
- 双向扩展：查询同时包含中文和英文术语的变体

**预期收益**：中英混杂查询（TC-008, TC-009, TC-023 等）效果提升。

#### 5. 跨组件查询的 query 改写 / 多查询

**现状**：cross_component 类别 File@5 = 43%，是三类中最低的。跨组件查询（如 `sensor和power_mgmt关系`）的语义太宽泛，单次检索难以覆盖多个组件。

**方案**：
- 检测跨组件意图，将查询拆分为多个子查询分别检索
- 或使用 LLM query analyzer 生成多个检索 query

**预期收益**：cross_component File@5 提升。

### 🟢 P2：评估框架增强

#### 6. pytest 基线值动态调整

**现状**：基线值硬编码在测试中，修复 P0 问题后需要手动更新。

**方案**：首次运行时记录基线到 `evaluation/baselines.json`，后续测试与文件中的基线对比。

#### 7. Agent 评估实际运行验证

**现状**：Agent 评估模块代码已完成，但尚未实际运行（需要 DashScope API 产生 LLM 调用费用）。

**方案**：先用 `--max-cases 3` 小规模运行，验证四层评分 Prompt 的有效性和 GLM Judge 的输出质量。

#### 8. CI 集成

**方案**：在 CI pipeline 中加入 `ubmc-rag eval retrieval --mode hybrid_reranked -o eval_report.json`，JSON 输出供 CI 解析，指标低于基线则构建失败。
