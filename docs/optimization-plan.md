# openUBMC RAG 检索优化方案

## 1. 当前基线

**评估时间**: 2026-06-01, commit `3a5ca27`（路径修复已应用）  
**数据集**: regression_v1, 50 条用例  
**搜索模式**: hybrid_reranked（生产默认）

### 总体指标

| 指标 | 值 | 目标 | 差距 |
|------|-----|------|------|
| File@5 | 56% | 84% | -28pp |
| Recall@5 | 41% | 78% | -37pp |
| File@10 | 56% | 90% | -34pp |
| MRR | 0.406 | >0.7 | -0.294 |
| CategoryHit@5 | 86% | 85% | ✅ 已达标 |
| SymbolHit@5 | 82% | - | - |

### 四模式对比

| 模式 | File@5 | File@10 | Recall@5 |
|------|--------|---------|----------|
| BM25 only | 38% | 50% | 28% |
| Dense only | 40% | 48% | 30% |
| Hybrid (无 rerank) | 48% | 50% | 36% |
| Hybrid + Rerank | **56%** | **56%** | **41%** |

**关键发现**: Reranker 贡献 +8pp File@5，是当前最大的单环节提升。但 File@5→File@10 无增长（56%→56%），说明很多正确文件根本没进 top-10 召回池。

### 按 Category 分解

| Category | Cases | File@5 | Recall@5 | 诊断 |
|----------|-------|--------|----------|------|
| single_component | 14 | 64.3% | 46.4% | 最好，组件名关键词匹配有效 |
| cross_component | 17 | 58.8% | 32.4% | Recall 低——跨组件文件难以全部召回 |
| single_function | 19 | 47.4% | 44.7% | 最差，语义查询和模糊查询易丢失 |

### 按 Difficulty 分解

| Difficulty | Cases | File@5 | Recall@5 | 诊断 |
|------------|-------|--------|----------|------|
| easy | 8 | 62.5% | 62.5% | 精确匹配仍有 37.5% 失败 |
| hard | 22 | 59.1% | 34.1% | 硬查询的 Recall 严重不足 |
| normal | 20 | 50.0% | 40.0% | 最差，语义查询是薄弱环节 |

---

## 2. 失败模式分析

### 模式 A: 召回池不足（最大瓶颈）

**现象**: File@5=56% 但 File@10 仍=56%，说明很多正确结果不在 top-10。

**根因**: 
- 搜索 `top_k * 3` 召回后 RRF 融合，但 top_k=10 时召回池仅 30 条
- BM25 和 Dense 各自 top-30 的重叠率有限
- 跨组件查询需要同时召回 2+ 个不同仓库的文件，单路 top-30 不够

**证据**: 
- cross_component 的 Recall@5 仅 32.4%
- Hybrid(no rerank) File@5=48%, Rerank 后=56%，但 Rerank 无法提升未召回的文件

### 模式 B: single_function 语义查询失败

**现象**: single_function File@5 仅 47.4%，19 条中有 10 条失败。

**根因**:
- 中文语义查询（如"传感器阈值设置的实现"）→ Dense 嵌入不够精准
- BM25 用扩展后的查询（加了 "sensor threshold set" 等），但 BM25 对语义理解有限
- 正确文件可能在 top-10 但不在 top-5（被其他高相关文件挤掉）

**典型案例**:
- TC-007 "SEL event logging function" → 语义泛化查询，BM25 难以匹配
- TC-032 "BMC 启动时如何初始化传感器" → 纯中文长查询
- TC-045 "sensr threshold config" → 拼写错误，BM25 无法容错

### 模式 C: CategoryHit@5=86% 但 File@5=56%，gap=30pp

**现象**: 正确的仓库/组件已找到，但具体文件不对。

**根因**:
- 同一仓库内有大量相似文件（如 sensor 有 600 chunks，power_mgmt 有 1965 chunks）
- 分块粒度过细：method 占 4755/9269（51%），一个文件可能有多个 chunk
- 多样性过滤 max_per_file=3 在 top-10 时过于激进，可能过早降权

### 模式 D: 迭代2 回退教训

**883abf7** (迭代1 峰值): BM25=0.5, Dense=0.5, top_k*3 召回
**4a8d92d** (迭代2 失败): 增大召回到 top_k*5 + content_keyword_boost=1.2

失败原因:
- content_keyword_boost 在 reranker 中对所有含关键词 chunk 做乘法提升，导致高频通用词（如 "data", "service"）误提大量无关结果
- 召回池增大引入了更多噪声，乘法 boost 放大了噪声

**教训**: 乘法 boost 对高频词过于敏感，需要更精细的条件约束。

---

## 3. 优化假设（按优先级排列）

### H1: 增大召回池 + RRF 融合池 [预期: +5~8pp File@5]

**假设**: 将 BM25/Dense 各自召回量从 top_k*3 增大到 top_k*5 或 top_k*8。

**理由**: 
- 当前召回池 30 条（top_k=10 * 3），跨组件查询需要 2+ 个仓库文件
- 迭代2 失败不是因为召回池大，而是因为 reranker 的乘法 boost 放大噪声
- 只增大召回池，不改 reranker，让 RRF 自然融合

**预期效果**: File@5 56%→61-63%, Recall@5 41%→46-49%

**风险**: 迭代2 已证明单改召回池不够，需配合 reranker 优化。但单独增大池不会导致倒退（最差持平）。

### H2: 优化 QueryProcessor 语义扩展 [预期: +3~5pp File@5]

**假设**: 改进查询扩展策略，特别是对纯语义查询和中文查询。

**具体改进**:
1. **结构性查询扩展**: 检测到组件名（如 "sensor", "power_mgmt"）时，自动附加 `mds/service.json` 路径关键词
2. **查询改写**: 对纯中文长查询（如 "BMC 启动时如何初始化传感器"），提取核心术语后构造更精确的搜索词
3. **拼写容错**: 对 BM25 查询做简单的 Levenshtein 距离纠错（如 "sensr" → "sensor"）
4. **查询分类路由**: 不同 query_type 使用不同权重（semantic_match → Dense 权重更高，exact_match → BM25 权重更高）

**理由**: 当前 QueryProcessor 只做字典映射扩展，不理解查询意图。semantic_match 类型 File@5 仅 50%，是最大的失败类别（34 条中 17 条失败）。

**预期效果**: File@5 +3-5pp, 主要改善 normal 和 semantic_match 类型

### H3: Reranker 改用加权加法而非乘法 [预期: +3~6pp File@5]

**假设**: 将 reranker 中的乘法 boost（`score *= boost`）改为加法 bonus（`score += bonus`），避免高频词误提。

**具体改进**:
1. 符号匹配: `score += constant * symbol_match_confidence`（而非 `*= 1.5`）
2. 路径匹配: 精确匹配给固定 bonus，部分匹配按比例衰减
3. 仓库名匹配: 精确匹配给较大 bonus，部分匹配给较小 bonus
4. MDS 类名匹配: 保持较高权重但改用加法

**理由**: 
- 迭代2 证明乘法 boost 对噪声过于敏感
- 加法 bonus 对高分和低分结果的边际效果相同，不会放大噪声
- 精确匹配仍然能有效提升正确结果

**预期效果**: File@5 +3-6pp，特别改善被噪声挤掉的 single_function 查询

### H4: BM25 分词增强——添加路径和元数据到索引内容 [预期: +2~4pp File@5]

**假设**: 在 BM25 索引中，将 file_path、repo_name、symbol_names 作为额外内容拼接到 chunk.content 中一起分词索引。

**理由**: 
- 当前 BM25 只索引 chunk 的源码 content
- 查询 "sensor_database 数据库操作" → BM25 无法匹配到文件名 `sensor_database.lua`
- 文件名和路径中蕴含大量可匹配信号，但 BM25 完全忽略

**预期效果**: File@5 +2-4pp, 主要改善 exact_match 和 mixed 类型

### H5: 自适应 RRF 权重 [预期: +1~3pp File@5]

**假设**: 根据 BM25 和 Dense 各自的置信度动态调整融合权重。

**具体改进**:
1. 如果 BM25 top-1 分数远高于后续（>2x gap），增大 BM25 权重
2. 如果 Dense top-1 距离很小（<0.1 cosine distance），增大 Dense 权重
3. 如果两路结果重叠率高，提升重合结果的权重

**理由**: 
- BM25 File@5=38%, Dense=40%, 差距不大但互补性强
- 某些查询 BM25 明显更强（精确匹配），某些 Dense 更强（语义匹配）
- 固定 0.5/0.5 权重浪费了这种互补性

**预期效果**: File@5 +1-3pp, 改善 hard 查询

### H6: 多样性过滤策略调整 [预期: +1~2pp File@5]

**假设**: 将 diversity_max_per_file 从 3 调为 2，但对跨组件查询放宽。

**理由**:
- 跨组件查询需要多个不同仓库的文件，当前同一仓库内可能占太多位置
- 单组件查询中，同一文件的多个 chunk 可能有价值，不宜过度降权

**预期效果**: Recall@5 +2-3pp, File@5 +1-2pp

---

## 4. 执行计划

### 迭代 3: 召回池 + Reranker 改造

**变量**: 只改召回池大小和 reranker 算法

**步骤**:
1. 增大召回池: `top_k * 3` → `top_k * 5`（在 hybrid_search.py）
2. Reranker 乘法→加法: 所有 `score *= boost` 改为 `score += bonus`
   - symbol_match: `+= 0.15`
   - filepath_match: `+= 0.10`
   - repo_match: `+= 0.10`
   - mds_model_match: `+= 0.20`
3. 运行 eval, 对比基线

**预期**: File@5 56% → 62-65%, Recall@5 41% → 48-52%

### 迭代 4: QueryProcessor 增强

**变量**: 只改查询处理逻辑

**步骤**:
1. 添加结构性扩展: 检测组件名时附加 service.json/model.json 路径关键词
2. 添加查询分类路由: semantic_match 时 Dense 权重 +0.1, BM25 -0.1
3. 添加简单拼写纠错（Levenshtein 距离 ≤1 的已知术语）
4. 运行 eval, 对比迭代3 基线

**预期**: File@5 → 65-68%, Recall@5 → 52-56%

### 迭代 5: BM25 索引增强

**变量**: 只改 BM25 索引内容（需重建索引）

**步骤**:
1. 修改索引构建流程，将 file_path、repo_name、symbol_names 拼入 BM25 文档
2. 重建索引: `uv run ubmc-rag index`
3. 运行 eval, 对比迭代4 基线

**预期**: File@5 → 67-70%, Recall@5 → 55-58%

### 迭代 6: 自适应权重 + 微调

**变量**: 改 RRF 融合策略

**步骤**:
1. 实现 BM25/Dense 置信度检测
2. 动态调整 RRF 权重
3. 调整 diversity 参数
4. 运行 eval, 对比迭代5 基线

**预期**: File@5 → 70-75%, Recall@5 → 58-65%

---

## 5. 回退策略

### 总则
- 每轮迭代只改一个变量（或一组强相关的变量）
- 每轮 eval 后记录完整指标
- 正向（任何指标提升或持平）→ `git commit -m "迭代N: ..."`
- 负向（任何指标下降 >2pp）→ `git revert HEAD`, 记录失败原因，调整参数重试

### 回退阈值
- File@5 下降 >2pp → 必须回退
- File@5 持平但 Recall@5 下降 >3pp → 回退
- File@5 持平、Recall@5 持平 → 检查 CategoryHit@5, 下降 >5pp → 回退

### 索引变更的特殊处理
- 迭代 5 需重建 BM25 索引，耗时较长
- 保留旧索引备份: `cp -r data/index data/index.iter4.bak`
- 回退时恢复: `rm -rf data/index && mv data/index.iter4.bak data/index`

### 不可逆变更
- ChromaDB 向量数据如果需要重刷，需全量 `ubmc-rag index`（约 10 分钟 + API 费用）
- 在迭代 5 之前避免修改向量索引相关逻辑

---

## 附录

### A. 索引统计

```
Total chunks: 9,269

By Repo:
  power_mgmt:   1965 (21.2%)
  bios:         1902 (20.5%)
  mdb_interface: 1372 (14.8%)
  sensor_mgmt:  1120 (12.1%)
  pcie_device:   721 (7.8%)
  sensor:        600 (6.5%)
  fructrl:       473 (5.1%)
  devmon:        367 (4.0%)
  libipmi:       329 (3.5%)
  frudata:       326 (3.5%)
  bus_tools:      83 (0.9%)
  vpd:             9 (0.1%)
  infrastructure:  2 (0.0%)

By Chunk Type:
  method:        4755 (51.3%)
  function:      2473 (26.7%)
  config_block:  1407 (15.2%)
  class:          223 (2.4%)
  mds_ipmi_cmd:   177 (1.9%)
  mds_model:      144 (1.6%)

By Language:
  lua:   5619 (60.6%)
  json:  1758 (19.0%)
  cpp:   1234 (13.3%)
  c:      658 (7.1%)
```

### B. 当前配置快照

```yaml
search:
  rrf_k: 60
  default_top_k: 10
  max_top_k: 50
  bm25_weight: 0.5
  dense_weight: 0.5
  code_query_bm25_boost: 0.15
  symbol_match_boost: 1.5       # 乘法
  filepath_match_boost: 1.3     # 乘法
  mds_model_match_boost: 2.0    # 乘法
  diversity_max_per_file: 3
```

### C. 迭代历史

| 迭代 | Commit | File@5 | Recall@5 | 变更 | 结果 |
|------|--------|--------|----------|------|------|
| 初始 | - | 20% | 15% | 基线 | - |
| 迭代1 | 883abf7 | 52% | 36% | 中英扩展+RRF权重+Reranker增强 | ✅ +32pp |
| 迭代2 | 4a8d92d | - | - | content_keyword_boost + 5x召回池 | ❌ 倒退 |
| 回退 | 636ac78 | ~52% | ~36% | 回退迭代2 | 恢复 |
| 路径修复 | 3a5ca27 | 56% | 41% | 7条路径修复 | ✅ +4pp |
| 迭代3 (H1+H3) | ca9ae76→eff48c8 | — | — | 召回池3x→5x + Reranker乘法→加法bonus | ✅ |
| 迭代3-fix | 9f58005 | — | — | RRF融合后传递top_k*3候选给Reranker | ✅ |
| 迭代3-fix评估 | 7bb8e96 | 58% | 42% | 评估结果 MRR=0.435 | ✅ +2pp |
| 迭代4-H4 | e5f11d7 | — | — | BM25索引增强: file_path+repo+symbols拼入 | ⏳ 待评估 |
| **当前** | **7bb8e96** | **58%** | **42%** | — | 距目标 -26pp/-36pp |

> **注**: 原计划迭代4(H2:QueryProcessor)未执行。迭代4-H4实际对应原计划迭代5(H4:BM25索引增强)，已提前完成。迭代4-H4尚未独立评估（被迭代3-fix评估覆盖）。
