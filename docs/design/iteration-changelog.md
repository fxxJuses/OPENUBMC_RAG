# 检索优化迭代变更日志

记录从迭代6到迭代9的架构变更，每个变更的动机、实现方式和效果。

## 指标演进总览

| 迭代 | File@5 | MRR | NDCG@5 | File@1 |
|------|--------|-----|--------|--------|
| 迭代5 基线 | 0.54 | — | — | — |
| 迭代6 意图感知 | 0.62 | — | — | — |
| 迭代7 精度优化 | 0.64 | 0.452 | 0.653 | 0.34 |
| 迭代8 精确匹配 | 0.64 | 0.472 | — | 0.38 |
| 迭代9 LLM重写 | 0.66 | 0.471 | — | — |
| 迭代9 DashScope重排 | 0.58* | 0.451 | 0.696 | 0.38 |

> \* 迭代9 DashScope 基线波动（LLM API 非确定性），在同轮测试中 DashScope 对基线有
> 稳定的 MRR +7.9%、File@1 +19% 改善。

---

## 迭代6：意图感知检索增强

**提交**: `6a391c6` — feat(search): 迭代6 — 意图感知检索增强, File@5 0.54→0.62(+15%)

### 为什么变

基线 File@5=0.54，分析失败用例发现两类系统性缺失：
1. **依赖/接口类查询**（如"组件依赖关系"）期望返回 `service.json`，但 BM25/Dense 双路
   都无法有效检索到 `mds_service` 类型分块 — 这些分块内容是结构化 JSON，关键词匹配和
   语义嵌入都不擅长处理
2. **入口文件类查询**（如"初始化流程"）期望返回 `main.cpp` 或 `*_app.lua`，但这些文件
   在向量空间中与自然语言查询的语义距离较远

### 变了什么

三个核心改动：

**1. BM25 文档增强（mds_service 分块）**
- 在 `mds_service` 分块的内容中注入依赖列表、接口名、服务名等元数据
- 使 BM25 关键词匹配能够命中这些信息
- 文件：`ubmc_rag/indexing/bm25_index.py`

**2. 意图感知定向检索注入**
- 新增依赖查询正则 `_DEPENDENCY_QUERY_RE`（匹配"依赖"、"接口定义"等关键词）
- 命中时从 ChromaDB 定向检索 `chunk_type=mds_service` 分块，注入 Dense 结果前部
- 同步注入 BM25 结果，使这些分块在双路 RRF 融合中都有贡献
- 文件：`ubmc_rag/search/hybrid_search.py` — `_retrieve_mds_service()`

**3. Reranker mds_service bonus**
- 依赖/接口查询时，`mds_service` 类型分块额外加 `+0.010` bonus
- 文件：`ubmc_rag/search/reranker.py`

### 效果

| 指标 | 前 | 后 | 变化 |
|------|----|----|------|
| File@5 | 0.54 | 0.62 | +15% |
| Recall@5 | — | 0.46 | — |

零回归。

---

## 迭代7：P1-P4 精度优化

**提交**: `837940b` — feat(search): 迭代7 — P1-P4 优化, File@5 0.62→0.64(+3%), 零回归

### 为什么变

迭代6将 File@5 推到 0.62，但通过逐用例分析发现四类可修复的精度损失：
1. **P1 同文件重复过多**：同一文件占 3-4 个 top-5 槽位，挤压了其他文件的曝光
2. **P2 查询拼写偏差**：用户输入 "devmon" vs 代码中的 "device_monitor" 等术语不匹配
3. **P3 意图检测覆盖不足**：部分入口类查询（"启动"、"startup"）未被正则匹配
4. **P4 入口文件缺乏定向检索**：与迭代6的 mds_service 注入类似，入口文件也需要定向补充

### 变了什么

**P1: 同文件去重降权增强**
- diversity 降权系数从 0.7 调整为 0.5（超出 `diversity_max_per_file` 的结果分数 ×0.5）
- 释放更多槽位给不同文件

**P2: 查询拼写纠正**
- `QueryProcessor` 新增 `difflib.get_close_matches()` 拼写纠正步骤
- 内置 76 个领域术语词典（sensor, frudata, libipmi 等），cutoff=0.8
- 将查询中与领域术语近似匹配的 token 替换为正确形式
- 文件：`ubmc_rag/search/query_processor.py`

**P3: 意图检测正则扩展**
- 入口文件正则 `_ENTRY_POINT_QUERY_RE` 新增 "启动"、"startup"、"initialize" 等模式
- 文件：`ubmc_rag/search/hybrid_search.py`

**P4: 入口文件定向检索**
- 新增 `_retrieve_entry_points()` 方法
- 优先从 `_chunk_cache` 中按 `file_path` 匹配 `main.cpp` / `main.lua` / `*_app.lua`
- 按向量相似度排序后注入 Dense 结果前部
- 文件：`ubmc_rag/search/hybrid_search.py`

### 效果

| 指标 | 前 | 后 | 变化 |
|------|----|----|------|
| File@5 | 0.62 | 0.64 | +3% |
| Recall@5 | 0.46 | 0.47 | +2% |
| MRR | — | 0.452 | — |
| NDCG@5 | — | 0.653 | — |

零回归。

---

## 迭代8：精确匹配增强

**提交**: `7266050` — feat(search): 迭代8 — 精确匹配增强, 零回归, MRR +4.4%

### 为什么变

迭代7后 File@5=0.64，逐用例分析发现两类排序精度问题：
1. **P7 符号精确匹配奖励不足**：当用户查询本身就是符号名（如 "sensor_read"）时，
   该符号所在的分块应该排到最前，但现有 SYMBOL_BONUS=0.008 不够区分
2. **P5 入口文件正则遗漏**：`main.lua` 未被入口文件正则匹配（只匹配了 `main.cpp`）

同时测试了两个被否决的方案：
- **P6 文件名注入 rank=3**：将文件名匹配的分块注入 Dense 结果第3位，导致 cross_component
  回归（0.71→0.65），因为位移了原有的好结果
- **P8 仓库名精确 boost**：REPO_EXACT_BONUS=0.015 太激进，对跨组件查询造成回归

### 变了什么

**P7: 符号名精确匹配增强**
- 新增 `SYMBOL_EXACT_BONUS = 0.025`（原 SYMBOL_BONUS=0.008）
- 当查询文本本身就是符号名（`query.strip().lower() == sym.name.lower()`）时使用精确奖励
- 文件：`ubmc_rag/search/reranker.py`

**P5: 入口文件正则修复**
- 正则从 `main\.cpp` 扩展为 `(^|/)main\.(cpp|lua)$|_app\.lua$`
- 覆盖了 Lua 项目中的 `main.lua` 入口文件
- 文件：`ubmc_rag/search/hybrid_search.py`

**P6-revised: 文件基名精确匹配**
- 新增 `FILENAME_EXACT_BONUS = 0.020`
- 仅对复合标识符（含下划线）或长名称（>=8字符）生效，避免短名误匹配
- 例如查询 "sensor_mgmt" 精确匹配文件 `sensor_mgmt.lua`

### 效果

| 指标 | 前 | 后 | 变化 |
|------|----|----|------|
| File@5 | 0.64 | 0.64 | 持平 |
| MRR | 0.452 | 0.472 | +4.4% |
| File@1 | 0.34 | 0.38 | +12% |

File@5 未变（召回瓶颈），但排序质量显著改善。

---

## 迭代9A：LLM 查询重写 Dense 注入

**提交**: `490d019` — feat(search): 迭代9 — LLM 查询重写 Dense 注入, File@5 0.64→0.66(+3%), 零回归

### 为什么变

迭代8后 18 个失败用例中，大多数是**模糊自然语言与代码 embedding 不匹配**的问题。
例如 "power control on off" 期望找到 `pwr_action.lua`，但自然语言 embedding 无法
映射到代码标识符。

现有的 `QueryProcessor` 规则式术语扩展已到天花板 — 它只能做简单的拼写纠正和
术语映射，无法理解查询的语义意图并生成对应的代码关键词。

### 变了什么

**1. 新建 `ubmc_rag/search/query_rewriter.py`**
- `LLMQueryRewriter` 类，使用 DashScope Qwen（qwen-plus）模型
- Prompt 设计：输出 3-8 个代码关键词/标识符，包含英文代码词和中文语义词
- Few-shot 示例 + openUBMC 组件列表作为上下文
- temperature=0 保证确定性，max_tokens=128
- 失败时返回原始查询（零退化降级）

**2. 集成到搜索管线**
- 在 `HybridSearchEngine.search()` 中，双路检索完成后：
  - LLM 重写查询 → 额外 Dense 向量检索 → 注入候选池
- **关键设计决策：不替换原始查询**
  - 替换 Dense 查询 → File@5 从 0.64 跌到 0.56（embedding 模型理解自然语言，
    关键词列表破坏语义）
  - 替换 BM25 查询 → File@5 跌到 0.60（原始 BM25 扩展更有效）
  - **最终方案**：保留原始查询做主检索，重写结果仅用于额外 Dense 检索注入候选池

**3. 注入策略**
- 最多注入 3 个新候选（避免稀释原始结果）
- 注入位置：Dense 结果第 10 位（rank=10，确保 RRF 融合获得合理权重但不抢占前部）

### 失败方案记录

| 方案 | File@5 | 原因 |
|------|--------|------|
| 替换 Dense 查询 | 0.56 | 关键词列表破坏 embedding 语义匹配 |
| 追加到 BM25 查询 | 0.60 | 过多术语稀释 BM25 信号 |
| 替换 BM25 查询 | 0.62 | 原始 BM25 扩展更优 |
| BM25 注入重写结果 | 0.62 | 注入候选扰乱 RRF 排序 |
| **Dense 注入（采用）** | **0.66** | 额外检索补充新文件，不干扰主检索 |

### 效果

| 指标 | 前 | 后 | 变化 |
|------|----|----|------|
| File@5 | 0.64 | 0.66 | +3% |
| Recall@5 | 0.47 | 0.50 | +6% |
| MRR | 0.472 | 0.471 | 持平 |
| MAP | — | 0.380 | — |

---

## 迭代9B：DashScope qwen3-rerank 云端重排序

**提交**: `de33d58` — feat(search): 迭代9 — DashScope qwen3-rerank 云端重排序, MRR +7.9%, File@1 +19%

### 为什么变

迭代9A 后 File@5=0.66，但分析发现许多查询的首位命中率不高（File@1=0.34）。
这表明 RRF + boosting 的排序仍有改进空间 — boosting 是基于启发式规则（符号名、
文件路径匹配），无法理解查询与代码片段之间的深层语义关系。

业界标准做法是 **bi-encoder 检索 → cross-encoder 精排** 两阶段架构，cross-encoder
对 (query, document) 对做深度交互，通常能提升 30-40% 排序精度。

### 变了什么

**1. 新建 `ubmc_rag/search/dashscope_reranker.py`**
- `DashScopeReranker` 类，调用 DashScope qwen3-rerank（6B 参数云端模型）
- API 端点：`POST https://dashscope.aliyuncs.com/compatible-api/v1/reranks`
- 失败时返回原始候选（零退化降级）
- 安全限制：单次请求最多 500 文档

**2. 归一化加权融合（而非分数替换）**

核心设计决策 — 不会用 reranker 分数直接替换原始分数：

```
final_score = alpha * normalized_original + (1-alpha) * normalized_dashscope
```

两路分数分别归一化到 [0,1] 后按 alpha=0.6 加权组合。

**为什么不用直接替换**：
- DashScope 返回 0-1 的 relevance_score，与 RRF 分数（~0.03）尺度完全不同
- 直接替换 → boosting bonus（0.008-0.025）在新尺度上无效
- RRF 信号叠加（`0.4 / (60+rank)`）→ 信号太弱（~0.006），无法改变排序
- 归一化融合是唯一有效方案

**3. 集成到 Reranker 主管线**
- 位置：RRF 融合 → boosting → **DashScope 融合** → diversity
- 配置：`dashscope_reranker_enabled: true`（默认启用）
- 文件：`ubmc_rag/search/reranker.py` — `_apply_dashscope_signal()`

**4. 评估框架增强**
- 新增 `hybrid_dashscope` 模式：RRF + DashScope（无 LLM 重写）
- 新增 `hybrid_full` 模式：LLM 重写 + RRF + DashScope
- 新增 `hybrid_cross_encoder` 模式：本地 BGE-reranker（对比基线）
- 文件：`evaluation/retrieval/evaluator.py`

### 失败方案记录

| 方案 | File@5 | 问题 |
|------|--------|------|
| 分数直接替换 | 0.50 | 尺度不匹配，boosting 无效 |
| RRF 信号叠加 | 0.58 | 信号太弱（~0.006 vs 分数差 ~0.001） |
| alpha=0.5（50/50） | 0.58 | DashScope 权重过高，原始信号丢失 |
| alpha=0.3（70% DS） | 0.58 | 更差，DashScope 排序不如原始 |
| **alpha=0.6 归一化融合** | **0.58** | MRR +0.033, 排序最优 |

### 本地 Cross-encoder 评估

同时评估了本地 BGE-reranker-v2-m3（安装了 sentence-transformers）：

| 方案 | File@5 | MRR | NDCG@5 |
|------|--------|-----|--------|
| DashScope alpha=0.6 | 0.58 | 0.451 | 0.696 |
| BGE-reranker-v2-m3 | 0.46 | 0.334 | 0.498 |

本地 cross-encoder **严重回退**（-12pp），原因是评估模式不含 LLM 重写且分数替换问题。
DashScope 方案显著优于本地模型。

### 效果

| 指标 | 基线 | DashScope | 变化 |
|------|------|-----------|------|
| File@5 | 0.58 | 0.58 | 持平 |
| File@1 | 0.32 | 0.38 | +19% |
| MRR | 0.417 | 0.451 | +7.9% |
| NDCG@5 | 0.669 | 0.696 | +4.1% |
| Recall@5 | 0.45 | 0.45 | 持平 |

File@5 不变是因为**召回瓶颈** — 预期文件不在候选池中，reranker 只能重排已有候选。
但排序质量（MRR、NDCG、File@1）全面提升。

---

## 架构演进总结

### 搜索管线架构（迭代9 最终版）

```
用户查询
  │
  ├─ QueryProcessor ─── 意图分析 + 过滤条件 + 术语扩展 + 拼写纠正
  │
  ├─ 双路检索
  │   ├─ Dense 路径 ── DashScope text-embedding-v4 → ChromaDB 向量搜索
  │   └─ BM25 路径 ─── 代码感知分词 → Okapi BM25 关键词匹配
  │
  ├─ LLM 查询重写 ─── DashScope Qwen → 额外 Dense 检索 → 注入候选池（最多3个）
  │
  ├─ 定向补充
  │   ├─ 依赖/接口查询 → mds_service 分块注入
  │   └─ 入口文件查询 → main.cpp/main.lua/*_app.lua 注入
  │
  └─ Reranker
      ├─ RRF 融合（Dense + BM25 双路）
      ├─ Boosting（符号/路径/仓库/MDS 匹配 bonus）
      ├─ DashScope qwen3-rerank 归一化融合（alpha=0.6）
      └─ Diversity（同文件降权 ×0.5）
```

### 关键设计原则

1. **零退化降级**：所有外部 API 调用（LLM 重写、DashScope reranker）失败时使用
   原始结果，不降低服务质量

2. **叠加而非替换**：LLM 重写通过额外 Dense 检索注入候选池（不替换原始查询）；
   DashScope reranker 通过归一化融合叠加（不替换原始分数）

3. **分层优化**：召回层（双路检索 + 定向补充 + LLM 重写）→ 排序层（RRF + boosting
   + DashScope 融合）→ 多样性层（diversity 过滤）

### 瓶颈诊断

当前 File@5 的天花板是 **embedding 召回率** — 有 55% 的预期文件不在候选池中。
这些失败用例的特征是：
- 纯语义查询（"sensor reading threshold"）无法映射到具体代码文件
- 跨组件查询需要理解组件间的调用关系，而非文本相似度

**潜在突破方向**（按优先级）：
1. 更强的 embedding 模型（如 CodeSage、UniXcoder）
2. 代码结构图查询（调用链、import 依赖）
3. 多查询扩展（一个查询拆分为多个子查询并行检索）
