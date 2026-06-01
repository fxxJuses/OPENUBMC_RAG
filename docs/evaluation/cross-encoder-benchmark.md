# BGE-reranker-v2-m3 交叉编码器效果评估报告

> 评估日期: 2026-06-01
> 数据集: regression_v1.yaml (50 cases, 10805 chunks)
> 模型: BAAI/bge-reranker-v2-m3 (568M params, CPU 推理)
> 评估者: dev-backend (Hermes Kanban)

## 摘要

**结论：BGE-reranker-v2-m3 交叉编码器对 openUBMC 代码检索任务有负面效果，File@5 从 52% 降至 40%（-12 pp），MRR 从 0.374 降至 0.304（-19%）。不推荐启用。**

## 评估方法

对比两种搜索模式的检索指标：

| 模式 | 流程 |
|------|------|
| `hybrid_reranked` (基线) | Dense+BM25 → RRF → Boosting → Diversity → Top-K |
| `hybrid_cross_encoder` | Dense+BM25 → RRF → **Cross-Encoder** → Boosting → Diversity → Top-K |

交叉编码器管道：RRF 融合后取 top-30 候选送入 BGE-reranker-v2-m3 做深度语义评分，然后应用 boosting 和 diversity。

## 核心指标对比

| 指标 | 基线 (hybrid_reranked) | 交叉编码器 (hybrid_cross_encoder) | 变化 |
|------|----------------------|----------------------------------|------|
| **File@5** | **0.5200** | **0.4000** | **-0.1200** ❌ |
| File@10 | 0.5200 | 0.5400 | +0.0200 |
| **MRR** | **0.3740** | **0.3035** | **-0.0705** ❌ |
| Recall@5 | 0.3800 | 0.3200 | -0.0600 ❌ |
| Recall@10 | 0.3900 | **0.4500** | **+0.0600** ✅ |
| NDCG@5 | 0.6150 | 0.4859 | -0.1291 ❌ |
| NDCG@10 | 0.7424 | 0.7005 | -0.0419 |
| Precision@5 | 0.1080 | 0.0920 | -0.0160 |
| MAP | 0.2937 | 0.2478 | -0.0459 ❌ |
| CategoryHit@5 | 0.8200 | **0.8600** | **+0.0400** ✅ |
| SymbolHit@5 | 0.8200 | 0.7800 | -0.0400 |

## 按查询类别拆解

| 类别 | 基线 File@5 | 交叉编码器 File@5 | 变化 | 评估 |
|------|------------|-------------------|------|------|
| single_component (14) | 0.5000 | **0.5714** | **+0.0714** | ✅ 提升 |
| single_function (19) | **0.4737** | 0.3158 | **-0.1579** | ❌ 严重退化 |
| cross_component (17) | **0.5882** | 0.3529 | **-0.2353** | ❌ 严重退化 |

## 关键发现

### 1. 交叉编码器对代码符号查询有害 (single_function: -16pp)

精确函数名查询（如 `get_sensor_data`、`power_management_init`）高度依赖 BM25 关键词匹配和 boosting 规则（符号名精确匹配奖励）。交叉编码器的纯语义评分无法区分 `getSensorData` 和 `getFanData` 这样的相似但不同的代码符号，导致 ranking 质量下降。

### 2. 交叉组件查询严重退化 (cross_component: -24pp)

跨组件语义查询（如 `sensor和power_mgmt关系`）是三类中最难的。交叉编码器的通用语义理解不足以捕获组件间的调用关系和架构模式，反而破坏了 RRF 融合的多信号平衡。

### 3. 单组件查询有提升 (single_component: +7pp)

对整体组件功能查询（如 `sensor组件的依赖关系`、`pcie_device加载`），语义理解有帮助。但提升幅度不大（7pp），不足以弥补其他类别的损失。

### 4. Recall@10 有提升但代价高昂

Recall@10 从 0.39 提升到 0.45 (+6pp)，说明交叉编码器确实召回了一些新文件。但这些文件排在前 6-10 名而非前 5 名，且 MRR 下降说明排序质量变差。

### 5. 推理成本高

- 单个查询：~1.5s (30 candidate pairs × ~50ms/pair, CPU)
- 50 条用例完整评估：~10 分钟
- 相比基线：推理开销增加 20x

## 根因分析

BGE-reranker-v2-m3 是通用文本重排序模型，预训练数据以自然语言 QA 对为主，不包含代码检索信号。对于 openUBMC 代码检索场景：

1. **信号差异**：代码检索的有效信号是符号名匹配、文件路径匹配、API 调用链，而非自然语言的句子级语义相似度
2. **score 归一化问题**：交叉编码器将所有候选映射到 [0, 1] 区间，丢失了 boosting 所需的分数动态范围，导致后续的符号/路径匹配 bonus 被稀释
3. **领域不匹配**：BGE-reranker 在 MTEB 基准上表现优秀，但这些基准以 Wikipedia/StackExchange/NQ 为主，不包含代码检索任务

## 建议

1. **不启用 BGE-reranker-v2-m3**：对当前检索任务有明确的负面效果
2. **探索代码专用重排序模型**：如 CodeBERT-based reranker、或基于代码 AST 相似度的重排序
3. **增强 boosting 规则**：当前 boosting（符号/路径/仓库匹配）已经证明有效，可以继续优化
4. **考虑轻量级语义重排序**：如使用本地 embedding 的余弦相似度（已在 dense_only 中验证），而非交叉编码器
5. **保留代码以备将来使用**：`cross_encoder.py` 模块保留但维持 `cross_encoder_enabled: False` 默认值，为未来代码专用重排序模型提供集成框架

## 附录：执行命令

```bash
# 基线评估
ubmc-rag eval retrieval -m hybrid_reranked -v

# 交叉编码器评估
ubmc-rag eval retrieval -m hybrid_cross_encoder -v

# 全模式对比（含交叉编码器）
ubmc-rag eval retrieval -m all -v -o results.json
```
