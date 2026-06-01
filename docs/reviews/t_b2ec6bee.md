# QA Review: t_b2ec6bee — 迭代五 RRF-in-Reranker 评估

**Reviewer:** qa  
**Date:** 2026-06-01  
**Commit:** `4d55eeb` (feat(search): 迭代5 — RRF 融入 Reranker)  
**Baseline:** `3688993`  
**Parent task:** t_dfc3f24e  
**Verdict:** ✅ PASS (零回归，等价重构)

---

## 1. 改动概述

将 RRF 融合逻辑从 `HybridSearchEngine._rrf_fuse()` 移入 `Reranker.rrf_fuse()`，使 Reranker 成为统一的融合+排序模块。

### 变更文件 (4 files, +416 -135)

| 文件 | 变更说明 |
|---|---|
| `ubmc_rag/search/reranker.py` | 新增 `rrf_fuse()` 方法；`rerank()` 改为接收 dense/bm25 双路结果；`_apply_boosts()` 和 `_apply_diversity()` 提取为独立方法 |
| `ubmc_rag/search/hybrid_search.py` | `search()` 不再执行 RRF，改为构建 SearchResult 列表后委托 Reranker；新增 `search_raw()` 供评估框架使用；移除 `_rrf_fuse()`/`_get_bm25_weight()`/`_get_dense_weight()` |
| `evaluation/retrieval/evaluator.py` | `_search_hybrid_no_rerank()` 改用 `engine.search_raw()` + `reranker.rrf_fuse()`；移除 `_reconstruct_from_dense_by_id()` |
| `docs/reviews/t_6ac5568e.md` | 前序 review 文档 |

## 2. 自动化测试

```
43 passed in 27.25s
```

全部通过，无失败、无跳过。

## 3. 评估结果 — 总体指标对比

| 指标 | 基线 (3688993) | 当前 (4d55eeb) | Delta | 判定 |
|---|---|---|---|---|
| File@1 | 0.3400 | 0.3400 | 0 | ✅ |
| File@3 | 0.5000 | 0.5000 | 0 | ✅ |
| **File@5** | **0.5800** | **0.5800** | **0** | **✅** |
| File@10 | 0.6200 | 0.6200 | 0 | ✅ |
| Precision@5 | 0.1240 | 0.1240 | 0 | ✅ |
| Recall@5 | 0.4200 | 0.4200 | 0 | ✅ |
| **MRR** | **0.4352** | **0.4352** | **0** | **✅** |
| MAP | 0.3347 | 0.3347 | 0 | ✅ |
| NDCG@5 | 0.6551 | 0.6551 | 0 | ✅ |
| NDCG@10 | 0.8567 | 0.8567 | 0 | ✅ |
| CategoryHit@5 | 0.8800 | 0.8800 | 0 | ✅ |
| SymbolHit@5 | 0.8200 | 0.8200 | 0 | ✅ |

**所有 12 项指标与基线完全一致，零回归。**

## 4. 按 Category 分组

| Category | Cases | File@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| cross_component | 17 | 0.6471 | 0.3529 | 0.4020 | 0.4558 |
| single_component | 14 | 0.6429 | 0.4643 | 0.5281 | 0.7802 |
| single_function | 19 | 0.4737 | 0.4474 | 0.3965 | 0.7413 |

与基线完全一致。

## 5. 按 Difficulty 分组

| Difficulty | Cases | File@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| easy | 8 | 0.6250 | 0.6250 | 0.5417 | 1.2051 |
| normal | 20 | 0.5000 | 0.4000 | 0.4421 | 0.6108 |
| hard | 22 | 0.6364 | 0.3636 | 0.3902 | 0.4954 |

与基线完全一致。

## 6. 代码审查要点

### 正面
- Reranker 成为单一融合+排序入口，职责清晰
- `search_raw()` 提供了评估框架需要的原始双路结果，接口设计合理
- `_apply_boosts()` 和 `_apply_diversity()` 提取为独立方法，可组合使用
- RRF 实现使用 chunk_map 字典去重，逻辑正确

### RRF 算法验证
逐项核验 RRF 公式实现：
- `score(d) = Σ w / (k + rank + 1)` — 公式正确
- 同一 chunk 在双路出现时分数累加 — 已验证（chunk_map 合并逻辑）
- 权重传递链：config → search() 计算 bm25_w/dense_w → rerank() → rrf_fuse() — 完整

### 关注点（非阻塞）
- `search_raw()` 与 `search()` 有部分重复代码（双路检索+构建 SearchResult），后续可考虑抽取公共方法
- `_reconstruct_chunk` 在 `search_raw()` 中仍使用 dense_raw 重建，与 search() 中改用 `CodeChunk.from_chroma_metadata` 的路径不一致，但不影响结果

## 7. 判定

**✅ PASS — 零回归**

判定依据：
- File@5 = 58%，与基线一致（Delta = 0pp，< 2pp 阈值）
- Recall@5 = 42%，无下降
- MRR = 0.4352，无下降
- 43/43 测试通过
- 纯结构性重构，RRF 算法等价实现

按任务判定标准："File@5 持平(±2pp) 且 Recall@5 不降 → ✅ 通过"
