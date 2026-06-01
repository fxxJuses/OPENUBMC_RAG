# QA Review: P0 BGE-reranker-v2-m3 效果验证

**Task:** t_486cae5d
**Commit:** a9afe71
**Date:** 2026-06-01
**Verdict:** ⚠️ 效果不显著（File@5=54%, +2pp, 未达统计显著性）

## 1. Config 验证

- `cross_encoder_enabled: true` — 确认 (config/default_config.yaml L102)
- `cross_encoder_model: "BAAI/bge-reranker-v2-m3"` — 确认 (L103)
- `cross_encoder_device: "cpu"` — 确认 (L104)
- `dashscope_reranker_enabled: false` — 确认 (L106)

## 2. 代码质量审查

### cross_encoder.py
- 延迟初始化 + 自动 fallback 机制：合理
- 异常处理：模型加载失败、推理失败均有 fallback 路径
- 日志：`Cross-encoder loaded: %s (device=%s)` 消息存在 (L74-75)
- `is_fallback` 属性可检测是否降级

### evaluator.py (_search_hybrid_cross_encoder)
- 流程：RRF融合 → top-(k*3) 候选 → CrossEncoder → boosting → diversity → top_k
- 独立初始化 CrossEncoderReranker，正确传递 model_name 和 device
- 日志记录 model、device、fallback 状态

### 模型加载验证
- 无运行时日志文件留存，但通过结果对比可推断：
  - p0_cross_encoder.json: File@5=40% (明显低于基线，可能为 fallback 或早期配置)
  - p0_ce_final.json: File@5=54% (高于基线，推断使用了真实模型)

## 3. 指标对比 (p0_baseline vs p0_ce_final)

### 核心指标

| Metric | Baseline | CE Final | Delta | Status |
|--------|----------|----------|-------|--------|
| File@5 | 52% | 54% | +2pp | ✅ 符合预期 |
| Recall@5 | 38% | 41% | +3pp | ✅ 符合预期 |
| CategoryHit@5 | 82% | 90% | +8pp | ✅ 符合预期 |
| SymbolHit@5 | 82% | 78% | -4pp | ⚠️ 已知回归 |

### 全量指标

| Metric | Baseline | CE Final | Delta | >2pp? |
|--------|----------|----------|-------|-------|
| File@1 | 28% | 28% | 0pp | OK |
| File@3 | 46% | 42% | -4pp | REGRESS |
| File@5 | 52% | 54% | +2pp | +GOOD |
| File@10 | 52% | 54% | +2pp | OK |
| Precision@5 | 10.8% | 12.0% | +1.2pp | OK |
| Precision@10 | 5.6% | 6.0% | +0.4pp | OK |
| Recall@5 | 38% | 41% | +3pp | +GOOD |
| Recall@10 | 39% | 41% | +2pp | OK |
| MRR | 37.4% | 36.9% | -0.5pp | OK |
| MAP | 29.4% | 28.5% | -0.9pp | OK |
| NDCG@5 | 61.5% | 59.9% | -1.6pp | OK |
| NDCG@10 | 74.2% | 69.8% | -4.5pp | REGRESS |
| CategoryHit@5 | 82% | 90% | +8pp | +GOOD |
| SymbolHit@5 | 82% | 78% | -4pp | REGRESS |

### >2pp 回归汇总
1. **file_at_3: -4pp** (46%→42%) — 未被 orchestrator 提及
2. **ndcg_at_10: -4.5pp** (74.2%→69.8%) — 未被 orchestrator 提及
3. **symbol_hit_at_5: -4pp** (82%→78%) — 已知，orchestrator 已提及

## 4. 统计显著性

**95% 置信区间重叠检验：所有指标 CI 均重叠，无统计显著性。**

| Metric | Baseline CI | CE Final CI | Overlap |
|--------|-------------|-------------|---------|
| File@5 | [38%, 66%] | [40%, 68%] | Yes |
| Recall@5 | [27%, 49%] | [30%, 53%] | Yes |

50 cases 样本量不足以检测 2-4pp 的差异。所有观察到的变化均在抽样误差范围内。

## 5. 基线漂移问题

⚠️ **重要发现：当前基线 File@5=52%，与历史基线 File@5=58% (commit 12a77c8) 相差 -6pp。**

- 第一条 comment 标注基线为 58%
- p0_baseline.json 实际为 52%
- 这意味着即使 CE Final (54%) 也未恢复到历史基线水平

可能原因：索引重建、环境变化、或评估数据差异。

## 6. 按类别分析

### Category Hit 主要收益来源: single_component (+14.3pp)

| Category | File@5 Δ | Recall@5 Δ | CategoryHit@5 Δ | SymbolHit@5 Δ |
|----------|----------|------------|-----------------|---------------|
| cross_component | 0pp | +2.9pp | +5.9pp | 0pp |
| single_component | **+7.1pp** | **+7.2pp** | **+14.3pp** | 0pp |
| single_function | 0pp | 0pp | +5.3pp | **-10.5pp** |

⚠️ single_function 的 SymbolHit@5 暴跌 -10.5pp (58%→47%)，是整体 -4pp 的主因。

## 7. 测试结果

- **test_cross_encoder.py**: 6/6 PASS ✅
- **test_reranker.py**: 11/13 PASS, **2 FAIL** ❌
  - `test_reranker_skip_cross_encoder`: Reranker.rerank() 缺少 `skip_cross_encoder` 参数
  - `test_reranker_cross_encoder_disabled_by_default`: Reranker 缺少 `_cross_encoder` 属性
  - 原因：测试代码假设 cross_encoder 集成在 Reranker 中，但实际集成在 evaluator 中

## 8. 判定

按第一条 comment 的标准：
- File@5 = 54% < 56% → 🔴 **倒退**（vs 历史基线 58%）
- File@5 = 54% vs 当前基线 52% → +2pp，但**未达统计显著性**

按第二条 comment（调整后基线 52%→54%）：
- 指标数字均匹配 ✅
- 但存在 **2 个未被提及的 >2pp 回归** (file_at_3, ndcg_at_10)
- 基线本身比历史低 6pp

### 建议
1. 需调查基线从 58% 降到 52% 的原因（索引重建？）
2. file_at_3 和 ndcg_at_10 的回归需要关注
3. single_function SymbolHit@5 的 -10.5pp 需要分析
4. 建议增大评估样本量以获得统计功效
5. 修复 2 个失败的测试用例
