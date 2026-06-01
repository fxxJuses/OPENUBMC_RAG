# QA Review: 迭代六-B — P3 分词修复 + DashScope rerank

- **Task**: t_1b7bda68
- **Dev Task**: t_60c19a2c
- **Commit**: 6351ba2
- **Reviewer**: qa
- **Date**: 2026-06-01
- **Verdict**: ⚠️ PARTIAL — P3 fix 合格，DashScope rerank 回退

---

## 1. 代码审查

### 1.1 P3 分词器简化 (bm25_index.py)

**结论**: ✅ PASS

- 移除了 `preserve_composites` 逻辑和 `_DOMAIN_MULTI_WORD` 字典
- 只保留子 token（camelCase/snake_case 拆分后的小写部分）
- 领域词典 `_DOMAIN_DICTIONARY` 保留完好（IPMI/SEL/SDR/FRU 等 30+ 术语）
- 正则 `_TOKENIZE_RE` 未变（迭代6-P3 修复版）
- BM25 索引从 8.2MB → 8.1MB（-1.2%），符合去复合 token 预期
- 与上一轮迭代 15.9MB 膨胀问题彻底解决

### 1.2 DashScope reranker (dashscope_reranker.py)

**结论**: ✅ 代码质量合格

- API 调用: `requests.post` + 指数退避重试（2次）
- 限流: `_MIN_INTERVAL = 0.1s` 最小调用间隔
- 降级: API 不可用/无 key/HTTP 错误均 graceful fallback 返回原始结果
- 超时: 30s HTTP timeout
- 文档截断: `_MAX_CHARS_PER_DOC = 4000`
- 懒加载: 通过 `_dashscope_init_attempted` 标志避免重复初始化

### 1.3 Reranker 集成 (reranker.py)

**结论**: ✅ 集成正确

- 新增 step 4: DashScope rerank，在 cross-encoder 之后、diversity 之前
- `skip_dashscope_reranker` 参数支持跳过
- `dashscope_reranker_enabled` 配置开关，默认 false

### 1.4 配置 (settings.py + default_config.yaml)

- `dashscope_reranker_enabled: bool = False`（默认关闭）
- `dashscope_reranker_model: str = "qwen3-rerank"`
- `dashscope_reranker_top_n: int = 20`
- ⚠️ 注意: default_config.yaml 中 `dashscope_reranker_enabled: true`（dev 测试后未改回）

### 1.5 单元测试

- **74/74 tests passed** (含 11 DashScope reranker tests + 10 BM25 tests)
- 覆盖: API 成功/失败/空结果/无 key/网络错误/HTTP 错误/工厂函数
- Mock 方式: `unittest.mock.patch("requests.post")` — 合理

---

## 2. 独立评估结果

### 2.1 BM25 索引

- 已使用简化分词器重建（MD5: 8ab3a874 匹配 dev 的 6b_simplified 版本）
- 索引大小: 8.1MB（旧 8.2MB，-1.2%）
- 文档数: 9269（不变，因为 chunk 数量未变）

### 2.2 指标对比表

| 指标 | 基线 (12eb002) | P3 fix only | P3 + DashScope | Δ P3 vs 基线 | Δ DashScope vs 基线 |
|---|---|---|---|---|---|
| **File@5** | **58%** | **56%** | **50%** | -2pp | **-8pp** |
| File@1 | 38% | 36% | 40% | -2pp | +2pp |
| File@3 | 52% | 52% | 44% | 0pp | -8pp |
| File@10 | 60% | 58% | 58% | -2pp | -2pp |
| Recall@5 | 42% | 41% | 38% | -1pp | -4pp |
| Recall@10 | 44% | 44% | 46% | 0pp | +2pp |
| MRR | 0.4352 | 0.4375 | 0.4409 | +0.0023 | +0.0057 |
| MAP | 0.3359 | 0.3359 | 0.3387 | 0 | +0.0028 |
| NDCG@5 | 0.6551 | 0.6697 | 0.6115 | +0.0146 | -0.0436 |
| CategoryHit@5 | 88% | 88% | 86% | 0 | -2pp |
| SymbolHit@5 | 82% | 82% | 82% | 0 | 0 |

### 2.3 分类别分析 (P3 fix only vs 基线)

| Category | 基线 File@5 | P3 File@5 | Δ |
|---|---|---|---|
| cross_component | 64.7% | 64.7% | 0 |
| single_component | 57.1% | 57.1% | 0 |
| single_function | 47.4% | 47.4% | 0 |

### 2.4 分类别分析 (P3 + DashScope)

| Category | P3 only | P3 + DashScope | Δ |
|---|---|---|---|
| cross_component | 64.7% | 64.7% | 0 |
| single_component | 57.1% | 50.0% | **-7.1pp** |
| single_function | 47.4% | 36.8% | **-10.5pp** |

---

## 3. 判定

### P3 分词修复: ⚠️ 小幅下降 (-2pp)，在 CI 内

- File@5 = 56%（基线 58%，-2pp）
- 95% CI: [42%, 70%] — -2pp 完全在统计噪声内
- 按类别: 三类全部零变化（与基线完全一致）
- BM25 索引大小正常（8.1MB vs 旧的 8.2MB）
- **结论**: 去复合 token 修复正确解决了 15.9MB 膨胀问题，性能与基线持平

### DashScope rerank: ❌ 回退 (-8pp)

- File@5 = 50%（基线 58%，-8pp）
- 超过 -2pp 阈值，判定为回退
- 主要退化在 `single_component` (-7.1pp) 和 `single_function` (-10.5pp)
- `cross_component` 未受影响（保持 64.7%）
- MRR 微涨 (+0.006)，说明 top-1 稍好，但整体排序质量下降
- **根因推测**: DashScope reranker 对代码搜索的语义匹配不如 RRF+boosting 的启发式规则，可能是因为 qwen3-rerank 训练语料以自然语言为主，对代码标识符/函数名的理解不如符号名精确匹配

---

## 4. 建议

1. **P3 fix**: 可合入。解决了 15.9MB 索引膨胀，性能在 CI 内持平。
2. **DashScope rerank**: 不建议默认启用。当前 `default_config.yaml` 中 `dashscope_reranker_enabled: true` 需要改回 `false`。
3. DashScope rerank 可作为实验性功能保留，但需要：
   - 调整 pipeline 位置（可能应该在 boosting 之前或与 boosting 融合，而非完全替代 boosting 分数）
   - 或者仅对 natural language 查询启用，代码查询跳过
   - 收集更多评测数据后重新评估

---

## 5. 审查清单

- [x] P3 分词器正确移除复合 token 保留逻辑
- [x] 领域词典完好
- [x] BM25 索引大小正常（8.1MB，非 15.9MB 膨胀版）
- [x] DashScope reranker 代码质量合格（重试/降级/限流）
- [x] Reranker pipeline 集成正确（step 4 位置）
- [x] 74/74 单元测试通过
- [x] P3 fix 独立评估: File@5=56%（-2pp，CI 内）
- [ ] DashScope rerank 独立评估: File@5=50%（-8pp，**回退**）
- [ ] default_config.yaml dashscope_reranker_enabled 需改回 false
