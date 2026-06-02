# SE分析: P0复盘 + 基线漂移 + Graph RAG评估 + 全新方向

> 生成日期: 2026-06-01 (Graph RAG 评估更新)
> 评估人: SE Agent (Hermes)
> 任务: t_97f3878c
> 基于: 代码审计(完整代码库) + P0评估数据 + QA review + web_search调研 + Graph RAG 集成评估

---

## 〇、Graph RAG Phase 2 评估结果

### 结论: Graph RAG 三路融合对检索质量零贡献

| 指标 | Baseline (无Graph) | Graph RAG (enabled) | 变化 |
|------|-------------------|---------------------|------|
| File@5 | 0.54 | 0.54 | 0 |
| Recall@5 | 0.39 | 0.39 | 0 |
| MRR | 0.41 | 0.41 | 0 |
| File@10 | 0.60 | 0.60 | 0 |

### 根因分析

1. **RRF 分数叠加效应**: Dense+BM25 双路命中的 chunk RRF 分数叠加（0.01-0.015），远高于 Graph 单路结果（0.0016）。Graph-only 结果排在 RRF rank 56+，无法进入 top 10。

2. **扩展结果质量不匹配**: DEPENDS_ON 边从 bios → fructrl/pcie_device，但沿 DEFINES 拉入的 entity 几乎全是 mds/model.json 定义项。测试数据集的跨组件答案通常在其他组件的 Lua/C++ 代码中，MDS model 定义与查询语义相关性极低。

3. **DEPENDS_ON 边太少（仅 9 条）**: 图构建器只从 service.json 解析组件依赖，很多组件没有 service.json。而 CALLS 边虽有 3351 条跨组件的，但沿 CALLS 扩展到的目标函数已在 dense/bm25 结果中。

4. **已尝试的优化（均无效）**:
   - Entity seed 提取修复（去除 sym.kind 过滤，symbol 从 1→3 个）
   - Component→Entity 桥接（非 seed 组件拉入 5 个 entity）
   - Graph 注入策略（`_fuse_with_graph` bonus + 跨组件追加）
   - graph_weight 调参（0.20 → 0.10）

### Graph RAG 价值保留

虽然三路融合无效，但图构建和扩展基础设施仍有价值：
- 图数据（8352 节点、31516 边、948 条跨组件 CALLS）可用于 MCP 工具和 Chat Agent
- Phase 3（Reranker graph adjacency bonus）和 Phase 4（MCP/Chat 图探索工具）可作为后续增强
- 当前保留 graph.enabled=true，对检索无副作用

---

## 一、基线漂移: File@5 58% → 52% (-6pp) 根因

### 证据链

| 时间 | 事件 | 索引状态 |
|------|------|----------|
| 10:07 | 迭代6-B P3fix实验: 索引重建 | pre_6b_backup: 908K tokens (旧) |
| 10:10 | P3fix简化版索引重建 | 6b_simplified: 927K tokens (+2.1%) |
| 10:16 | 当前索引生成 | bm25_index.json: 927K tokens (同simplified) |
| 10:36 | SE终审报告(12a77c8): 声称58% | **使用历史数据，非实测** |
| 12:14 | P0 eval: ChromaDB重建 | chroma.sqlite3 更新 |
| 12:15 | P0 eval: baseline实测**52%** | a9afe71 |

### 根因

**BM25索引在迭代6-B实验中重建后未被恢复。** 

3d512b0的revert只回退了代码，但索引文件(10:16生成，含+19K tokens的token膨胀)留在了磁盘上。当P0 eval在12:15实测baseline时，使用的是"被污染"的索引，导致File@5从58%降至52%。

**量化影响**:
- BM25 token总量: 908K → 927K (+19K, +2.1%)
- 平均chunk token数: 98.0 → 100.1 (+2.1%)
- BM25长度归一化惩罚随之增大 → 关键词检索精度下降 → RRF融合后整体退化

**修复方案**: 恢复pre_6b_backup索引(908K tokens)或从3d512b0干净重建索引，预计可恢复52%→58%。

---

## 二、P0复盘: BGE-reranker-v2-m3 为何仅 +2pp

### 2.1 按类别分解: 收益分布极不均衡

| 类别 | 占比 | 基线 | CE | Delta | 贡献 |
|------|------|------|-----|-------|------|
| cross_component | 34% (17) | 58.8% | 58.8% | **0pp** | 0pp |
| single_component | 28% (14) | 50.0% | 57.1% | **+7.1pp** | +2.0pp |
| single_function | 38% (19) | 47.4% | 47.4% | **0pp** | 0pp |
| **整体** | 100% (50) | 52% | 54% | **+2pp** | +2pp |

### 2.2 三大根因

#### 根因1: 交叉编码器候选池过小（召回瓶颈 > 排序瓶颈）

当前管道: RRF融合 → top-15候选 → CE重排 → boosting → diversity → top-5

**cross_component的34%用例File@5零提升**，说明正确答案不在RRF top-15中。交叉编码器只能重排已有候选，不能凭空找回遗漏的chunk。

**数据佐证**: cross_component Recall@5基线仅32.4%，即使CE提升2.9pp也才35.3%。问题在**召回端**，不在排序端。

#### 根因2: 分数体系混搭（Boosting在CE分数上失效）

```python
# evaluator.py _search_hybrid_cross_encoder:
ce_reranked = self._cross_encoder.rerank(query, ce_candidates)  # 分数范围: 0~1
boosted = self.engine.reranker._apply_boosts(ce_reranked, query)  # bonus: 0.006~0.012
```

**致命问题**: CE产生的分数(0~1)与RRF的boosting bonus(0.006~0.012)差了**两个数量级**。在CE分数上加0.008的符号匹配bonus完全无效——相当于什么都没做。

这解释了为什么cross_component的CategoryHit@5能提升5.9pp(CE模型本身对类别判断更准)，但File@5零提升(Boosting无法帮助排序)。

#### 根因3: BGE-reranker-v2-m3不是代码专用模型

**single_function SymbolHit@5暴跌-10.5pp (58%→47%)** 是最直接的证据。BGE交叉编码器不理解代码符号语义——它对函数名、变量名的判断比BM25/Dense还要差。

BGE-reranker-v2-m3是基于XLM-RoBERTa的通用跨语言模型，训练数据以自然语言为主。代码中的符号匹配(如`getSensorReading` vs `sensor_reading`)需要的是**标识符级别的理解**，CE模型不具备。

### 2.3 为什么学术预期(+5~12pp)落空

1. 学术论文的reranker评估用的是**通用检索数据集**(BEIR/MS MARCO)，文档是维基百科段落，不是代码
2. 代码检索的正确率瓶颈在**召回**(cross_component Recall@5=32%)，不在**排序**。CE解决的是排序问题，不是召回问题
3. 工业实践中，两阶段管道(BM25+Dense→CE)的提升主要来自**候选池足够大**(top-50~100)，而不是top-15

---

## 三、全新方向建议

> **排除清单**: H2(查询权重路由 -8pp) / BGE(CE仅+2pp) / P3(代码分词 -10pp) / DashScope(qwen3-rerank -14pp)

### 方向A [P0-立即]: 召回池扩大 + 候选去重 + 分数重整

**核心洞察**: 当前瓶颈在召回，不在排序。cross_component的正确答案根本不在top-15候选池中。

**方案**:
1. 召回池从top_k*3=15扩大到top_k*10=50
2. Dense和BM25各自召回50，合并去重
3. 重新设计分数归一化: BM25分数min-max归一化到[0,1]; Dense cosine距离线性映射到[0,1]; 两路分数做线性加权融合(替代RRF)
4. 在50个候选上应用boosting→diversity→取top-5

**预期收益**: +8~15pp File@5（主要是cross_component的提升）
**成本**: 零新依赖，纯架构调整
**风险**: 召回池扩大会增加延迟(BM25查50条 ~5ms, Dense查50条 ~20ms, 总计可接受)
**证据**: cross_component Recall@5仅32% vs 整体42%——召回不足是主因

### 方向B [P0-立即]: LLM查询分解 (针对cross_component)

**核心洞察**: cross_component占34%用例，是最大单一失败类别。这类查询(`sensor如何通过I2C读取温度`)本质上是多跳问题，单次检索无法覆盖所有相关文件。

**方案**:
1. 用DashScope qwen-turbo(低成本，~$0.001/调用)分解查询
2. 例: `sensor如何通过I2C读取温度` → [`sensor temperature reading`, `I2C bus communication`, `sensor I2C interface`]
3. 每个子查询独立检索，合并去重
4. Dasher/multi-query-fusion模式

**预期收益**: cross_component File@5 58% → 70-75% (+12~17pp单类别, 整体+4~6pp)
**成本**: API调用费(50个query × 2轮eval × ~$0.001 = $0.1), 延迟+200ms
**风险**: 低(失败也只是退化到单查询，不回退)
**证据**: Query decomposition是2025-2026 RAG领域最有效的技术之一(LevelRAG, arXiv:2502.18139)

### 方向C [P1-本周]: FlashRank轻量重排器替代BGE

**核心洞察**: BGE失败不在于交叉编码器概念，而在于(1)分数混搭 (2)候选池太小。换一个更轻更快的重排器，在50候选池上用，效果会完全不同。

**方案**:
1. 安装FlashRank: `uv pip install flashrank`
2. 使用`ms-marco-MiniLM-L-6-v2`模型(CPU推理 <30ms/query, 对比BGE的150-200ms)
3. 在top-50候选上运行CE → 保留CE分数作为排序依据
4. **移除RRF boosting**: CE分数已是最终排序，不再叠加规则bonus

**预期收益**: +3~7pp File@5 (前提:配合方向A的50候选池)
**成本**: 零API费用，模型下载一次(~80MB)
**风险**: 低
**证据**: FlashRank在MS MARCO上NDCG@10提升25%，CPU推理比BGE快5倍

### 方向D [P2-本月]: AST符号图增强检索

**核心洞察**: single_function的SymbolHit@5暴跌-10.5pp证明CE不懂代码符号。需要结构化的符号关系来增强检索。

**方案**:
1. 利用已有的Tree-sitter解析结果，构建符号调用图(symbol_calls, symbol_called_by)
2. 当查询中提到函数名时，从符号图中查找caller/callee
3. 将关联符号所在chunk加入候选池
4. 加权: 直接调用者+0.5 bonus, 间接调用者+0.25 bonus

**预期收益**: single_function SymbolHit@5恢复+提升+5~10pp, File@5 +2~4pp
**成本**: 构建符号图(<10s, 可用已有AST数据), 检索时O(1)查找
**风险**: 中(需要通过web_search确认当前Tree-sitter解析粒度是否足够)
**证据**: Tree-Sitter-Based Knowledge Graphs (arXiv:2603.27277, 2026年3月)，CodeGraph的"结构感知搜索"已被证明有效

### 方向E [P2-本月]: 分数体系彻底重整

**核心洞察**: 当前分数体系有三个不可调和的矛盾:
1. BM25分数(BM25公式，无界) vs Dense分数(cos距离，[-1,1])
2. RRF rank融合(丢失分数幅度) vs 学术最佳实践CC融合(保留分数幅度)
3. Boosting bonus(0.006~0.012) vs CE分数(0~1)

**方案**:
1. BM25分数 → min-max归一化到[0,1] (在50候选池内)
2. Dense分数 → (distance + 1) / 2 映射到[0,1]
3. 线性加权融合: final = α*BM25_norm + (1-α)*Dense_norm
4. 在50条标注数据上网格搜索α
5. 50候选 → CE重排(如果启用) → diversity → top-5
6. **完全废弃RRF+boosting bonus**

**预期收益**: +3~5pp File@5 (分数体系合理化的系统收益)
**成本**: 重构reranker.py (~200行代码), 网格搜索α(<1分钟)
**风险**: 中(需要确保新体系不退化,50候选池上grid search验证)
**证据**: ACM SIGIR 2023证实CC优于RRF 3-5% NDCG; 50条标注数据足以拟合单一α参数

### 方向F [P3-下季度]: 领域微调小交叉编码器

**核心洞察**: BGE是通用模型。如果有50~200条(query, positive_chunk, negative_chunk)三元组，可以微调一个小模型(如ModernBERT-base, 149M)专门适配openBMC代码域。

**方案**:
1. 从50条评估数据生成训练样本(正例:正确chunk, 负例:随机+hard negative)
2. 使用sentence-transformers训练API微调CrossEncoder
3. 目标: ModernBERT-base-149M, 推理<50ms on CPU

**预期收益**: +8~15pp File@5 (领域适配的核心收益)
**成本**: 微调30min on MPS GPU, 需扩充训练数据(50→200条，可LLM生成hard negatives)
**风险**: 中高(需要足够的训练数据，50条可能不够)
**证据**: LoRA微调在MS MARCO上可将reranker精度提升10-15%，数据效率远高于从头训练

---

## 四、优先级路线图

```
当前:  RRF + boosting              [52% File@5] (实际基线, 恢复索引可达58%)
+A:   召回池50 + 去重              [60-65%]
+B:   LLM查询分解                  [64-70%]
+E:   分数重整(线性CC)             [68-75%]
+C:   FlashRank重排(on 50候选)     [72-80%]
+D:   AST符号图增强                [74-82%]
+F:   领域微调CE                    [80-88%]

目标: 84% File@5, 78% Recall@5
```

### 执行顺序

| 优先级 | 方向 | 时间 | 预期收益 | 依赖 |
|--------|------|------|----------|------|
| **立即** | A: 召回池扩大 | 2h | +8~15pp | 无 |
| **立即** | B: LLM查询分解 | 2h | +4~6pp | 无 |
| **本周** | E: 分数重整 | 0.5天 | +3~5pp | A完成 |
| **本周** | C: FlashRank | 1h | +3~7pp | A+E完成 |
| **本月** | D: AST符号图 | 2天 | +2~4pp | A完成 |
| **下季度** | F: 领域微调CE | 1周 | +8~15pp | 需扩充训练数据 |

### 为什么这个顺序

1. **A和B互不依赖，可以并行**: A解决召回量的问题，B解决召回质的问题
2. **E必须在C之前**: 分数重整后，FlashRank才能正常工作(C分数的0~1不与boosting混搭)
3. **D的优先级低**: 只解决single_function的SymbolHit问题，性价比低于A+B+E+C

---

## 五、关键洞察总结

1. **召回是瓶颈，不是排序** — cross_component Recall@5=32%说明答案根本不在候选池中。扩大候选池是性价比最高的优化
2. **58%→52%的漂移是索引"污染"** — 恢复pre_6b_backup索引即可修复，不是代码退化
3. **BGE +2pp不是交叉编码器路线失败** — 是管道设计错误(候选池太小+分数混搭)。在正确的管道中(A+E+C)，交叉编码器仍然可能是关键突破
4. **放弃"一变量一评估"的渐进思维** — 当前架构(52%)已经证明单变量优化已到天花板。需要A+B+E的组合拳才能突破
5. **FlashRank比BGE更适合本项目** — 更小(80MB vs 560MB)、更快(30ms vs 200ms)、更专注(MS MARCO微调的MiniLM)。配合50候选池，预期效果远超BGE在15候选池上的+2pp

---

## 参考文献

1. Bruch et al. "An Analysis of Fusion Functions for Hybrid Retrieval." ACM SIGIR 2023
2. "LevelRAG: Enhancing RAG with Multi-hop Logic." arXiv:2502.18139 (2025)
3. "Tree-Sitter-Based Knowledge Graphs for LLM Code Exploration via MCP." arXiv:2603.27277 (2026)
4. FlashRank: Ultra-lite re-ranking. github.com/PrithivirajDamodaran/FlashRank (2025)
5. Jina Reranker v2: Code search capable. jina.ai/news/jina-reranker-v2 (2024)
6. "Hybrid RAG: BM25 + RRF + Cross-Encoder." AI Workflow Lab (2026)
7. "Best Rerankers for RAG 2026: 7 Compared." futureagi.com (2026)
8. Cohere Rerank 4. cohere.com/blog/rerank-4 (2026)
