# RAG Anti-Hallucination 改进方案

## 1. 问题发现

### 1.1 触发场景

用户通过 `ubmc-rag chat -m qwen-max` 提问：

```
给我解释一下openubmc中，sensor和电源的管理是否有联合的关系
```

系统检索到 5 个代码片段：

| 编号 | 来源文件 | 内容 |
|------|---------|------|
| [1] | `sensor_const.lua` | 传感器常量定义（事件类型、能力、读值类型等） |
| [2] | `sel_management.lua:1072-1087` | `get_sensor_prop` 函数（查找 SensorNumber 和 OwnerLun） |
| [3] | `sensor_management.lua:79-92` | `register_ipmi` 函数（注册 IPMI 请求处理函数） |
| [4] | `psu_def.lua` | 电源常量定义（设备状态、轮询参数等） |
| [5] | `protocol/init.lua` | 电源协议配置（PMBUS、CANBUS 等） |

### 1.2 LLM 回答中的三类幻觉

#### 幻觉 1：超出证据范围的推理（中等严重）

LLM 声称 sensor 和 power_mgmt 存在"联合管理"关系，给出了两个具体论断：

- "传感器可以用来监控电源的状态，如温度、电压、电流等"
- "当电源发生故障或异常时，可以通过传感器触发报警事件"

**事实**：5 个源文件中**没有任何证据**支持两个组件之间存在交互。经手动检查跨组件引用：
- `power_mgmt/src/lualib/power_ipmi.lua` 有一个 `get_threshold_sensor_reading` 函数，但这是 power_mgmt 自己实现的 IPMI sensor reading 接口，与 sensor 微组件无关
- sensor 微组件代码中没有任何对 power_mgmt 的引用

LLM 基于通用 BMC/服务器领域知识"推理"出了这个关系，而非基于检索到的代码。

#### 幻觉 2：编造示例代码（严重）

"示例"部分包含完全编造的 Lua 代码：

```lua
-- 假设的代码示例
local temperature = get_sensor_reading(sensor_id)
if temperature > critical_threshold then
    trigger_alarm(sensor_id)
end
```

虽然标注了"假设的代码示例"，但与真实代码引用混在一起呈现，容易误导读者认为这是系统实际运行方式。

#### 幻觉 3：常量误用（轻微）

回答中将 `OFFSET_CRITICAL_UGL=8` 和 `OFFSET_CRITICAL_UGH=9` 用于"定义一个温度传感器，并设置其阈值"的场景。实际上这两个常量是**门限传感器的事件类型偏移量**（event type offset），不是温度阈值。

### 1.3 根因分析

```
                    用户问题
                       │
                       ▼
            ┌─────────────────────┐
            │ "sensor和电源管理    │
            │  是否有联合的关系"   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ QueryProcessor      │  ← 纯规则：正则匹配、关键词提取
            │ 无意图识别           │  ← 没有识别出这是"跨组件关系查询"
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ 单次混合检索         │  ← 一个 query 只捞到各自独立的代码
            │ sensor_const.lua    │     没有捞到跨组件交互证据
            │ sensor_mgmt.lua     │
            │ psu_def.lua         │     (power_ipmi.lua 被遗漏)
            │ protocol/init.lua   │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ LLM 生成回答         │  ← System Prompt 缺少证据约束
            │ 无 citation 要求     │     LLM 自由使用先验知识推理
            │ 无"证据不足"兜底     │     编造假设性代码示例
            └─────────────────────┘
```

三个根本原因：

1. **检索缺失**：关系类问题需要多角度子查询交叉检索，单次检索只能捞到各自独立的代码
2. **无意图识别**：QueryProcessor 是纯规则的，无法理解"是否有联合关系"需要搜索什么
3. **Prompt 约束不足**：缺少 citation enforcement 和 grounded generation 规则

---

## 2. 解决方案设计

采用三层防御策略，按成本和收益排序：

### 2.1 P0：Prompt 加固（Citation Enforcement + Grounded Generation）

**改动文件**：`ubmc_rag/chat/chain.py` — `_SYSTEM_PROMPT`

**改动内容**：在 system prompt 中新增 5 条证据约束规则：

```
## 证据约束（严格遵守）
4. 每个事实性论断必须标注来源，格式：论断内容 [Source N]
5. 只根据检索到的代码回答，不要使用你的先验知识进行推测
6. 如果检索结果不足以回答问题，明确说"根据检索到的代码，无法确定"
7. 不要编写假设性或示例性代码。如果要说明某个机制，只引用源码中实际存在的代码
8. 不要对组件之间的关系做推理，除非源码中有明确的调用、require、import 等直接证据
```

**原理**：不是"禁止编造"（负面禁令效果有限），而是**正面要求每个论断必须锚定到具体来源**。研究表明 "Context Highlighting" + "Citation Enforcement" 在减少 RAG 幻觉方面效果最稳定。

**预期效果**：直接消除编造示例和超出证据推理，预计减少 60-70% 的幻觉。

### 2.2 P1：LLM 查询分析器（意图分类 + 子查询生成）

**新增文件**：`ubmc_rag/search/query_analyzer.py`

**设计**：在检索之前，使用 LLM 分析用户问题，输出结构化检索策略。

**意图分类**：

| 意图类型 | 判断依据 | 检索策略 |
|---------|---------|---------|
| `relationship` | 询问两个组件/模块间的关系、交互 | 生成 3-5 个交叉引用子查询 |
| `code_lookup` | 查找函数、类、变量的定义或用法 | 直接透传原始查询 |
| `architecture` | 询问整体架构、设计模式 | 生成 2-3 个架构相关子查询 |
| `debug` | 排查问题、分析错误 | 生成 2-3 个相关子查询 |
| `general` | 其他 | 直接透传 |

**关系查询的子查询生成策略**：

对于 "sensor 和电源管理是否有联合关系" 这类问题，分析器会生成：

```json
{
  "intent": "relationship",
  "components": ["sensor", "power_mgmt"],
  "sub_queries": [
    "sensor require power OR power_mgmt",
    "power_mgmt require sensor OR sensor_const",
    "sensor_management ipmi power reading",
    "get_threshold_sensor_reading power"
  ],
  "reasoning": "用户询问两个组件间的关系，需要检索交叉引用"
}
```

这样就能捞到 `power_ipmi.lua` 里的 `get_threshold_sensor_reading`——**真正的跨组件证据**。

**实现关键**：
- 使用 LLM 做意图分析（基于 few-shot prompt，无训练成本）
- 输出结构化 JSON，包含 intent、components、sub_queries
- 异常兜底：解析失败时退化为原始查询

### 2.3 P2：多查询检索 + 结果去重聚合

**改动文件**：`ubmc_rag/chat/retriever.py` — `UBMCRetriever`

**设计**：

```
子查询 1 ──▶ HybridSearchEngine.search() ──▶ 结果集 1 ─┐
子查询 2 ──▶ HybridSearchEngine.search() ──▶ 结果集 2 ─┤
子查询 3 ──▶ HybridSearchEngine.search() ──▶ 结果集 3 ─┤
...                                                     │
                                                        ▼
                                              按 chunk_id 去重
                                              保留最高 score
                                              按 score 排序
                                              截断 top_k
                                                        │
                                                        ▼
                                              最终检索结果
```

**去重策略**：以 `repo/file_path:start_line-end_line` 作为唯一键，重复 chunk 保留最高分数。

**P1 + P2 的集成方式**（在 `chain.py` 中）：

```
分析结果 > 1 个子查询 → multi_query_search（多查询聚合）
分析结果 = 1 个子查询 → 原始 invoke（单查询，无额外开销）
```

---

## 3. 改动后的完整 Pipeline

```
用户问题
    │
    ▼
┌──────────────────────┐
│ _rewrite_query()     │  追问改写（已有）
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ QueryAnalyzer        │  ← 新增：LLM 意图分析 + 子查询生成
│ .analyze(query)      │
│ → AnalyzedQuery      │
│   intent             │
│   components         │
│   sub_queries        │
└──────────┬───────────┘
           │
     sub_queries 数量
     ┌──────┴──────┐
     │ > 1         │ = 1
     ▼             ▼
┌──────────┐  ┌──────────┐
│ Multi    │  │ Single   │
│ Query    │  │ Query    │  ← 新增：多查询路径
│ Search   │  │ Search   │
└────┬─────┘  └────┬─────┘
     │             │
     ▼             ▼
┌──────────────────────┐
│ HybridSearchEngine   │  BM25 + Dense + RRF + Rerank（已有）
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 加固后的 System Prompt│  ← 新增：citation enforcement
│ + 检索上下文          │
│ + 用户问题            │
└──────────┬───────────┘
           │
           ▼
        LLM 生成
```

---

## 4. 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `ubmc_rag/chat/chain.py` | 修改 | 加固 `_SYSTEM_PROMPT`，集成 `QueryAnalyzer`，多查询检索分支 |
| `ubmc_rag/search/query_analyzer.py` | **新增** | LLM 查询分析器：意图分类 + 子查询生成 |
| `ubmc_rag/chat/retriever.py` | 修改 | 新增 `multi_query_search()` 方法 |

---

## 5. 证据收集模板

### 5.1 测试查询集

| 编号 | 查询 | 类型 | 预期行为 |
|------|------|------|---------|
| Q1 | "给我解释一下openubmc中，sensor和电源的管理是否有联合的关系" | relationship | 检索跨组件证据，若不足则明确告知 |
| Q2 | "sensor_management 的 register_ipmi 函数做了什么" | code_lookup | 正常返回函数说明（回归测试） |
| Q3 | "power_mgmt 中有没有获取传感器读值的功能" | relationship | 应检索到 power_ipmi.lua 的 get_threshold_sensor_reading |
| Q4 | "fru 和 sensor 的通信机制是什么" | relationship | 证据不足时应明确告知 |
| Q5 | "sensor_const.lua 中定义了哪些常量" | code_lookup | 正常列举常量（回归测试） |

### 5.2 评估指标

| 指标 | 定义 | 采集方式 |
|------|------|---------|
| 幻觉率 | 无支撑的事实性论断 / 总论断数 | 人工逐条核对 |
| 引用准确率 | 每条 [Source N] 引用是否真实对应 | 对照源码验证 |
| "无法确定"正确性 | 证据不足时是否正确识别 | 人工判断 |
| 回归率 | code_lookup 查询是否仍正常工作 | 对比改动前后 |

### 5.3 实验记录

| 日期 | 方案版本 | 模型 | 测试查询 | 幻觉率 | 引用准确率 | 回归通过 | 备注 |
|------|---------|------|---------|--------|-----------|---------|------|
| 2026-05-16 | 基线（改动前） | qwen-max | Q1 | 3/3 论断幻觉 | N/A | - | 编造关系、代码、常量误用 |
| | | | | | | | |

---

## 6. 参考来源

- [Understanding RAG Part VIII: Mitigating Hallucinations in RAG](https://machinelearningmastery.com/understanding-rag-part-viii-mitigating-hallucinations-in-rag/) — 三类幻觉缓解策略：数据层、上下文层、推理层
- [VOTE-RAG (2026)](https://arxiv.org/html/2603.27253v2) — 多 Agent 投票缓解 RAG 幻觉，证明 ensemble voting 比复杂辩论更有效
- [RAG Prompting to Reduce Hallucination](https://futureagi.com/blog/rag-prompting-to-reduce-hallucination/) — Context Highlighting + Citation Enforcement 效果最稳定
- [Self-RAG](https://arxiv.org/abs/2310.11511) — 自反思检索生成，生成后验证
- [CRAG](https://arxiv.org/abs/2401.15884) — 纠正型 RAG，检索结果评估与纠正
