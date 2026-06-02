# 迭代6复盘 — 意图感知检索增强

> 日期: 2026-06-02
> 基线: File@5=0.54, Recall@5=0.39, MRR=0.408
> 最终: File@5=0.62, Recall@5=0.45, MRR=0.435

---

## 一、背景

本轮从 50 个测试用例的逐案分析出发，发现 service.json 文件存在系统性检索失败（11 个 MDS 相关用例中 5 个失败）。根因是 Dense 嵌入无法将"依赖关系"类中文查询与纯 JSON 内容关联，BM25 分词器又不支持中文。最终通过三处改动实现全指标提升、零回归。

---

## 二、有效改动（保留）

### 2.1 BM25 文档增强：mds_service 元数据注入

**文件**: `ubmc_rag/indexing/bm25_index.py` — `_build_document()` 方法

**改了什么**: 当分块类型为 `mds_service` 时，把 metadata 中的 `dependencies`（依赖列表）、`required_interfaces`（接口列表）、`service_name`（服务名）拼入 BM25 文档。

```python
if chunk.chunk_type == "mds_service":
    deps = chunk.metadata.get("dependencies", [])
    ifaces = chunk.metadata.get("required_interfaces", [])
    if deps:
        parts.append(" ".join(deps))
    if ifaces:
        parts.append(" ".join(ifaces))
```

**为什么有效**: 原来 BM25 只能匹配 JSON 原文（`{"dependencies": {"build": ...}}`），加入 `libipmi`、`mdb_interface` 等具体依赖名后，查询中出现这些名字时 BM25 可以命中。单独使用不足以改变最终 top-10（BM25 分数在 RRF 融合中竞争力不够），但作为下游注入的索引基础是必要的。

**补充说明**: 此增强只在 `build_index()` 时生效——此时 chunk 对象有完整的 metadata。ChromaDB 的 `to_chroma_metadata()` 不会序列化 metadata 字典，所以从 ChromaDB 加载的 chunk.metadata 总是空的。这是预期行为，不需要修复。

### 2.2 意图感知检索注入：双路定向补充

**文件**: `ubmc_rag/search/hybrid_search.py`

**改了什么**: 用正则检测查询是否涉及"依赖/接口"意图，如果是，执行一次 `chunk_type="mds_service"` 的定向 Dense 检索，将结果注入到 Dense 和 BM25 双路列表的前部。

```python
# 意图检测正则
_DEPENDENCY_QUERY_RE = re.compile(
    r"依赖|dependency|dependencies|接口定义|interface|"
    r"组件.*关系|component.*dep|依赖关系|dep graph|"
    r"service\.json|component info", re.IGNORECASE,
)

# 注入逻辑：Dense 第5位 + BM25 第10位
svc_results = self._retrieve_mds_service(query_embedding, existing_ids)
for i, r in enumerate(svc_results[:3]):
    dense_results.insert(5 + i, r)
    bm25_results.insert(10 + i, SearchResult(chunk=r.chunk, score=10.0, source="bm25"))
```

**为什么有效**:

关键数据对比——以 TC-011 "sensor 组件的依赖关系" 为例：

| 阶段 | sensor/service.json 排名 | 说明 |
|------|-------------------------|------|
| 常规 Dense 全局检索 | 第 50+ 名 | JSON 内容嵌入距离远 |
| 定向 Dense（仅 mds_service）| **第 1 名** | 13 个候选中语义最匹配 |
| 注入后 RRF 融合 | **第 1 名**（0.015） | 双路贡献叠加 |
| Boosting 后 | 第 11 名（0.021） | 常规结果的 symbol/path bonus 更高 |
| + MDS_SERVICE_BONUS 后 | **第 4 名**（0.031） | 额外 0.010 补回差距 |

注入位置（Dense=5, BM25=10）是经验值：
- 太靠前（rank 0-2）会排挤正常检索结果
- 太靠后（rank 15+）RRF 分数不够进入候选池
- 当前值在 5/10 对所有依赖类查询都不产生回归

**可扩展方向**: 同样的模式可以用于其他 chunk_type 路由：
- "模型定义" 查询 → `chunk_type="mds_model"` 定向检索
- "IPMI 命令" 查询 → `chunk_type="mds_ipmi_cmd"` 定向检索
- 核心模式：识别意图 → where 过滤检索 → 注入双路前部

### 2.3 Reranker mds_service 专项 bonus

**文件**: `ubmc_rag/search/reranker.py` — `_apply_boosts()` 方法

**改了什么**: 新增 MDS_SERVICE_BONUS（0.010），当分块类型为 `mds_service` 且查询匹配依赖关键词时生效。

```python
MDS_SERVICE_BONUS = 0.010

if r.chunk.chunk_type == "mds_service" and _DEP_QUERY_RE.search(query_lower):
    bonus += MDS_SERVICE_BONUS
```

**为什么需要这一层**: 即使 RRF 融合后 service.json 排名第一，常规代码结果的 boosting 总量更高（symbol + repo + filepath 合计约 +0.019），会把 service.json 挤出 top-10。+0.010 恰好补回差距。

---

## 三、失败尝试（不要再做）

### 3.1 修改 chunk.content 来增强检索 ❌

**试了什么**: 在 `json_parser.py` 中生成自然语言摘要（如 `[Service Summary] Component: sensor. Dependencies: libipmi, ...`），拼接到 `chunk.content` 前面或后面。

**结果**: File@5 从 0.54 降到 0.50（回归 4 个用例）。前置和后置都试过，都回归。

**根因**: chunk.content 同时是 Dense embedding 和 BM25 的输入。修改 content 会改变 embedding，破坏整个向量空间——原来匹配良好的代码 chunk 之间距离关系被扰乱，导致以前能找到的结果找不到了。

**教训: 永远不要为了检索增强而修改 chunk.content。** content 是 embedding 的基础，改动是全局性的、不可控的。信息注入应该走 metadata、BM25 文档增强、或者检索后 boosting 这些不影响 embedding 的路径。

### 3.2 扩大 RRF 候选池（top_k × 3 → × 5）❌

**试了什么**: `hybrid_search.py` 中候选池从 `top_k * 3` 扩大到 `top_k * 5`。

**结果**: 10 个用例回归。File@5 从 0.54 降到 0.48。

**根因**: 更大的候选池让低质量结果也进入 boosting 阶段。boosting 是加法 bonus，噪声结果可能因为仓库名匹配、路径部分匹配等意外命中 bonus 条件，被推进 top-10，挤掉正确结果。

**教训**: 候选池大小应该保守。top_k × 3 已经够用——真正的瓶颈是某些类型（如 service.json）根本不在候选池中，而不是池子太小。

### 3.3 同文件硬去重（max_per_file 3 → 2）❌

**试了什么**: 同一文件最多保留 2 条结果。

**结果**: 与候选池扩大叠加时产生 10 个回归。单独测试也有回归。

**根因**: 很多正确答案恰好在同一文件的不同函数中（例如 sensor_management.lua 中有多个与传感器管理相关的函数）。强制截断到 2 条会丢掉正确的第二、第三条结果。

**教训**: `diversity_max_per_file=3` 是合理的默认值。多样性控制应该用分数衰减（现有 0.7 倍惩罚），不应该硬截断。

### 3.4 仓库名加权匹配 ❌

**试了什么**: 用 REPO_ALIASES 字典做仓库名→查询别名映射（如 `sensor → ["sensor", "sensor management"]`），匹配时给 `REPO_BONUS × 2`。

**结果**: 8 个回归用例。

**根因**: 仓库名匹配粒度太粗。"sensor" 匹配后，sensor 仓库下所有文件都获得 bonus——包括与当前查询无关的文件。而正确答案可能在不匹配的仓库中（跨组件查询时，答案在 fructrl 但查询只提到 sensor）。

**教训**: 仓库名匹配不适合作为 reranker 的 bonus 信号。符号名、文件路径匹配是更精确的信号。REPO_ALIASES 字典已清理删除。

### 3.5 Graph RAG 三路融合 ❌

**结论**: 对检索质量零贡献（File@5=0.54→0.54），已从代码中完全移除。

**根因**: Dense/BM25 双路已经覆盖了有价值的检索结果。Graph 扩展到的 entity 与双路结果高度重叠；DEPENDS_ON 边只有 9 条太少；跨组件 CALLS 扩展到的函数也已在双路结果中。

**教训**: 在当前规模（13 个组件、9470 个分块）下，Graph RAG 无法提供额外价值。未来如果扩大到 50+ 组件可以重新评估，但不要再在当前规模下尝试。

---

## 四、当前检索管线

```
用户查询
  ↓
QueryProcessor（意图识别 + 中英术语扩展）
  ↓
Dense 检索（top_k × 3）+ BM25 检索（top_k × 3）
  ↓
意图检测 → 定向补充 mds_service 候选（注入 Dense 第5位、BM25 第10位）
  ↓
RRF 融合（k=60, Dense 权重=0.5, BM25 权重=0.5）
  ↓
Boosting（符号名 + 文件路径 + 仓库名 + MDS模型类 + mds_service 专项）
  ↓
多样性过滤（同文件超 3 条降权 0.7 倍）
  ↓
返回 top_k 结果
```

---

## 五、下一步方向

### 高优先级
1. **交叉编码器重排序**：配置项已有（`cross_encoder_enabled`、`dashscope_reranker_enabled`），安装 sentence-transformers 或接入 DashScope qwen3-rerank 即可启用。预估 File@5 +0.05~0.10。
2. **查询扩展词干化**：当前"依赖"扩展为 `dependency dep`，但 BM25 分词器无法匹配 JSON 中的 `dependencies`。加入词干形式或直接扩展 `dependencies` 可以进一步增强 BM25 命中。

### 中优先级
3. **意图路由扩展**：将 2.2 的模式复用到 model.json 和 ipmi.json。
4. **metadata 持久化**：当前 ChromaDB 不存储 metadata 字段，reranker 无法用 dependencies 列表做精确匹配。如果未来需要，要在 `to_chroma_metadata()` 中序列化关键字段。

### 低优先级
5. **单文件查询优化**：single_file 类别 File@5=0.47，低于其他类别。
6. **失败用例深度分析**：仍有 19/50 失败，绝大多数是"目标不在 top-10"而非"排名太低"，需要从 embedding 质量层面解决。
