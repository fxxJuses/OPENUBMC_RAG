# 方案设计中的困难与决策

## 困难一：openUBMC 代码托管平台不明确

### 问题

openUBMC 代码不在主流的开源平台（GitHub、GitLab）上，初期无法定位代码仓库。公开搜索的结果指向 GitCode 和 AtomGit 两个平台，且仓库分布方式特殊——不是统一在 `openUBMC` 组织下，而是分散在多个贡献者的个人仓库中。

### 影响

数据摄取管线的设计必须适应多平台、分散式仓库的情况，不能假设单一 Git Forge API。

### 解决

最终确认主要代码在 [GitCode (gitcode.com/org/openUBMC)](https://gitcode.com/org/openUBMC)。采用 `git clone` 作为统一获取手段，通过配置文件声明仓库列表，而非依赖平台 API 自动发现。这样做的代价是新增仓库需要手动配置，但换来了平台无关性。

---

## 困难二：Tree-sitter 节点类型在不同语言/版本间的命名差异

### 问题

开发初期假设 Tree-sitter 所有语言使用统一的节点类型命名（如 `function_definition`），但实际上不同语言的 grammar 命名差异很大：

| 预期 | Lua 实际 | C 实际 | Python 实际 |
|------|----------|--------|-------------|
| `function_definition` | `function_declaration` | `function_definition` | `function_definition` |
| `local_declaration` | `variable_declaration` | — | — |
| `class_definition` | — (用函数调用 `class()`) | `class_specifier` | `class_definition` |

### 影响

如果按统一假设编码，Lua 解析器完全无法提取函数和方法，导致主导语言（Lua）的索引为空。

### 解决

对每种语言单独做 AST 节点类型探测（打印整棵 AST 树），确认每种语言的实际节点类型后分别实现。这导致了 BaseParser 抽象基类的设计——每种语言有自己的 AST 语义，不能做过度抽象。

**教训**：Tree-sitter 的节点类型是 grammar 级别的，不是标准化的。每种新语言必须先做 AST 探测再写解析逻辑。

---

## 困难三：MDS JSON 的 Schema 感知分块

### 问题

openUBMC 的 JSON 文件不是普通配置文件，而是承载架构知识的模型定义：

- `service.json`：声明组件的依赖和接口
- `model.json`：定义 MDB 资源类（如 ThresholdSensor、DiscreteSensor）
- `ipmi.json`：定义 IPMI 命令（含 netfn、cmd byte）
- `.sr` 文件：CSR 设备描述（ManagementTopology + Objects）

如果用通用的 JSON 切分（按大小/行数），会破坏语义边界。例如把 `model.json` 里两个不同类的定义切碎混合。

### 影响

代码检索的准确性高度依赖分块质量。混合了多个 MDS 类的 chunk 会导致搜索 "ThresholdSensor" 时返回包含 DiscreteSensor 定义的无关内容。

### 解决

设计了 Schema 感知的 JSON 解析器（`JsonParser`），按文件名路由到不同的解析策略：

```
文件名 → 解析策略
service.json → 整文件一个 chunk，提取 dependencies/required 作为符号
model.json → 按顶层类键拆分，每个 MDS 类一个 chunk
ipmi.json → 按 cmds 下每个命令拆分
types.json → 按 defs 下每个类型定义拆分
*.sr → 拆分 ManagementTopology + 每个 Object
```

这要求对 openUBMC 的 MDS 规范有领域知识，不是通用方案。

**教训**：领域特定的配置文件需要领域特定的解析策略，通用切分会损失检索质量。

---

## 困难四：BM25 与 Dense 检索的分数融合

### 问题

BM25 输出的分数是 TF-IDF 系的（0 到正无穷），Dense 检索输出的是余弦相似度（-1 到 1）。两个分数尺度完全不同，不能直接相加。

SourceGraph 的方案是自研 Zoekt 引擎，同时支持三元组索引和 BM25，不存在跨系统融合问题。但我们需要组合两个独立系统。

### 影响

错误的融合方式会导致某一方始终主导结果：
- 线性相加：BM25 分数范围大，会压倒向量相似度
- 归一化后相加：归一化方法选择（min-max / z-score）影响结果稳定性

### 解决

采用 **Reciprocal Rank Fusion (RRF)** 替代分数融合：

```
RRF_score(d) = w × 1/(k + rank_bm25(d)) + (1-w) × 1/(k + rank_dense(d))
```

RRF 只使用排名位置（rank），不使用原始分数，天然不受分数尺度影响。k=60 是信息检索领域的经验值，防止排名靠前的结果过度主导。

**教训**：跨系统融合时，rank-based 方法比 score-based 方法更稳健。

---

## 困难五：代码嵌入模型对 Lua 语言的支持

### 问题

openUBMC 的主导语言是 Lua，但主流代码嵌入模型对 Lua 的支持非常有限：

- CodeBERT：不支持 Lua
- UniXcoder：不支持 Lua
- StarCoder Embeddings：支持有限
- 通用文本模型（如 BGE）：不理解代码结构

### 影响

如果嵌入模型不理解 Lua 语法，`local class = require 'mc.class'` 和 `function SensorApp:init(config)` 这样的模式无法被正确编码，语义搜索会退化为关键词匹配。

### 解决

选择 **jinaai/jina-embeddings-v2-base-code**：
- 明确支持 30 种编程语言，包含 Lua
- 8192 token 上下文（足够覆盖完整 MDS JSON 或长 Lua 类）
- 同时支持 code-to-code 和 code-to-text 相似度
- Apache 2.0 协议，无使用限制

这是调研范围内唯一满足所有条件的模型。

**教训**：小众语言（Lua）的代码搜索，模型选择面非常窄，需要仔细验证语言支持列表。

---

## 困难六：查询意图判断（自然语言 vs 代码片段）

### 问题

用户的查询可能是自然语言（"sensor 组件如何获取温度数据"）也可能是代码片段（`get_sensor_data` 或 `db:select(db.Sensor):where(...)`）。两种查询的最优检索策略不同：

- 自然语言查询 → Dense 检索权重应更高（语义匹配）
- 代码片段查询 → BM25 权重应更高（精确标识符匹配）

### 解决

设计 `QueryProcessor` 做查询分类：
- 检测代码特征：`{}()[];=::` 运算符、`function/class/local/return` 关键字
- 动态调整 RRF 权重：
  - 自然语言：BM25=0.4, Dense=0.6
  - 代码片段：BM25=0.6, Dense=0.4

同时，CLI 提供 `--code` 标志让用户显式声明查询类型。

**教训**：代码搜索系统必须区分查询意图，一刀切的权重配置会牺牲某类查询的效果。

---

## 困难七：SourceGraph 的开源与闭源边界

### 问题

SourceGraph 的核心搜索能力来自 Zoekt（开源，三元组索引）和 SCIP（开源，代码智能协议），但其搜索排名、代码智能聚合、Web UI 等关键能力是闭源的。借鉴时需要区分哪些可以直接用、哪些需要重新设计。

### 解决

采取"借鉴思想，自建实现"策略：

| SourceGraph 能力 | 我们的方案 | 区别 |
|------------------|-----------|------|
| Zoekt 三元组索引 | BM25 (rank_bm25) | BM25 更成熟，但三元组索引在大规模下更快 |
| ctags 符号提取 | Tree-sitter AST 符号提取 | 更精确，但需要为每种语言单独实现 |
| BM25 排名（跳过 IDF） | 标准 BM25（含 IDF） | SourceGraph 发现 IDF 在代码搜索中可能反作用，我们暂时保留 |
| SCIP 代码智能 | MCP Server 资源定义 | 不同协议，但达成类似目标（定义跳转、引用查找） |
| Web UI | CLI + MCP Server | 面向开发者而非浏览器，更轻量 |
