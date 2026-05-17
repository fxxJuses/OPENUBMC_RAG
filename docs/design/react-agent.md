# RAG Chat 架构变更：固定管线 → ReAct Agent

## 1. 变更背景

### 1.1 原架构

`ubmc_rag/chat/chain.py` 使用固定 4 步管线处理每个查询：

```
用户问题 → Query Rewrite → Query Analysis → Multi-Query Retrieval → RAG Generation
```

每一步都是串行执行的，无论查询是否需要。

### 1.2 问题

**问题 1：简单追问触发无意义检索**

用户先问"pcie_device 设备是如何加载的"（需要检索），再追问"用流程图的方式绘制出来"（不需要检索）。旧管线对第二个查询仍然执行完整的 Rewrite → Analyze → Retrieve → Generate 流程，Query Analyzer 还将意图误判为 `architecture` 而非 `general`。

**问题 2：单次检索策略无法应对复杂查询**

用户问"sensor 和电源管理的关系"需要多角度检索，但管线只做一轮 Analyze → Retrieve，检索策略不可动态调整。

**问题 3：检索决策与生成分离**

检索多少、用什么工具检索，全由规则和单次 LLM 调用决定，LLM 生成回答时无法根据需要主动补充检索。

### 1.3 目标

- LLM 自主决定是否检索代码、用什么工具检索、检索几次
- 简单追问直接基于对话历史回答，零工具调用
- 复杂查询可多次调用不同工具后再综合回答

---

## 2. 方案选型

### 2.1 候选方案

| 方案 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| **ReAct Agent** | 单 Agent + Tool 注册，LLM 自主决策 | 实现简单、延迟低、LLM 已能自主规划 | 复杂多步查询可能"遗忘"计划 |
| **Plan-and-Solve** | 先 LLM 规划步骤，再逐步执行 | 显式计划、步骤可审计、错误可按步重试 | 额外 LLM 调用增加延迟、实现复杂 |
| **混合方案** | 简单查询走 ReAct，复杂查询自动升级 Plan-and-Solve | 兼顾两者优势 | 实现最复杂、路由判断困难 |

### 2.2 选择 ReAct 的理由

1. **查询类型匹配**：代码助手的核心场景是单次或少量检索 + 生成，不需要复杂的分步编排
2. **核心痛点已解决**：追问不触发检索是 ReAct 的天然行为——LLM 看到对话历史有足够上下文就不调用工具
3. **延迟友好**：简单查询 1 次 LLM 调用（无工具），复杂查询 2-3 次（决策 + 工具 + 综合），与原管线相当
4. **渐进演进**：未来发现 LLM 多步规划不足时，可升级为 Plan-and-Solve

---

## 3. 架构设计

### 3.1 整体流程

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────┐
│  ReAct Agent（create_agent）                     │
│                                                   │
│  System Prompt: 角色定义 + 工作策略 + 证据约束    │
│  Tools: search_code, find_definitions,            │
│         find_references, list_components,          │
│         get_component_deps                         │
│                                                   │
│  LLM 循环：                                       │
│    1. 分析问题 → 决定是否调用工具                  │
│    2. 调用工具 → 获取结果                          │
│    3. 基于结果生成最终回答                          │
│                                                   │
│  对话历史：完整 message 链（含 ToolMessage）       │
└─────────────────────────────────────────────────┘
    │
    ▼
最终回答
```

### 3.2 与旧管线的关键区别

| 维度 | 旧管线 | ReAct Agent |
|------|--------|-------------|
| 流程控制 | 固定 4 步串行 | LLM 自主决策 |
| 追问处理 | 显式 Rewrite LLM 调用 | 对话历史自然理解 |
| 检索触发 | 每次查询必触发 | LLM 按需调用 |
| 检索工具 | 统一 multi_query_search | 5 个专用 Tool |
| 意图分析 | 单独 LLM 调用（QueryAnalyzer） | Agent 自主判断 |
| 上下文传递 | 手动拼接 context | ToolMessage 链自动传递 |

### 3.3 消息流

一次 Agent 交互的消息链：

```
HumanMessage("帮我查 pcie_device 加载流程")
  ↓
AIMessage(tool_calls=[{name: "search_code", args: {query: "pcie_device load"}}])
  ↓
ToolMessage(content="[1] device_loader.lua:308-350\nfunction task_load_unload_device...")
  ↓
AIMessage(tool_calls=[{name: "find_definitions", args: {symbol_name: "load_device"}}])
  ↓
ToolMessage(content="[1] device_loader.lua:851\nfunction on_pcie_card_bdf_changed...")
  ↓
AIMessage(content="根据检索到的代码，PCIe 设备加载流程如下...")  ← 最终回答
```

追问"绘制流程图"时：

```
HumanMessage("绘制成流程图")
  ↓
AIMessage(content="根据上述流程，可以绘制为...")  ← 直接回答，无工具调用
```

---

## 4. Tool 设计

### 4.1 Tool 注册

使用 LangChain `@tool` 装饰器 + 工厂函数模式，通过闭包捕获 `HybridSearchEngine` 和 `IndexManager`：

```python
def create_tools(engine: HybridSearchEngine, index_mgr: IndexManager) -> list[BaseTool]:
    @tool
    def search_code(query: str, ...) -> str:
        """在 openUBMC 代码库中进行混合语义+关键词搜索。..."""
        ...
    return [search_code, find_definitions, find_references, list_components, get_component_deps]
```

### 4.2 Tool 列表

| Tool | 底层调用 | 适用场景 |
|------|---------|---------|
| `search_code` | `engine.search()` | 理解代码逻辑、查找代码片段、探索架构 |
| `find_definitions` | `engine.search(is_code_query=True)` + 符号名匹配 | 查找函数/类定义位置 |
| `find_references` | `engine.search(is_code_query=True)` | 查找符号引用位置 |
| `list_components` | `index_mgr.get_all_chunks()` 聚合 | 列出所有微组件 |
| `get_component_deps` | `index_mgr.get_all_chunks()` 过滤 | 获取组件依赖关系 |

### 4.3 设计原则

- **Tool description 即工具说明**：每个 Tool 的 `description` 字段用中文描述功能和适用场景，`create_agent` 自动注入给 LLM，不在 System Prompt 中重复
- **返回格式化文本**：Tool 返回带 `[Source N]` 标记的文本，供 LLM 在最终回答中引用
- **复用现有能力**：逻辑与 MCP Server 的同名工具一致，只是从 FastMCP Tool 适配为 LangChain Tool

---

## 5. Agent System Prompt

### 5.1 Prompt 结构

```
角色定义（1 句）
  ↓
工作策略（3 条：判断是否检索、选择工具、利用对话历史）
  ↓
基本规则（3 条：引用格式、语言、架构背景）
  ↓
证据约束（5 条：citation、不推测、无法确定兜底、不编造代码、不推理关系）
```

### 5.2 关键设计

- **工作策略第一条**：明确告诉 LLM "不需要检索"的场景，引导其对追问直接回答
- **证据约束**：从 anti-hallucination 方案中继承，保持 citation enforcement 能力
- **工具说明不在此处**：由 `create_agent` 自动从 Tool description 注入

---

## 6. 对话历史管理

### 6.1 消息类型

| 类型 | 来源 | 大小控制 |
|------|------|---------|
| `HumanMessage` | 用户输入 | 保留原文 |
| `AIMessage`（含 tool_calls） | Agent 工具选择决策 | 保留原文 |
| `ToolMessage` | 工具返回结果 | 历史 > 40 条时截断到 2000 字符 |
| `AIMessage`（最终回答） | Agent 最终输出 | 保留原文 |

### 6.2 历史裁剪策略

```python
def _trim_history(messages, max_messages=40):
    if len(messages) <= max_messages:
        return messages
    trimmed = messages[-max_messages:]
    # 截断 ToolMessage 内容控制 token 开销
    for msg in trimmed:
        if isinstance(msg, ToolMessage) and len(msg.content) > 2000:
            msg.content = msg.content[:2000] + "\n...[truncated]"
    return trimmed
```

为什么是 40 条：一条工具调用链 = HumanMessage + AIMessage(tool_calls) + ToolMessage + AIMessage(回答) = 4 条。10 轮对话 ≈ 40 条消息。

---

## 7. Debug 模式

遍历 Agent 交互中新增的消息，用 Rich Panel 分类渲染：

| 消息类型 | Panel 标题 | 颜色 | 内容 |
|---------|-----------|------|------|
| AIMessage + tool_calls | Agent: Tool Selection | 青色 | 工具名 + 参数 |
| ToolMessage | Tool Result: {name} | 黄色 | 结果摘要（前 500 字符） |
| AIMessage 无 tool_calls | Agent: Final Response | 绿色 | 回答摘要（前 300 字符） |

---

## 8. 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `ubmc_rag/chat/tools.py` | **新增** | 5 个 LangChain Tool 定义 |
| `ubmc_rag/chat/chain.py` | **重构** | 固定管线 → `create_agent` ReAct Agent |
| `ubmc_rag/cli/chat_cmd.py` | 不变 | `run_chat()` 签名不变 |
| `ubmc_rag/chat/retriever.py` | 不变 | Tool 直接使用 `engine` |
| `ubmc_rag/search/query_analyzer.py` | 不再调用 | Agent 自主决策取代显式意图分析 |

**删除的组件**：
- `_REWRITE_PROMPT`、`_rewrite_query()`：Agent 通过对话历史自动处理追问
- `create_rag_chain()`：被 Agent 替代
- `QueryAnalyzer` 调用：Agent 自主判断意图

---

## 9. 验证结果

### 9.1 测试用例

| 场景 | 输入 | Agent 行为 | 预期 |
|------|------|-----------|------|
| 代码查询 | "pcie_device 和 bios 的关系" | 调用 `search_code` 1 次 | 获取相关代码后回答 |
| 复杂分析 | "梳理 pcie_device 全部加载流程" | 调用 `find_definitions` + `find_references` | 多次检索后综合 |
| 追问（非检索） | "绘制成流程图" | 无工具调用 | 直接基于对话历史生成 Mermaid 图 |
| 追问（需检索） | "pcie_device 有哪些依赖组件？" | 调用 `get_component_deps` | 针对性检索后回答 |

### 9.2 幻觉验证

对 Agent 的 bios-pcie_device 关系回答逐条核对源码：

| Agent 论断 | 源码位置 | 判定 |
|-----------|---------|------|
| `set_fault_status_by_bios` 函数存在 | `device_service.lua:1147` | 准确 |
| `listen_bios_path` 监听 BIOS 变化 | `biz_topo_service.lua:595` | 准确 |
| `on_pcie_card_bdf_changed` 调用 `load_device` | `device_loader.lua:851-868` | 准确 |
| `bios_object:get_pcie_info()` 存在 | `bios_object_mutihost.lua:308` | 准确 |

**结论**：未发现幻觉。证据约束规则在 ReAct Agent 模式下仍然有效。

---

## 10. 技术依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `langchain` | ≥ 0.3.0 | Agent 框架 |
| `langgraph` | ≥ 1.1.10 | `create_agent` 底层图引擎 |
| `langchain-openai` | ≥ 0.3.0 | DashScope LLM 接入 |

`langgraph` 是 `langchain` 的传递依赖，无需新增安装。
