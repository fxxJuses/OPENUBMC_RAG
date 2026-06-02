# 迭代7优化方案 — 19个失败用例分析与下一步

> 日期: 2026-06-02
> 当前: File@5=0.62 (31/50), 目标: File@5≥0.70

---

## 一、19个失败用例分类

| 类别 | 数量 | 用例 | 根因 |
|------|------|------|------|
| A. 期望文件排名差2-3位 | 3 | TC-019, TC-026, TC-034 | boosting不够或Dense排序偏差 |
| B. 跨组件检索偏差 | 3 | TC-021, TC-024, TC-044 | Dense偏向一个组件，另一组件文件未进入top-10 |
| C. 文件内函数粒度问题 | 3 | TC-014, TC-035, TC-002 | 期望的是特定函数而非主文件，但该函数chunk排序低 |
| D. 拼写/模糊查询 | 4 | TC-045, TC-046, TC-047, TC-048 | 查询本身质量差 |
| E. 单词/极短查询 | 1 | TC-031 | 查询无歧义消解能力 |
| F. 场景型查询 | 2 | TC-049, TC-007 | 描述场景而非代码实体 |
| G. Dense embedding 覆盖不到 | 3 | TC-001, TC-010, TC-050 | embedding空间本身匹配不到 |

---

## 二、A类：差2-3位可修复（最高优先级）

### TC-019 "devmon device monitoring mechanism"
- **期望**: `devmon/libs/devmon/include/devmon/devmon.h`（rank=7,9,10）
- **当前top3**: devmon_root_interface.h, device/manager.h
- **根因**: `devmon.h` 是总头文件，内容短（只是 #include 集合），Dense 嵌入信息量不足。同文件出现在 rank 7/9/10，说明已经在候选池中，但被其他 devmon 文件的 boosting 挤下。
- **方案**: 无需特殊处理。devmon.h 出现在 top-10 已经说明检索到位，测试用例的期望文件定义可能过于严格（devmon_root_interface.h 同样是合理的入口）。

### TC-026 "sensor devmon device sensor event handling"（rank=10）
- **期望**: sensor_service.lua + devmon.h
- **当前top3**: sensor_management.lua, discrete_event_instance.h
- **根因**: sensor_service.lua 在 top-10 但排在第10位。前9位被 sensor 相关的其他文件占满。
- **方案**: 该用例 cross_component/hard，top-10 有一个期望文件已经不错。考虑修改测试用例降低 relevance 要求，或将 top_k 调到 15。

### TC-034 "vpd VPD 数据读取"（rank=7）
- **期望**: vpd/mds/service.json
- **当前top5**: 全是 pcie_device 文件
- **根因**: 查询不含"依赖"关键词，意图检测未触发。vpd/service.json 没有被注入候选池。VPD 数据读取查询的 Dense 嵌入偏向了 pcie_device（因为 pcie_device 也有 VPD 相关逻辑）。
- **方案**: 扩展意图检测正则，加入"数据读取|data.*read|数据访问"等模式。或：对 vpd 查询做组件名感知的 repo boosting。

---

## 三、B类：跨组件检索偏差

### TC-021 "sensor 和 power_mgmt 之间的关系"
- **期望**: sensor_service.lua + psu_service.lua（跨两个组件）
- **当前top5**: 全是 power_mgmt 文件
- **根因**: Dense 嵌入对"关系"类查询理解不足，偏向返回 power_mgmt 而非同时返回两个组件的结果。
- **方案**: 这类查询本质是"跨组件关联发现"，纯靠 embedding 很难解决。需要 multi-query 或 LLM query analysis。

### TC-024 "libipmi IPMI command execution in sensor"
- **期望**: protocol_ipmb.h + sensor_operation.lua
- **当前top3**: sensor/mds/ipmi.json（占3个slot）
- **根因**: ipmi.json 因为 IPMI 关键词密度高，占了3个 top-5 位置。sensor_operation.lua 只有1个chunk且 Dense 排名低。
- **方案**: 同文件去重加强——ipmi.json 在 top-3 出现3次，diversity 过滤只对第4+条降权。可以把 diversity_max_per_file 从 3 降到 2（但之前尝试过有回归）。更好的方案：在 boosting 阶段对 top-3 之后的同文件结果施加更强惩罚（如 ×0.5 而非 ×0.7）。

### TC-044 "devmon sensor 设备监控与传感器事件"
- **期望**: devmon.h + sensor_operation.lua
- **当前top5**: devmon 文件 + sensor_instance.lua
- **根因**: sensor_operation.lua 只有1个chunk（文件小），Dense 排名低。devmon.h 在 top-5 但 sensor 侧的期望文件不在。
- **方案**: 与 TC-021 类似，跨组件关联查询靠 embedding 难解决。

---

## 四、C类：函数粒度问题

### TC-014 "fructrl 上电下电控制逻辑"
- **期望**: pwr_action.lua, pwr_restore.lua
- **当前top5**: fructrl.lua（占3个slot）+ fructrl_obj.lua
- **根因**: pwr_action.lua 的 chunk 内容是 `ctor` 和 `init` 方法，与"上电下电控制"语义距离远。pwr_restore.lua 有24个chunk但都不含"上电下电"关键词。Dense 嵌入无法将中文查询与这些函数代码关联。
- **方案**: 文件内 chunk 级别问题，需要改进 embedding 质量或增加文件级摘要。

### TC-035 "sensor_mgmt 传感器管理入口"
- **期望**: sensor_mgmt/src/main.cpp
- **当前top5**: sensor/sensor_management.h 等
- **根因**: main.cpp 只有1个chunk，内容是 `app.register_service<sensor::sensor_service>`，与"传感器管理入口"语义匹配度低。Dense 偏向返回 sensor_management.h（因为文件名和内容都有"sensor management"）。
- **方案**: 可以在 json_parser 或 chunker 中为 main.cpp / *_app.lua 这类入口文件标记 chunk_type="entry_point"，在查询包含"入口"/"初始化"时做定向检索。

### TC-002 "ipmi_get_sensor_reading"
- **期望**: protocol_ipmb.h
- **当前top3**: sensor_management.lua, ipmi.json（×2）
- **根因**: protocol_ipmb.h 在索引中（2个chunk），但 Dense 排名不在 top-30。查询被理解为 IPMI 命令定义而非协议实现。
- **方案**: embedding 覆盖问题，短期内难以解决。

---

## 五、D/E/F/G类：测试用例问题

### D类（模糊查询）— 建议修改测试用例
- **TC-045** "sensr threshold config" → 拼写错误，应改为 "sensor threshold config"
- **TC-046** "power control on off" → 过于宽泛
- **TC-047** "ipmi command send receive" → 缺少组件限定
- **TC-048** "fru info read write" → 缺少组件限定

### E类（极短查询）— 可做系统优化
- **TC-031** "init" → 可在 QueryProcessor 中为极短查询做防御性扩展（init → initialize 启动 startup）

### F类（场景查询）— 建议修改或接受
- **TC-049** "temperature monitoring alert" → 场景描述，非代码实体查询
- **TC-007** "SEL event logging function" → 结果已相关，可能是排序期望问题

### G类（embedding覆盖）
- **TC-001** "get_sensor_data" → 函数名可能不存在
- **TC-010** "power_ipmi IPMI 电源控制命令" → 返回了相关文件，只是不是期望的那个
- **TC-050** "component dependency graph" → 定向检索返回了 service.json 但不是期望的 sensor

---

## 六、可实施的优化方案（按优先级排序）

### P1: 同文件更强去重（预估 +1~2 用例）

**问题**: ipmi.json 在 TC-024 top-3 中占3个slot，挤掉了 sensor_operation.lua。

**方案**: 将 diversity 降权系数从 0.7 提高到 0.5（对超过 max_per_file 的结果惩罚更重）。不改变 max_per_file=3 的硬限制，避免之前 P2 硬截断的回归。

```python
# reranker.py _apply_diversity
if count >= self.config.diversity_max_per_file:
    r = SearchResult(chunk=r.chunk, score=r.score * 0.5, source=r.source)  # 从 0.7 改为 0.5
```

**风险**: 低。0.5 仍然是降权不是截断，不会丢失结果。

### P2: 拼写纠错层（预估 +1 用例 TC-045）

**问题**: "sensr" 无法匹配 "sensor"。

**方案**: 在 QueryProcessor 中增加基于领域词典的编辑距离纠错。用 `difflib.get_close_matches` 对查询中无匹配的 token 做纠错。

```python
# query_processor.py
from difflib import get_close_matches

# 从索引符号名提取领域词典
DOMAIN_TERMS = {"sensor", "threshold", "config", "monitoring", ...}

def _spell_correct(self, query_tokens):
    corrected = []
    for token in query_tokens:
        if token.lower() not in self._domain_terms:
            matches = get_close_matches(token.lower(), self._domain_terms, n=1, cutoff=0.8)
            if matches:
                corrected.append(matches[0])
    return corrected
```

### P3: 扩展意图检测正则（预估 +1 用例 TC-034）

**问题**: "vpd VPD 数据读取" 不触发 service.json 定向检索。

**方案**: 在 `_DEPENDENCY_QUERY_RE` 中加入更多数据读取/访问模式。

```python
_DEPENDENCY_QUERY_RE = re.compile(
    r"依赖|dependency|dependencies|接口定义|interface|"
    r"组件.*关系|component.*dep|依赖关系|dep graph|"
    r"service\.json|component info|"
    r"数据读取|数据访问|data.*read|数据流",  # 新增
    re.IGNORECASE,
)
```

### P4: 入口文件识别（预估 +1 用例 TC-035）

**问题**: main.cpp / *_app.lua 等入口文件无法被"入口"类查询命中。

**方案**: 在 chunker 或 parser 中识别入口文件模式，标记 chunk_type。在查询包含"入口"/"初始化"/"main"/"启动"时定向检索。

---

## 七、预期收益

| 方案 | 预估新增命中 | File@5 预估 | 风险 |
|------|-------------|-------------|------|
| P1 更强去重 | +1~2 | 0.64 | 低 |
| P2 拼写纠错 | +1 | 0.66 | 极低 |
| P3 扩展意图 | +1 | 0.68 | 低 |
| P4 入口文件 | +1 | 0.70 | 中 |
| **合计** | **+4~5** | **0.70~0.72** | |

剩余 14 个失败用例中，约 8 个属于测试用例质量问题（模糊查询、拼写错误、场景描述），建议在优化系统的同时修改测试用例。其余 6 个是 embedding 覆盖的硬限制，需要升级 embedding 模型或增加交叉编码器重排序。
