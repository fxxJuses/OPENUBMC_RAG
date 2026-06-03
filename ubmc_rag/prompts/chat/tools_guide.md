## 工具使用指南

### search_docs — openUBMC 文档知识库搜索
- **优先使用**：当用户询问 openUBMC 的概念、架构、开发流程、配置方法时
- 覆盖内容：架构说明、开发指南、API 参考、CSR/MDS 配置规范、IPMI 命令文档、FAQ
- 查询技巧：
  - 直接使用自然语言描述概念（如"MDS 框架设计"、"组件开发流程"）
  - 包含技术术语的中英文（如"硬件自发现 hardware discovery"）
  - 指定配置类型（如"CSR 配置"、"service.json 字段"）
- 典型场景：
  - "MDS 是什么" → search_docs
  - "如何开发一个新组件" → search_docs
  - "IPMI Sensor 命令有哪些" → search_docs

### search_code — 混合语义+关键词搜索
- 最常用的代码搜索工具，适合理解代码逻辑、查找代码片段、探索架构
- 查询技巧：
  - 同时包含中英文关键词（如"传感器阈值 sensor threshold"）
  - 包含可能的函数名（snake_case 推测）
  - 涉及特定组件时用 repo 参数过滤
  - 涉及特定语言时用 language 参数过滤（lua, c, cpp, json）
  - 涉及特定代码类型时用 chunk_type 过滤（function, class, mds_model, mds_service）
  - 查找精确代码/函数名时设置 intent_hint="code"
  - 理解概念和逻辑时设置 intent_hint="semantic"

### search_multi — 多角度交叉检索
- 适合复杂问题需要从不同角度检索代码
- 提供 2-5 个不同角度的查询词，结果自动去重合并
- 例如查询"sensor 和 power 的关系"时，可用 ["sensor require power", "power_mgmt sensor reading", "sensor power_monitor"] 三个查询交叉检索

### find_definitions — 符号定义查找
- 适合查找函数、类、变量的定义位置
- 输入精确的符号名效果最佳
- 如果不确定完整名称，先用 search_code 搜索

### find_references — 符号引用查找
- 适合查找某个函数/类在哪里被调用或使用
- 用于追踪代码调用链和理解依赖关系

### list_components — 组件列表
- 列出所有已索引的 openUBMC 微组件
- 适合用户需要了解有哪些组件时使用

### get_component_deps — 组件依赖分析
- 获取组件的依赖关系、接口定义、MDS 类和 IPMI 命令
- 适合理解组件间的依赖和交互关系
