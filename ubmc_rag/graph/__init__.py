"""代码知识图模块 —— 基于 AST 的代码关系图构建与检索。

利用 Tree-sitter 解析结果构建代码知识图，支持：
- 组件依赖关系（service.json dependencies）
- 导入关系（Lua require() / C #include）
- 函数调用关系（call_expression 匹配）
- 接口提供/消费关系（service.json interfaces）

图存储基于 NetworkX DiGraph，检索集成到混合搜索管线。
"""
