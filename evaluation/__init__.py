"""openUBMC Code RAG 评估框架。

提供检索质量评估和 Agent 回答质量评估，支持：
- 回归测试集驱动的检索指标计算（File@K, Recall@K, MRR, NDCG 等）
- LLM-as-Judge 的 Agent 回答四层加权评分
- 多搜索模式 A/B 对比
- Rich 表格报告 + JSON 导出
"""
