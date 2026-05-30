"""Agent 评估的 Judge 评分 Prompt 模板。

四层评分体系适配自日志分析 Agent 的评估框架：
1. 解决方案质量 (0.30) — 回答是否准确解决查询
2. 定位准确性 (0.25) — 文件路径和行号是否准确
3. 完整性 5W1H (0.25) — 六维信息覆盖
4. 证据可靠性 (0.20) — 每个论断是否有来源标注

加权得分阈值 ≥ 6.0 (0.6 × 10) 判 pass。
"""

# 评分权重
WEIGHTS = {
    "solution_quality": 0.30,
    "localization": 0.25,
    "completeness": 0.25,
    "evidence_reliability": 0.20,
}

# 通过阈值 (10 分制)
PASS_THRESHOLD = 6.0

JUDGE_SYSTEM_PROMPT = """\
你是一个代码 RAG 系统评估助手。你的任务是严格评估系统回答的质量。
你必须严格按照 JSON 格式输出评分，不要添加任何 markdown 标记或额外文字。

## 评分维度

### 1. 解决方案质量 (solution_quality, 0-10)
- 回答是否准确、直接地解决了用户的查询？
- 引用的代码是否正确且相关？
- 有没有答非所问或遗漏关键信息？

### 2. 定位准确性 (localization, 0-10)
- 引用的文件路径和行号是否准确？
- 是否指向了正确的组件和代码位置？
- 如果没有引用具体文件，此项应低分。

### 3. 完整性 5W1H (completeness, 0-10)
- What: 解释了是什么？
- Where: 指出了在哪个文件/函数？
- Why: 解释了为什么这样设计？
- How: 描述了如何工作？
- Who: 涉及哪些组件/模块？
- When: 在什么场景/时机下触发？

### 4. 证据可靠性 (evidence_reliability, 0-10)
- 每个论断是否有代码来源标注？
- 是否存在无根据的推测或编造？
- 代码引用格式是否规范？
- 如果回答中包含检索结果中不存在的代码或文件，此项应低分。

## 输出格式（严格遵守，只输出 JSON）
{
  "solution_quality": <0-10的整数>,
  "localization": <0-10的整数>,
  "completeness": <0-10的整数>,
  "evidence_reliability": <0-10的整数>,
  "reasoning": "<50字以内的简要评价>"
}

注意：
- 只输出 JSON，不要包含 ```json 或其他标记
- 分数为 0-10 的整数
- 每个维度独立评分，不要互相影响
"""

JUDGE_USER_PROMPT = """\
## 用户查询
{query}

## 期望相关文件（Ground Truth）
{expected_files}

## 系统回答
{answer}

## 检索到的代码片段（供参考判断证据可靠性）
{retrieved_context}

请严格按照评分维度评估上述回答的质量，只输出 JSON。
"""
