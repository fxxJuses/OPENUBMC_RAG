"""基于 LLM 的查询重写器，将模糊自然语言查询转为代码检索关键词。

使用 DashScope Qwen 模型将用户输入的模糊查询（如 "power control on off"）
重写为代码相关的精确关键词（如 "pwr_action power_on power_off fructrl 上电下电"）。

降级策略：LLM 调用失败时返回原始查询，确保零降级。
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_REWRITE_PROMPT = """\
将用户的代码检索查询重写为适合代码搜索引擎的关键词组合。

## 规则
- 输出 3-8 个关键词/标识符，用空格分隔，不要写完整句子
- 同时包含英文代码词（函数名、模块名、变量名）和中文语义词
- 推测可能的 snake_case 函数名（如"电源控制" → power_control pwr_action）
- 如果查询提到了组件名，保留它
- 只输出重写后的关键词，不要解释

## openUBMC 已知微组件
sensor, sensor_mgmt, devmon, vpd, frudata, fructrl, bus_tools, libipmi,
power_mgmt, bios, pcie_device, mdb_interface

## 示例
用户: "power control on off"
重写: pwr_action power_on power_off fructrl 上电下电控制 power_ctrl

用户: "fru info read write"
重写: fru_management frudata frudata_service fru_read fru_write FRU信息读写

用户: "ipmi command send receive"
重写: protocol_ipmb ipmi_send_command ipmi_receive libipmi command_execute

用户: {query}
重写:"""


class LLMQueryRewriter:
    """基于 LLM 的查询重写器，将模糊查询转为代码检索关键词。

    Attributes:
        llm: LangChain ChatOpenAI 实例
    """

    def __init__(self, api_key: str = "", model: str = "qwen-plus"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=_DASHSCOPE_BASE_URL,
            temperature=0.0,
            max_tokens=128,
        )

    def rewrite(self, query: str) -> str:
        """重写查询为代码检索关键词。

        Args:
            query: 原始查询文本

        Returns:
            重写后的关键词，失败时返回原始查询
        """
        try:
            prompt = _REWRITE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            if rewritten and len(rewritten) >= 3:
                logger.debug("Query rewritten: '%s' → '%s'", query, rewritten)
                return rewritten
            return query
        except Exception:
            logger.warning("LLM query rewrite failed, using original query")
            return query
