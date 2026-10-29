"""
大语言模型（LLM）封装模块

统一从这里获取对话模型（llm）和网络搜索工具（web_search_tool）。
所有密钥和地址均在 .env 中配置，请勿在此硬编码。
"""

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    TAVILY_API_KEY,
)

# 对话模型：使用 OpenAI 兼容接口
llm = ChatOpenAI(
    temperature=0,                 # 温度设为 0，输出更稳定、可复现
    model=OPENAI_MODEL,            # 模型名称，见 .env 的 OPENAI_MODEL
    api_key=OPENAI_API_KEY,        # API Key，见 .env
    base_url=OPENAI_BASE_URL,      # 接口地址，见 .env
)

# 网络搜索工具：仅当配置了 TAVILY_API_KEY 时可用
# graph2（自适应 RAG）会把"知识库外的通用问题"路由到网络搜索
if TAVILY_API_KEY:
    web_search_tool = TavilySearchResults(
        max_results=2,
        tavily_api_key=TAVILY_API_KEY,
    )
else:
    # 未配置 Key 时为 None，网络搜索分支会在调用时给出明确报错提示
    web_search_tool = None
