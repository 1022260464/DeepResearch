import os
from dotenv import load_dotenv
from pydantic import BaseModel
# LangChain / LangGraph 核心依赖
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
# 模型与工具库
from langchain_deepseek import ChatDeepSeek
# 注意：构建 Agent 时，通常使用的是 TavilySearchResults 工具
from langchain_tavily import TavilySearchResults

# 加载环境变量
load_dotenv(override=True)
Deepseek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 初始化模型
model = ChatDeepSeek(
    model_name='deepseek-chat',
    api_key=Deepseek_API_KEY
)
# 提示词
PLANNER_INSTRUCTIONS = (
    "You are a helpful research assistant, Given a query, come up with a set of web searches "
    "to perform to best answer the query, Output between 5 and 7 terms to query for."
)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_INSTRUCTIONS),
    ("human",  "{query}")
])

class WebSearchItem(BaseModel):
    query: str
    "The search term to use for the web search."
    "用于网络搜索的关键词"

    reason: str
    "You reasoning for why this search is important to the query."
    "为什么这个搜索对于解答该问题很重要的理由"

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    "A list of web searches to perform to best answer the query"
    "为了尽可能全面回答该问题而需要执行的网页搜索列表"
