import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
# 1. 按照最新版 LangChain 警告的提示，使用专属包导入 Tavily
from langchain_tavily import TavilySearch
from .BaseDeepSeekModel import model
SEARCH_INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)
load_dotenv(override=True)
# 3. 初始化工具 (使用最新的 TavilySearch 类)
# 如果环境变量中已经有了 TAVILY_API_KEY，这里会自动读取，不需要手动传参
search_tool = TavilySearch(max_results=5, topic="general")
tools = [search_tool]

# # 4. 初始化模型
# model = ChatDeepSeek(
#     model='deepseek-chat',
# )

# 5. 构建 Agent
search_agent = create_agent(
    model,
    tools=tools
)

# 6. 测试运行
query = "介绍一下ai教学"

try:
    search_agent_res = search_agent.invoke({
        "messages": [
            SystemMessage(content=SEARCH_INSTRUCTIONS), # <-- 将你的核心指令作为系统消息传入
            HumanMessage(content=query)                 # <-- 用户的实际提问
        ]
    })

    # 打印最终结果
    print(search_agent_res["messages"][-1].content)
except Exception as e:
    print(f"运行出错: {e}")