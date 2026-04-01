import os
import asyncio
from dotenv import load_dotenv

# 🌟 核心修改 1：使用 LangGraph 的 create_react_agent 替代已废弃的 OpenAI Tools Agent
from langgraph.prebuilt import create_react_agent

# 导入工具和模型
from langchain_tavily import TavilySearch
from tools.rag_tool import local_knowledge_retriever
from .BaseDeepSeekModel import model
from tools.reader_tool import jina_reader_tool
SEARCH_INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you must search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and strictly under 300 "
    "words. Capture the main points using bullet points where appropriate. Write succinctly; there is no need "
    "for complete sentences or perfect grammar. Your output will be consumed directly by an automated report "
    "synthesizer, so it is vital you capture the raw essence, extract hard data/numbers, and ignore all fluff. "
    "CRITICAL: Output ONLY the summary. Do not include any introductory or concluding commentary."
)
load_dotenv(override=True)

# 2. 初始化工具
search_tool = TavilySearch(max_results=5, topic="general")
tools = [search_tool, local_knowledge_retriever,jina_reader_tool]

# 3. 使用 LangGraph 的 create_react_agent 构建 Agent
# noinspection PyDeprecation
search_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=SEARCH_INSTRUCTIONS
)

# ==========================================
# 5. 执行逻辑封装
# ==========================================
def perform_search(query: str) -> str:
    """
    接收来自外层的 query，调用搜索 Agent，并打印执行过程
    """
    print(f"\n[SearchAgent] 🔍 正在执行工具搜索任务：'{query}'...")

    try:
        # 使用 LangGraph 的 create_react_agent 时，传入的消息格式为列表
        search_agent_res = search_agent.invoke({
            "messages": [("human", query)]
        })

        # 获取最后一条 AI 消息的内容
        messages = search_agent_res["messages"]
        result = messages[-1].content
        print(f"[SearchAgent] ✅ 搜索完成，提取到有效信息 {len(result)} 字符")

        return result

    except Exception as e:
        print(f"[SearchAgent] ❌ 运行出错：{e}")
        raise e


async def perform_search_async(query: str) -> str:
    """
    【异步版本】接收 query，调用搜索 Agent 进行并发搜索
    """
    print(f"  [SearchAgent] 🚀 启动并发子任务：'{query}'...")
    try:
        # 使用 ainvoke 替代 invoke 进行异步调用
        search_agent_res = await search_agent.ainvoke({
            "messages": [("human", query)]
        })

        # 获取最后一条 AI 消息的内容
        messages = search_agent_res["messages"]
        result = messages[-1].content
        print(f"  [SearchAgent] ✅ 子任务完成，提取到 {len(result)} 字符")

        return result

    except Exception as e:
        print(f"  [SearchAgent] ❌ 子任务出错：{e}")
        return f"搜索失败: {str(e)}"

# 独立测试入口
if __name__ == "__main__":
    test_query = "介绍一下ai教学的发展现状，并对比一下公司内部研报的数据"
    result = perform_search(test_query)
    print("\n【最终搜索结果】\n")
    print(result)