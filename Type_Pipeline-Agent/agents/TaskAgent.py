from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# 假设你在 BaseDeepSeekModel.py 中已经正确初始化了 model
from .BaseDeepSeekModel import model

# 1. 提示词定义
PLANNER_INSTRUCTIONS = (
    "You are a helpful research assistant. You have access to TWO sources of information: "
    "1. The public internet (Web Search). "
    "2. A private internal knowledge base containing company reports and technical whitepapers. "
    "Given a query, come up with a set of 3 to 5 searches to perform. "
    "In your 'reason', explicitly state whether this specific search targets public web information or internal company data."
)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_INSTRUCTIONS),
    ("human",  "{query}")
])

# 2. 结构化输出定义
class WebSearchItem(BaseModel):
    query: str = Field(description="The precise search term to use for the web search. 用于网络搜索的精确关键词。")
    reason: str = Field(description="Your reasoning for why this search is important to the query. 为什么这个搜索对解答该问题很重要的理由。")

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query. 为了全面回答该问题而需要执行的网页搜索列表。")

# 3. 构建 Chain
planner_chain = planner_prompt | model.with_structured_output(WebSearchPlan)

# ==========================================
# 4. 核心改造：将执行逻辑和打印封装成函数
# ==========================================
def plan_searches(query: str) -> WebSearchPlan:
    """
    接收来自外层的 query，调用大模型生成计划，并打印执行过程
    """
    print(f"\n[TaskAgent] 🧠 正在为问题生成搜索计划：'{query}'...")

    try:
        # 调用大模型
        planner_result = planner_chain.invoke({'query': query})

        # 保留你想要的精美打印输出
        print("[TaskAgent] ✅ 规划完成！搜索步骤如下：")
        print("-" * 40)
        for i, item in enumerate(planner_result.searches, 1):
            print(f"搜索 {i}: 【{item.query}】")
            print(f"理由: {item.reason}\n")

        # 将结果返回给外层（供 LangGraph 存入 State）
        return planner_result

    except Exception as e:
        print(f"[TaskAgent] ❌ 运行出错：{e}")
        raise e  # 抛出异常，让外层的图网络知道这里卡住了

# 如果你想单独测试这个文件，可以保留下面的 main 判断
# 当其他文件 import 这个模块时，下面的代码不会执行
if __name__ == "__main__":
    test_query = '请问你对AI+教育有何看法'
    plan_searches(test_query)