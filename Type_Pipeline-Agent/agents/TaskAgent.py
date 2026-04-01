from pydantic import BaseModel, Field # 引入 Field
from langchain_core.prompts import ChatPromptTemplate
# 假设你在 BaseDeepSeekModel.py 中已经正确初始化了 model
from .BaseDeepSeekModel import model

# 1. 提示词定义
PLANNER_INSTRUCTIONS = (
    "You are a helpful research assistant. Given a query, come up with a set of web searches "
    "to perform to best answer the query. Output between 3 and 5 terms to query for."
)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_INSTRUCTIONS),
    ("human",  "{query}")
])

# 2. 结构化输出定义 (使用 Field 确保 LLM 能读到描述)
class WebSearchItem(BaseModel):
    query: str = Field(
        description="The precise search term to use for the web search. 用于网络搜索的精确关键词。"
    )
    reason: str = Field(
        description="Your reasoning for why this search is important to the query. 为什么这个搜索对解答该问题很重要的理由。"
    )

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="A list of web searches to perform to best answer the query. 为了全面回答该问题而需要执行的网页搜索列表。"
    )

# 3. 构建 Chain 并绑定结构化输出
planner_chain = planner_prompt | model.with_structured_output(WebSearchPlan)

# 4. 测试运行
test_query = '请问你对AI+教育有何看法'
print(f"正在为问题生成搜索计划：'{test_query}'...\n")

try:
    planner_result = planner_chain.invoke({'query': test_query})

    # 打印生成的计划
    print("✅ 规划完成！搜索步骤如下：\n" + "-"*40)
    for i, item in enumerate(planner_result.searches, 1):
        print(f"搜索 {i}: 【{item.query}】")
        print(f"理由：{item.reason}\n")

except Exception as e:
    print(f"运行出错：{e}")
