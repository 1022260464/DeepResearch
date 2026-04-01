from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from agents.TaskAgent import planner_chain, WebSearchPlan
from agents.SearchAgent import search_agent
from agents.WritterAgent import writer_chain, ReportData, ReportSection
# 这是一个为分层搜索和报告生成工作流，通过定义状态、节点和状态图来实现工作流，实际开发中需要进行分层架构，将不同功能模块化
# 该项目已经实现模块化，程序的主入口在main.py
# 1. 定义状态 (State)：用于在各个节点之间传递数据
class ResearchState(TypedDict):
    query: str
    search_plan: WebSearchPlan # 你的 Pydantic 模型
    search_results: List[str]
    final_report: ReportData   # 你的 Pydantic 模型

# 辅助函数：调用各个 Agent
def plan_searches(query: str) -> WebSearchPlan:
    """使用 TaskAgent 生成搜索计划"""
    return planner_chain.invoke({"query": query})

def perform_searches(search_plan: WebSearchPlan) -> List[str]:
    """使用 SearchAgent 执行搜索计划中的所有搜索项"""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    SEARCH_INSTRUCTIONS = (
        "You are a research assistant. Given a search term, you search the web for that term and "
        "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
        "words. Capture the main points. Write succinctly, no need to have complete sentences or good "
        "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
        "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
    )
    
    results = []
    for item in search_plan.searches:
        print(f"  - 正在搜索: {item.query}")
        response = search_agent.invoke({
            "messages": [
                SystemMessage(content=SEARCH_INSTRUCTIONS),
                HumanMessage(content=item.query)
            ]
        })
        results.append(response["messages"][-1].content)
    return results

def write_report(query: str, search_results: List[str]) -> ReportData:
    """使用 WritterAgent 撰写最终报告"""
    context = "\n\n".join([f"搜索结果 {i+1}:\n{result}" for i, result in enumerate(search_results)])
    return writer_chain.invoke({
        "query": query,
        "context": context
    })

# 2. 定义节点 (Nodes)：包装你的原始函数
def node_plan(state: ResearchState):
    print(">> 节点: 规划搜索")
    plan = plan_searches(state["query"])
    return {"search_plan": plan}

def node_search(state: ResearchState):
    print(f">> 节点: 执行搜索 ({len(state['search_plan'].searches)} 项)")
    results = perform_searches(state["search_plan"])
    return {"search_results": results}

def node_write(state: ResearchState):
    print(">> 节点: 撰写报告")
    report = write_report(state["query"], state["search_results"])
    return {"final_report": report}

# 3. 构建状态图 (Graph)
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("planner", node_plan)
workflow.add_node("searcher", node_search)
workflow.add_node("writer", node_write)

# 串联边 (定义流程顺序)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "writer")
workflow.add_edge("writer", END)

# 编译成可执行的 Agent
app = workflow.compile()

# ====== 运行图 ======
if __name__ == "__main__":
    # 初始状态只需传入用户的 query
    initial_state = {"query": "AI在医疗影像诊断中的应用现状"}

    # invoke 会跑完整个流程，并返回最终的状态字典
    final_state = app.invoke(initial_state)

    print("\n✅ 报告已生成！")
    report = final_state["final_report"]
    print(f"标题: {report.title}")
    print(f"摘要: {report.executive_summary}")