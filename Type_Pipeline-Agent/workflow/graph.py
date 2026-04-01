# workflow/graph.py
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, START, END

from agents import planner_chain, search_agent, writer_chain
from agents.TaskAgent import WebSearchPlan, WebSearchItem  # 导入自定义类型以支持序列化
from agents.SearchAgent import perform_search_async  # 导入异步搜索函数
from schema.state import ResearchState

# 创建支持自定义类型的序列化器，消除 msgpack 反序列化警告
# 精确配置白名单，将 WebSearchPlan 加入允许列表
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[('agents.TaskAgent', 'WebSearchPlan')]
)

# ==========================================
# 1. 定义节点函数 (Nodes)
# ==========================================
def node_plan(state: ResearchState):
    print(">> 节点: 规划搜索")
    plan = planner_chain.invoke({"query": state["query"]})
    return {"search_plan": plan}

async def node_search(state: ResearchState):
    """
    【异步并发版本】执行多个搜索任务
    """
    # 兼容 dict 和对象两种形式
    search_plan = state['search_plan']
    if isinstance(search_plan, dict):
        searches = search_plan.get('searches', [])
        search_queries = [item.get('query', '') if isinstance(item, dict) else item.query for item in searches]
    else:
        search_queries = [item.query for item in search_plan.searches]
    print(f">> 节点: 执行并发搜索 (共 {len(search_queries)} 项)")

    # 核心魔法：创建异步任务列表
    tasks = [perform_search_async(q) for q in search_queries]

    # 同时启动所有搜索任务，并等待全部完成
    results = await asyncio.gather(*tasks)

    # 过滤掉可能的空结果或失败结果
    valid_results = [r for r in results if r and not r.startswith("搜索失败")]

    print(f">> 节点: 所有并发搜索已完成，获取到 {len(valid_results)} 份有效资料")
    return {"search_results": valid_results}

def node_write(state: ResearchState):
    print(">> 节点: 撰写报告")
    # 拼接上下文
    context = "\n\n".join([f"搜索结果 {i+1}:\n{result}" for i, result in enumerate(state["search_results"])])
    report = writer_chain.invoke({
        "query": state["query"],
        "context": context
    })
    return {"final_report": report}

# ==========================================
# 2. 构建状态图 (Graph)
# ==========================================
def create_workflow():
    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("planner", node_plan)
    workflow.add_node("searcher", node_search)
    workflow.add_node("writer", node_write)

    # 串联边
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "writer")
    workflow.add_edge("writer", END)

    # 编译并返回可执行的 Agent (App)
    # 在搜索节点前中断，允许用户审核/修改搜索计划
    # 使用 MemorySaver 启用状态检查点（支持中断恢复）
    # 传入自定义序列化器以消除 WebSearchPlan 的 msgpack 警告
    checkpointer = MemorySaver(serde=serde)
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["searcher"])

# 导出编译好的 app
app = create_workflow()